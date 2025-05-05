import cv2
import numpy as np
import requests
from io import BytesIO

# ESP32-CAM IP address
ESP32_IP = "http://192.168.79.69"  # Change this to your ESP32's IP

# Function to send movement command to ESP32
def move(direction):
    url = f"{ESP32_IP}/action?go={direction}"
    response = requests.get(url)
    if response.status_code == 200:
        print(f"Moved {direction}")
    else:
        print(f"Failed to move {direction}")

# Function to detect red object in the frame
def detect_red_object(frame):
    # Convert frame to HSV (Hue, Saturation, Value)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define the range of red color in HSV
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    
    # Create a mask for red areas
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    
    lower_red = np.array([170, 120, 70])
    upper_red = np.array([180, 255, 255])
    
    # Another mask for the other range of red
    mask2 = cv2.inRange(hsv, lower_red, upper_red)
    
    # Combine the two masks
    mask = mask1 | mask2
    
    # Bitwise-AND the mask and the frame to extract the red regions
    red_object = cv2.bitwise_and(frame, frame, mask=mask)
    
    return red_object, mask

# Function to find the center of the red object and send movement commands
def track_red_object(frame):
    red_object, mask = detect_red_object(frame)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get the largest contour (assumed to be the red object)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get the bounding box of the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Get the center of the object
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Draw the center and the bounding box on the frame
        cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Calculate the movement direction
        frame_center_x = frame.shape[1] // 2
        frame_center_y = frame.shape[0] // 2
        
        # Move the camera to center the red object
        if center_x < frame_center_x - 50:
            move("left")
        elif center_x > frame_center_x + 50:
            move("right")
        
        if center_y < frame_center_y - 50:
            move("down")
        elif center_y > frame_center_y + 50:
            move("up")
    else:
        print("No red object detected")

# Open the video stream from ESP32-CAM using requests
def get_frame_from_stream(url):
    response = requests.get(url, stream=True)
    bytes_data = b''

    for chunk in response.iter_content(chunk_size=1024):
        bytes_data += chunk
        # Check if we have a full frame
        a = bytes_data.find(b'\xff\xd8')  # JPEG start
        b = bytes_data.find(b'\xff\xd9')  # JPEG end
        
        if a != -1 and b != -1:
            jpg = bytes_data[a:b+2]
            bytes_data = bytes_data[b+2:]
            
            # Convert JPEG to OpenCV image
            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            return frame

    return None

# Main loop
stream_url = f"{ESP32_IP}:81/stream"

while True:
    frame = get_frame_from_stream(stream_url)
    
    if frame is not None:
        # Track the red object and send movement commands
        track_red_object(frame)
        
        # Display the frame with red object detection
        cv2.imshow("ESP32-CAM Stream", frame)
        
        # Break the loop if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("Failed to get frame from stream")

# Release resources
cv2.destroyAllWindows()

