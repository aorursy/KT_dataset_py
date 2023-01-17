import cv2

import numpy as np

import time
# A required callback method that goes into the trackbar function.
def nothing(x):
    pass

# Initializing the webcam feed.
cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

# Create a window named trackbars.
cv2.namedWindow("Trackbars")

# Now create 6 trackbars that will control the lower and upper range of 
# H,S and V channels. The Arguments are like this: Name of trackbar, 
# window name, range,callback function. For Hue the range is 0-179 and
# for S,V its 0-255.
cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

while True:
    
    
    # Start reading the webcam feed frame by frame.
    ret, frame = cap.read()
    if not ret:
        break
    # Flip the frame horizontally (Not required)
    frame = cv2.flip( frame, 1 ) 
    
    # Convert the BGR image to HSV image.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Get the new values of the trackbar in real time as the user changes 
    # them
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")
 
    # Set the lower and upper HSV range according to the value selected
    # by the trackbar
    lower_range = np.array([l_h, l_s, l_v])
    upper_range = np.array([u_h, u_s, u_v])
    
    # Filter the image and get the binary mask, where white represents 
    # your target color
    mask = cv2.inRange(hsv, lower_range, upper_range)
 
    # You can also visualize the real part of the target color (Optional)
    res = cv2.bitwise_and(frame, frame, mask=mask)
    
    # Converting the binary mask to 3 channel image, this is just so 
    # we can stack it with the others
    mask_3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # stack the mask, orginal frame and the filtered result
    stacked = np.hstack((mask_3,frame,res))
    
    # Show this stacked frame at 40% of the size.
    cv2.imshow('Trackbars',cv2.resize(stacked,None,fx=0.4,fy=0.4))
   
    # If the user presses ESC then exit the program
    key = cv2.waitKey(1)
    if key == 27:
        break
    
    # If the user presses `s` then print this array.
    if key == ord('s'):
        
        thearray = [[l_h,l_s,l_v],[u_h, u_s, u_v]]
        print(thearray)
        
        # Also save this array as penval.npy
        np.save('penval',thearray)
        break
    
# Release the camera &amp; destroy the windows.    
cap.release()
cv2.destroyAllWindows()
# This variable determines if we want to load color range from memory 
# or use the ones defined here. 
# This variable determines if we want to load color range from memory 
# or use the ones defined here. 
load_from_disk = True
import numpy as np
import cv2
# If true then load color range from memory
if load_from_disk:
    penval = np.load('penval.npy')

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

# Creating A 5x5 kernel for morphological operations
kernel = np.ones((5,5),np.uint8)
while(1):
    
    ret, frame = cap.read()
    if not ret:
        break
        
    frame = cv2.flip( frame, 1 )

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # If you're reading from memory then load the upper and lower ranges 
    # from there
    if load_from_disk:
            lower_range = penval[0]
            upper_range = penval[1]
            
    # Otherwise define your own custom values for upper and lower range.
    else:             
       lower_range  = np.array([26,80,147])
       upper_range = np.array([81,255,255])
    
    mask = cv2.inRange(hsv, lower_range, upper_range)
    
    # Perform the morphological operations to get rid of the noise.
    # Erosion Eats away the white part while dilation expands it.
    mask = cv2.erode(mask,kernel,iterations = 1)
    mask = cv2.dilate(mask,kernel,iterations = 2)

    res = cv2.bitwise_and(frame,frame, mask= mask)

    mask_3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # stack all frames and show it
    stacked = np.hstack((mask_3,frame,res))
    cv2.imshow('Trackbars',cv2.resize(stacked,None,fx=0.4,fy=0.4))
    
    k = cv2.waitKey(5)  ;0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
# This variable determines if we want to load color range from memory 
# or use the ones defined in the notebook. 
load_from_disk = True
import numpy as np
import cv2
# If true then load color range from memory
if load_from_disk:
    penval = np.load('penval.npy')

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

# kernel for morphological operations
kernel = np.ones((5,5),np.uint8)

# set the window to auto-size so we can view this full screen.
cv2.namedWindow('image', cv2.WINDOW_NORMAL)

# This threshold is used to filter noise, the contour area must be 
# bigger than this to qualify as an actual contour.
noiseth = 500
while(1):
    
    _, frame = cap.read()
    frame = cv2.flip( frame, 1 )

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # If you're reading from memory then load the upper and lower 
    # ranges from there
    if load_from_disk:
            lower_range = penval[0]
            upper_range = penval[1]
            
    # Otherwise define your own custom values for upper and lower range.
    else:             
       lower_range  = np.array([26,80,147])
       upper_range = np.array([81,255,255])
    
    mask = cv2.inRange(hsv, lower_range, upper_range)
    
    # Perform the morphological operations to get rid of the noise
    mask = cv2.erode(mask,kernel,iterations = 1)
    mask = cv2.dilate(mask,kernel,iterations = 2)
    
    # Find Contours in the frame.
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
    
    # Make sure there is a contour present and also make sure its size 
    # is bigger than noise threshold.
    if contours and cv2.contourArea(max(contours, 
                               key = cv2.contourArea)) > noiseth:
        
        # Grab the biggest contour with respect to area
        c = max(contours, key = cv2.contourArea)
        
        # Get bounding box coordinates around that contour
        x,y,w,h = cv2.boundingRect(c)
        
        # Draw that bounding box
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,25,255),2)        

    cv2.imshow('image',frame)
    
    k = cv2.waitKey(5); 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
import numpy as np
import cv2
load_from_disk = True
if load_from_disk:
    penval = np.load('penval.npy')

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

kernel = np.ones((5,5),np.uint8)

# Initializing the canvas on which we will draw upon
canvas = None

# Initilize x1,y1 points
x1,y1=0,0

# Threshold for noise
noiseth = 800
while(1):
    _, frame = cap.read()
    frame = cv2.flip( frame, 1 )
    
    # Initialize the canvas as a black image of the same size as the frame.
    if canvas is None:
        canvas = np.zeros_like(frame)

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # If you're reading from memory then load the upper and lower ranges 
    # from there
    if load_from_disk:
            lower_range = penval[0]
            upper_range = penval[1]
            
    # Otherwise define your own custom values for upper and lower range.
    else:             
       lower_range  = np.array([26,80,147])
       upper_range = np.array([81,255,255])
    
    mask = cv2.inRange(hsv, lower_range, upper_range)
    
    # Perform morphological operations to get rid of the noise
    mask = cv2.erode(mask,kernel,iterations = 1)
    mask = cv2.dilate(mask,kernel,iterations = 2)
    
    # Find Contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Make sure there is a contour present and also its size is bigger than 
    # the noise threshold.
    if contours and cv2.contourArea(max(contours, 
                                 key = cv2.contourArea)) > noiseth:
                
        c = max(contours, key = cv2.contourArea)    
        x2,y2,w,h = cv2.boundingRect(c)
        
        # If there were no previous points then save the detected x2,y2 
        # coordinates as x1,y1. 
        # This is true when we writing for the first time or when writing 
        # again when the pen had disappeared from view.
        if x1 == 0 and y1 == 0:
            x1,y1= x2,y2
            
        else:
            # Draw the line on the canvas
            canvas = cv2.line(canvas, (x1,y1),(x2,y2), [255,0,0], 4)
        
        # After the line is drawn the new points become the previous points.
        x1,y1= x2,y2

    else:
        # If there were no contours detected then make x1,y1 = 0
        x1,y1 =0,0
    
    # Merge the canvas and the frame.
    frame = cv2.add(frame,canvas)
    
    # Optionally stack both frames and show it.
    stacked = np.hstack((canvas,frame))
    cv2.imshow('Trackbars',cv2.resize(stacked,None,fx=0.6,fy=0.6))

    k = cv2.waitKey(1) ; 0xFF
    if k == 27:
        break
        
    # When c is pressed clear the canvas
    if k == ord('c'):
        canvas = None

cv2.destroyAllWindows()
cap.release()
import cv2
import numpy as np
import time
load_from_disk = True
if load_from_disk:
    penval = np.load('penval.npy')

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

kernel = np.ones((5,5),np.uint8)

# Making window size adjustable
cv2.namedWindow('image', cv2.WINDOW_NORMAL)

# This is the canvas on which we will draw upon
canvas=None

# Initilize x1,y1 points
x1,y1=0,0

# Threshold for noise
noiseth = 800

# Threshold for wiper, the size of the contour must be bigger than for us to
# clear the canvas 
wiper_thresh = 40000

# A variable which tells when to clear canvas, if its True then we clear the canvas
clear = False
while(1):
    _, frame = cap.read()
    frame = cv2.flip( frame, 1 )
    
    # Initialize the canvas as a black image
    if canvas is None:
        canvas = np.zeros_like(frame)

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # If you're reading from memory then load the upper and lower ranges 
    # from there
    if load_from_disk:
            lower_range = penval[0]
            upper_range = penval[1]
            
    # Otherwise define your own custom values for upper and lower range.
    else:             
       lower_range  = np.array([26,80,147])
       upper_range = np.array([81,255,255])
    
    mask = cv2.inRange(hsv, lower_range, upper_range)
    
    # Perform the morphological operations to get rid of the noise
    mask = cv2.erode(mask,kernel,iterations = 1)
    mask = cv2.dilate(mask,kernel,iterations = 2)

    # Find Contours.
    contours, hierarchy = cv2.findContours(mask,
    cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    # Make sure there is a contour present and also its size is bigger than 
    # the noise threshold.
    if contours and cv2.contourArea(max(contours,
                                   key = cv2.contourArea)) > noiseth:
                
        c = max(contours, key = cv2.contourArea)    
        x2,y2,w,h = cv2.boundingRect(c)
        
        # Get the area of the contour
        area = cv2.contourArea(c)
        
        # If there were no previous points then save the detected x2,y2 
        # coordinates as x1,y1. 
        if x1 == 0 and y1 == 0:
            x1,y1= x2,y2
            
        else:
            # Draw the line on the canvas
            canvas = cv2.line(canvas, (x1,y1),(x2,y2),
            [255,0,0], 5)
        
        # After the line is drawn the new points become the previous points.
        x1,y1= x2,y2
        
        # Now if the area is greater than the wiper threshold then set the  
        # clear variable to True and warn User.
        if area > wiper_thresh:
           cv2.putText(canvas,'Clearing Canvas', (100,200), 
           cv2.FONT_HERSHEY_SIMPLEX,2, (0,0,255), 5, cv2.LINE_AA)
           clear = True 

    else:
        # If there were no contours detected then make x1,y1 = 0
        x1,y1 =0,0
    
   
    # Now this piece of code is just for smooth drawing. (Optional)
    _ , mask = cv2.threshold(cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY), 20, 
    255, cv2.THRESH_BINARY)
    foreground = cv2.bitwise_and(canvas, canvas, mask = mask)
    background = cv2.bitwise_and(frame, frame,
    mask = cv2.bitwise_not(mask))
    frame = cv2.add(foreground,background)

    cv2.imshow('image',frame)
    
    k = cv2.waitKey(5) ; 0xFF
    if k == 27:
        break
    
    # Clear the canvas after 1 second if the clear variable is true
    if clear == True:
        
        time.sleep(1)
        canvas = None
        
        # And then set clear to false
        clear = False
        
cv2.destroyAllWindows()
cap.release()
import numpy as np
import cv2
import time
load_from_disk = True
if load_from_disk:
    penval = np.load('penval.npy')

cap = cv2.VideoCapture(0)

# Load these 2 images and resize them to the same size.
pen_img = cv2.resize(cv2.imread('pen.jpg',1), (50, 50))
eraser_img = cv2.resize(cv2.imread('download.jpg',1), (50, 50))

kernel = np.ones((5,5),np.uint8)

# Making window size adjustable
cv2.namedWindow('image', cv2.WINDOW_NORMAL)

# This is the canvas on which we will draw upon
canvas = None

# Create a background subtractor Object
backgroundobject = cv2.createBackgroundSubtractorMOG2(detectShadows = False)

# This threshold determines the amount of disruption in the background.
background_threshold = 600

# A variable which tells you if you're using a pen or an eraser.
switch = 'Pen'

# With this variable we will monitor the time between previous switch.
last_switch = time.time()

# Initilize x1,y1 points
x1,y1=0,0

# Threshold for noise
noiseth = 800
# Threshold for wiper, the size of the contour must be bigger than this for # us to clear the canvas
wiper_thresh = 40000

# A variable which tells when to clear canvas
clear = False

while(1):
    _, frame = cap.read()
    frame = cv2.flip( frame, 1 )
    
    # Initilize the canvas as a black image
    if canvas is None:
        canvas = np.zeros_like(frame)
        
    # Take the top left of the frame and apply the background subtractor
    # there    
    top_left = frame[0: 50, 0: 50]
    fgmask = backgroundobject.apply(top_left)
    
    # Note the number of pixels that are white, this is the level of 
    # disruption.
    switch_thresh = np.sum(fgmask==255)
    
    # If the disruption is greater than background threshold and there has 
    # been some time after the previous switch then you. can change the 
    # object type.
    if switch_thresh>background_threshold and (time.time()-last_switch) > 1:

        # Save the time of the switch. 
        last_switch = time.time()
        
        if switch == 'Pen':
            switch = 'Eraser'
        else:
            switch = 'Pen'

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # If you're reading from memory then load the upper and lower ranges 
    # from there
    if load_from_disk:
            lower_range = penval[0]
            upper_range = penval[1]
            
    # Otherwise define your own custom values for upper and lower range.
    else:             
       lower_range  = np.array([26,80,147])
       upper_range = np.array([81,255,255])
    
    mask = cv2.inRange(hsv, lower_range, upper_range)
    
    # Perform morphological operations to get rid of the noise
    mask = cv2.erode(mask,kernel,iterations = 1)
    mask = cv2.dilate(mask,kernel,iterations = 2)
    
    # Find Contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
    cv2.CHAIN_APPROX_SIMPLE)
    
    # Make sure there is a contour present and also it size is bigger than 
    # noise threshold.
    if contours and cv2.contourArea(max(contours,
                                      key = cv2.contourArea)) > noiseth:
                
        c = max(contours, key = cv2.contourArea)    
        x2,y2,w,h = cv2.boundingRect(c)
        
        # Get the area of the contour
        area = cv2.contourArea(c)
        
        # If there were no previous points then save the detected x2,y2 
        # coordinates as x1,y1. 
        if x1 == 0 and y1 == 0:
            x1,y1= x2,y2
            
        else:
            if switch == 'Pen':
                # Draw the line on the canvas
                canvas = cv2.line(canvas, (x1,y1),
                (x2,y2), [255,0,0], 5)
                
            else:
                cv2.circle(canvas, (x2, y2), 20,
                (0,0,0), -1)
            
            
        
        # After the line is drawn the new points become the previous points.
        x1,y1= x2,y2
        
        # Now if the area is greater than the wiper threshold then set the 
        # clear variable to True
        if area > wiper_thresh:
           cv2.putText(canvas,'Clearing Canvas',(0,200), 
           cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 1, cv2.LINE_AA)
           clear = True 

    else:
        # If there were no contours detected then make x1,y1 = 0
        x1,y1 =0,0
    
   
    # Now this piece of code is just for smooth drawing. (Optional)
    _ , mask = cv2.threshold(cv2.cvtColor (canvas, cv2.COLOR_BGR2GRAY), 20, 
    255, cv2.THRESH_BINARY)
    foreground = cv2.bitwise_and(canvas, canvas, mask = mask)
    background = cv2.bitwise_and(frame, frame,
    mask = cv2.bitwise_not(mask))
    frame = cv2.add(foreground,background)

    # Switch the images depending upon what we're using, pen or eraser.
    if switch != 'Pen':
        cv2.circle(frame, (x1, y1), 20, (255,255,255), -1)
        frame[0: 50, 0: 50] = eraser_img
    else:
        frame[0: 50, 0: 50] = pen_img

    cv2.imshow('image',frame)

    k = cv2.waitKey(5); 0xFF
    if k == 27:
        break
    
    # Clear the canvas after 1 second, if the clear variable is true
    if clear == True: 
        time.sleep(1)
        canvas = None
        
        # And then set clear to false
        clear = False
        
cv2.destroyAllWindows()
cap.release()