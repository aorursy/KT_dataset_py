import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

import matplotlib.pyplot as plt
#image = cv2.imread('/kaggle/input/operations-with-opencv/1Trump.jpg')

image = cv2.imread('/kaggle/input/opencv-samples-images/someshapes.jpg')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

plt.imshow(image)
#grayscale

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# Find Canny edges

edged = cv2.Canny(gray, 30, 200)

plt.imshow(edged)
# Finding Contours

contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

plt.imshow(edged)
print("Number of Contours found = " + str(len(contours)))
cv2.drawContours(image, contours, -1, (0,255,0), 3)



plt.imshow(image)
# Load our image

image = cv2.imread('/kaggle/input/operations-with-opencv/1bunchofshapes.jpg')

plt.imshow( image)
# Create a black image with same dimensions as our loaded image

blank_image = np.zeros((image.shape[0], image.shape[1], 3))
# Create a copy of our original image

orginal_image = image
# Grayscale our image

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# Find Canny edges

edged = cv2.Canny(gray, 50, 200)

plt.imshow(edged)
# Find contours and print how many were found

contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

print ("Number of contours found = ", len(contours))
#Draw all contours

cv2.drawContours(blank_image, contours, -1, (0,255,0), 3)

plt.imshow(blank_image)
# Draw all contours over blank image

cv2.drawContours(image, contours, -1, (0,255,0), 3)

plt.imshow(image)
# Draw all contours over blank image

cv2.drawContours(image, contours, -1, (15,8,12), 3)

plt.imshow(image)
# Load our image

image = cv2.imread('/kaggle/input/opencv-samples-images/house.jpg')

plt.imshow( image)
orig_image = image.copy()

plt.imshow( orig_image)
# Grayscale and binarize

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
# Find contours 

contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
# Iterate through each contour and compute the bounding rectangle

for c in contours:

    x,y,w,h = cv2.boundingRect(c)

    cv2.rectangle(orig_image,(x,y),(x+w,y+h),(0,0,255),2)    

    plt.imshow(orig_image)

# Iterate through each contour and compute the approx contour

for c in contours:

    # Calculate accuracy as a percent of the contour perimeter

    accuracy = 0.03 * cv2.arcLength(c, True)

    approx = cv2.approxPolyDP(c, accuracy, True)

    cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)

    plt.imshow( image)
# Load our image

image = cv2.imread('/kaggle/input/opencv-samples-images/hand.jpg')

plt.imshow( image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)



plt.imshow( image)
# Threshold the image

ret, thresh = cv2.threshold(gray, 176, 255, 0)
# Find contours 

contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
# Sort Contors by area and then remove the largest frame contour

n = len(contours) - 1

contours = sorted(contours, key=cv2.contourArea, reverse=False)[:n]

# Iterate through contours and draw the convex hull

for c in contours:

    hull = cv2.convexHull(c)

    cv2.drawContours(image, [hull], 0, (0, 255, 0), 2)

    plt.imshow(image)

# Load our image

image = cv2.imread('/kaggle/input/operations-with-opencv/1chess.jpg')

plt.imshow(image)
# Grayscale and Canny Edges extracted

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 100, 170, apertureSize = 3)
plt.imshow(edges)

lines = cv2.HoughLines(edges, 1, np.pi/180, 200)

# We iterate through each line and convert it to the format

# required by cv.lines (i.e. requiring end points)

for line in lines:

    rho, theta = line[0]

    a = np.cos(theta)

    b = np.sin(theta)

    x0 = a * rho

    y0 = b * rho

    x1 = int(x0 + 1000 * (-b))

    y1 = int(y0 + 1000 * (a))

    x2 = int(x0 - 1000 * (-b))

    y2 = int(y0 - 1000 * (a))

    cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
plt.imshow(image)

### Probabilistic Hough Lines
# Grayscale and Canny Edges extracted

image = cv2.imread('/kaggle/input/operations-with-opencv/1chess.jpg')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 100, 170, apertureSize = 3)
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 200, 5, 10)

print(lines.shape)
for line in lines:

    for x1,y1,x2,y2 in line:

        cv2.line(image, (x1, y1), (x2, y2),(0, 255, 0), 3)

plt.imshow(image)
image = cv2.imread('/kaggle/input/operations-with-opencv/1bottlecaps.jpg')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.imshow(gray)
blur = cv2.medianBlur(gray, 5)



circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1.5, 10,

                           param1=100,param2=100,minRadius=25,maxRadius=60)



circles = np.uint16(np.around(circles))
for i in circles[0,:]:

    # draw the outer circle

    cv2.circle(image,(i[0], i[1]), i[2], (255, 0, 0), 2)

    

    # draw the center of the circle

    cv2.circle(image, (i[0], i[1]), 2, (0, 255, 0), 5)
plt.imshow( image)
image = cv2.imread('/kaggle/input/operations-with-opencv/1Sunflowers.jpg')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image)
# Set up the detector with default parameters.

detector = cv2.SimpleBlobDetector_create()
# Detect blobs.

keypoints = detector.detect(image)
# Draw detected blobs as red circles.

# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of

# the circle corresponds to the size of blob

blank = np.zeros((1,1)) 

blobs = cv2.drawKeypoints(image, keypoints, blank, (13,15,170),

                                      cv2.DRAW_MATCHES_FLAGS_DEFAULT)
plt.imshow(blobs)