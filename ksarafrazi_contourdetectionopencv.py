import numpy as np
import cv2
import matplotlib.pyplot as plt

#Reading image
img = cv2.imread('../input/detect_blob.png',1)

#Converting RGB image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

#Applying adaptive thresholding
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)

fig, ax = plt.subplots(1,2,figsize=(10,5))

ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax[0].axis('off')
ax[1].imshow(thresh, cmap='gray')
ax[1].axis('off')

#Extracting contours
_, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

img2 = img.copy()
index = -1
thickness = 4
color = (255, 0, 255)

#Adding contours to image
cv2.drawContours(img2, contours, index, color, thickness)

fig, ax = plt.subplots(1,2,figsize=(10,5))

ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax[0].axis('off')
ax[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
ax[1].axis('off')
#Claculating each object's center 
objects = np.zeros([img.shape[0], img.shape[1],3], 'uint8')
for c in contours:
    cv2.drawContours(objects, [c], -1, color, -1)

    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)

    M = cv2.moments(c)
    cx = int( M['m10']/M['m00'])
    cy = int( M['m01']/M['m00'])
    cv2.circle(objects, (cx,cy), 4, (0,0,255), -1)
    
plt.figure(figsize=(5,5))
plt.imshow(objects)
plt.axis('off')
