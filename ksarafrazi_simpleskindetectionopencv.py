import numpy as np
import cv2
import matplotlib.pyplot as plt

#Reading Image
img = cv2.imread('../input/faces.jpeg',1)

#Creating new figure
plt.figure(figsize=(8,8))
#Displaying image
#Converting OpenCV BRG to RGB first
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
#Converting to HSV color mode
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h = hsv[:,:,0]
s = hsv[:,:,1]
v = hsv[:,:,2]

hsv_split = np.concatenate((h,s,v), axis=1)

#Displaying HSV channels
plt.figure(figsize=(15,5))
plt.imshow(hsv_split, cmap='gray')
plt.axis('off')

#Creating a binary threshold on S channel
ret, min_sat = cv2.threshold(s,40,255, cv2.THRESH_BINARY)

plt.figure(figsize=(8,8))
plt.imshow(min_sat, cmap='gray')
plt.axis('off')
#Creating an inverse binary threshold on H channel
ret, max_hue = cv2.threshold(h,15, 255, cv2.THRESH_BINARY_INV)

plt.figure(figsize=(8,8))
plt.imshow(max_hue, cmap='gray')
plt.axis('off')
#Combining the two filters 
final = cv2.bitwise_and(min_sat,max_hue)

fig, ax = plt.subplots(1,2,figsize=(10,5))

ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax[0].axis('off')
ax[1].imshow(final, cmap='gray')
ax[1].axis('off')


