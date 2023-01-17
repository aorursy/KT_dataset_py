import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob, os
    
image = cv2.imread("../input/train_sm/set160_1.jpeg")
plt.figure(figsize=(15,20))
plt.imshow(image)
plt.show()


# load the image and convert it to grayscale
image = cv2.imread("../input/train_sm/set160_1.jpeg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
# initialize the AKAZE descriptor, then detect keypoints and extract
# local invariant descriptors from the image
detector = cv2.AKAZE_create()
(kps, descs) = detector.detectAndCompute(gray, None)
print("keypoints: {}, descriptors: {}".format(len(kps), descs.shape))
 
# draw the keypoints and show the output image
cv2.drawKeypoints(image, kps, image, (0, 255, 0))
plt.figure(figsize=(15,20))
plt.imshow(image)
plt.show()