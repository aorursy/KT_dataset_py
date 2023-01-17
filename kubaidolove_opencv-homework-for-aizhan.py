import cv2

import numpy as np

from matplotlib import pyplot as plt

%matplotlib inline

import imutils

import zipfile
#Load image and convert to grey.

img = cv2.imread('../input/samples/samples/2cegf.png', 0)

plt.imshow(img, 'gray')
#Resize

img_resized = cv2.resize(img,(360,480))



plt.imshow(img_resized, 'gray')
#Rotate

(h, w) = img.shape[:2]

# calculate the center of the image

center = (w / 2, h / 2)

 

angle90 = 90

angle180 = 180

angle270 = 270

 

scale = 1.0



M = cv2.getRotationMatrix2D(center, angle180, scale)

rotated180 = cv2.warpAffine(img, M, (w, h))



plt.imshow(rotated180, 'gray')
#Transform

rows,cols = img.shape

pts1 = np.float32([[50,50],[200,50],[50,200]])

pts2 = np.float32([[10,100],[200,50],[100,250]])

Mat = cv2.getAffineTransform(pts1,pts2)

dst = cv2.warpAffine(img,Mat,(cols,rows))

plt.subplot(121),plt.imshow(img),plt.title('Input')

plt.subplot(122),plt.imshow(dst),plt.title('Output')

plt.show()
#saving (wont save though, just an example of imwrite)

cv2.imwrite('/home/img/rotated180.jpg',rotated180)