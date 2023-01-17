import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import cv2

import matplotlib.pyplot as plt
img=cv2.imread('../input/640_640x480.jpg')

img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img)
#Image Rotation

r,c=img.shape[:2]

M=cv2.getRotationMatrix2D((c/2,r/2),180,1)

dst=cv2.warpAffine(img,M,(c,r))

plt.imshow(dst)
#Translation

M=np.float32([[1,0,-100],[0,1,-100]])

dst=cv2.warpAffine(img,M,(c,r))

plt.imshow(dst)
img_gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

t_global=cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)

plt.imshow(t_global)
#Edge detection

edges=cv2.Canny(img,200,200)

plt.imshow(edges)
#Contours

gray_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

ret,thresh=cv2.threshold(gray_image,127,255,0)

contours,h=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

with_contours = cv2.drawContours(img,contours,-1,(0,255,0),3) 

plt.imshow(with_contours)
#ReRead Image

img=cv2.imread('../input/640_640x480.jpg')

img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#Constrast

img_weighted=cv2.addWeighted(img,1.5,np.zeros(img.shape, img.dtype), 0, 0)

plt.imshow(img_weighted)
#Blur

img_blur=cv2.GaussianBlur(img,(7,7),0)

plt.imshow(img_blur)
#Blur

img_blur=cv2.medianBlur(img,7)

plt.imshow(img_blur)
#GrayScale

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.imshow(gray_img,cmap='gray')
#Contours

retval, thresh = cv2.threshold(gray_img, 127, 255, 0)

img_contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img, img_contours, -1, (0, 255, 0))

plt.imshow(img)