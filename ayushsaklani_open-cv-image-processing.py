import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt 

import cv2



import os

print(os.listdir("../input"))



%matplotlib inline

img = cv2.imread('../input/cars_train/cars_train/00044.jpg')
plt.imshow(img)
im = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

plt.imshow(im)
type(im)
im.shape
fig ,axes = plt.subplots(3,1, figsize=(40,25))

axe = [(i,j) for i in range(3) for j in range(1)]

for ax in range(3):

    axes[ax].imshow(im[:,:,ax])
img2 = im[150:400,150:550]

plt.imshow(img2)
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)



# remove noise

img = cv2.GaussianBlur(gray,(3,3),0)



# convolute with proper kernels

laplacian = cv2.Laplacian(img,cv2.CV_64F)

sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)  # x

sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)  # y



####

plt.figure(figsize=(20,15))

plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')

plt.title('Original'), plt.xticks([]), plt.yticks([])

plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')

plt.title('Laplacian'), plt.xticks([]), plt.yticks([])

plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')

plt.title('Sobel X'), plt.xticks([]), plt.yticks([])

plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')

plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])



plt.show()


