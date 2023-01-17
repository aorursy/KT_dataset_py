import cv2

from matplotlib import pyplot as plt

import numpy as np 

import pandas as pd 





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

img = cv2.imread('/kaggle/input/cutlery/cutlery.jpg')

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edgeCanny = cv2.Canny(img_gray, 100, 200)
sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1,0, ksize=5)

sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0,1, ksize=5)
plt.subplot(2,2,1), plt.imshow(img_gray,cmap = 'gray')

plt.title('Original'), plt.xticks([]), plt.yticks([])

plt.subplot(2,2,2), plt.imshow(edgeCanny,cmap = 'gray')

plt.title('Canny'), plt.xticks([]), plt.yticks([])

plt.subplot(2,2,3), plt.imshow(sobelx,cmap = 'gray')

plt.title('Sobel X'), plt.xticks([]), plt.yticks([])

plt.subplot(2,2,4), plt.imshow (sobely, cmap = 'gray')

plt.title('Sobel y'), plt.xticks([]), plt.yticks([])



plt.show()

gaussian_blur = np.hstack([

    cv2.GaussianBlur(img,(3,3),0),

    cv2.GaussianBlur(img,(5,5),0),

    cv2.GaussianBlur(img,(9,9),0),

])



plt.imshow(gaussian_blur,cmap = 'gray')

plt.title('Gaussian')
blur = np.hstack ([

    cv2.blur(img,(3,3)),

    cv2.blur(img,(5,5)),

    cv2.blur(img,(9,9))

])



plt.imshow(blur, cmap = 'gray')

plt.title('Blur')
filter = np.array([[-1, -1, -1],[-1, 9, -1],[-1, -1, -1]])

sharpen_img = cv2.filter2D(img, -1, filter)

plt.imshow(sharpen_img, cmap = 'gray')

plt.title('Sharpen')