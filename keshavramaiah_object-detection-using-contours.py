import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from keras.layers.core import Activation,Dense,Dropout

import cv2

img=cv2.imread('../input/bot.tiff',1)

img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

plt.imshow(img)

plt.xticks([])

plt.yticks([])

plt.title('water bottle')

plt.show()


imgg=img

imgg=cv2.GaussianBlur(imgg,(5,5),0)

imgg=cv2.pyrMeanShiftFiltering(imgg,131,201)

gray=cv2.cvtColor(imgg,cv2.COLOR_RGB2GRAY)

ret,thresh=cv2.threshold(gray,240,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

k=np.ones((5,5),np.uint8)

erosion=cv2.erode(thresh,k,iterations=11)

plt.imshow(erosion)

plt.show()

img2,contours=cv2.findContours(erosion,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)

#print(type(img2))

for con in img2:

    episilon=0.01*cv2.arcLength(con,True)

    approx=cv2.approxPolyDP(con,episilon,True)

    c=cv2.drawContours(img,[approx],0,(255,0,0),10)

    plt.imshow(c)

    plt.title('Contours in water bottle')

    plt.xticks([])

    plt.yticks([])

    plt.show()

print("Number of Contours:",len(img2))




