# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import cv2

import matplotlib.pyplot as plt
img = cv2.imread("../input/lena_color_512.tif",0)

plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

plt.show()
hsv_img=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

plt.imshow(hsv_img)

plt.show()
img1=hsv_img.copy()

value=hsv_img[:,:,2]

plt.imshow(value)

plt.show()
equ=cv2.equalizeHist(value)

plt.imshow(equ)

plt.show()
img2=cv2.merge((img1[:,:,0],img1[:,:,1],equ))

plt.imshow(img2)

plt.show()
plt.imshow(cv2.cvtColor(img,cv2.COLOR_HSV2BGR))

plt.show()
kernel=np.ones((5,5),np.uint8)
erosion=cv2.erode(img,kernel,iterations=2)
plt.imshow(erosion,cmap="gray")

plt.show()
dialation=cv2.dilate(img,kernel,iterations=2)

plt.imshow(dialation,cmap="gray")

plt.show()
opening=cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)

plt.imshow(opening,cmap="gray")

plt.show()
closing=cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)

plt.imshow(opening,cmap="gray")

plt.show()