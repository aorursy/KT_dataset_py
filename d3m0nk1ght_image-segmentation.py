# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import cv2

#import os

#print(os.listdir("../input/cameraman"))

lt=50

ht=100

img=cv2.imread('../input/mobile/mobile_hough.jpg')

img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

imgr=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

edges=cv2.Canny(img,50,100)



plt.imshow(edges,cmap='gray')

plt.show()

lines=cv2.HoughLinesP(edges,1,np.pi/180,60,np.array([]),50,5)

line_img=np.copy(img)



for line in lines:

    for x1,y1,x2,y2 in line:

        cv2.line(line_img,(x1,y1),(x2,y2),(255,0,0),3)

plt.imshow(line_img)

plt.show()



#kernel=np.ones((3,3),np.uint8)

imgr=np.float32(imgr)

ch=cv2.cornerHarris(imgr,2,3,0.04)

ch=cv2.dilate(ch,None)

plt.imshow(ch,cmap='gray')

# Any results you write to the current directory are saved as output.

thresh=0.1*ch.max()

img[ch>0.01*ch.max()]=[0,255,0]

plt.imshow(img)

            

    