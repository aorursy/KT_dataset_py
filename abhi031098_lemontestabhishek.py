# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/lemons'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from math import sqrt
import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
unknown = []
unknown1 = []
boxes = []
img = []
import os
path='../input/coding-round-images/'

ldseg=np.array(os.listdir(path))
#Reading Images
for filename in ldseg:
  img.append(plt.imread(path+filename))

thres_list =[]
#Stroring Threshold of each image
for i in img:
  gray = cv2.cvtColor(i,cv2.COLOR_BGR2GRAY)
  ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
  thres_list.append(thresh)

def distance(a,b):
  return int(sqrt((a[0]-b[0])**2+(a[1]-b[1])**2))

#Testing for various values of i1,i2,x
for i1 in [2]:
  for i2 in [10]:
    for x in [0.01]:
      i=0
      for thresh in thres_list:
        kernel = np.ones((3,3),np.uint8)
        thresh = ~thresh
        opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = i1)
        # sure background area
        sure_bg = cv2.dilate(opening,kernel,iterations=i2)
        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,3)
        ret, sure_fg = cv2.threshold(dist_transform,x*dist_transform.max(),255,0)
        # Finding unknown region
        sure_fg = np.uint8(sure_fg)

        thresh_gray = cv2.subtract(sure_bg,sure_fg)

        contours, hier = cv2.findContours(sure_fg, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in contours:
          rect = cv2.minAreaRect(c)
          box = cv2.boxPoints(rect)
          #p0 is lowest
          p0 = box[0]
          p1 = box[1]
          p2 = box[2]
          p3 = box[3]

          x_1 = distance(p0,p1)
          y_1 = distance(p0,p3)
          

          #p3 l > w?
          if(x_1>y_1):
            l,w = 1,3
          else:
            l,w = 3,1
            y_1,x_1 = x_1,y_1



          # l>w
          if(x_1>1.5*y_1):

            boxnew1 = box.copy()
            boxnew2 = box.copy()
            boxnew1[l] = [(box[l][0]+box[0][0])//2-2,(box[l][1]+box[0][1])//2-2 ]
            boxnew1[2] = [(box[w][0]+box[2][0])//2 -2,(box[w][1]+box[2][1])//2-2 ]
            boxnew2[0] = [(box[l][0]+box[0][0])//2+2,(box[l][1]+box[0][1])//2 +2]
            boxnew2[w] =  [(box[w][0]+box[2][0])//2+2,(box[w][1]+box[2][1])//2 +2] 
            box = np.int0(boxnew1)
            boxes.append(box)
            box = np.int0(boxnew2)
            boxes.append(box)           
          else:
            box = np.int0(box)
            boxes.append(box)
        print("Total Lemons in an image "+ldseg[i]+":",len(boxes))
        ans1 = cv2.drawContours(img[i], boxes, -1, (255, 255, 255),3)
        unknown.append(ans1)
        unknown.append(sure_fg)
        i+=1


plt.figure(figsize=(40,20))
columns = 8
for i, image in enumerate(unknown):
  plt.subplot(len(unknown) / columns + 1, columns, i + 1)
  plt.imshow(image)