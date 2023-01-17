# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# Any results you write to the current directory are saved as output.



import matplotlib.pyplot as plt

import cv2
image=cv2.imread('/kaggle/input/gundetection/445.jpg')

shape=image.shape

plt.figure()

plt.imshow(image[:,:,::-1])
label=open('/kaggle/input/gundetection/445.txt','r').read()

print(label)
def from_yolo_to_cor(box, shape):

    img_h, img_w, _ = shape

    x1, y1 = int((box[0] + box[2]/2)*img_w), int((box[1] + box[3]/2)*img_h)

    x2, y2 = int((box[0] - box[2]/2)*img_w), int((box[1] - box[3]/2)*img_h)

    return x1, y1, x2, y2
labels=label.split('\n')

labels0=[float(i) for i in labels[0].split()]

labels1=[float(i) for i in labels[1].split()]



labels0.remove(1) #removing label=1

labels1.remove(1)



[x11,y11,x12,y12]=from_yolo_to_cor(labels0,shape)

[x21,y21,x22,y22]=from_yolo_to_cor(labels1,shape)
image=cv2.rectangle(image, (x11, y11), (x12, y12), (255,0,0), 3)

image=cv2.rectangle(image, (x21, y21), (x22, y22), (255,0,0), 3)

plt.imshow(image)