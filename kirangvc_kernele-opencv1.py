# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
img_path = "../input/computer-vision-with-python/Computer-Vision-with-Python/DATA/00-puppy.jpg"

img = cv2.imread(img_path)
plt.imshow(img)
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

plt.imshow(img)
img = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)

plt.imshow(img)
img1 = cv2.imread("../input/computer-vision-with-python/Computer-Vision-with-Python/DATA/dog_backpack.jpg")

img2 = cv2.imread("../input/computer-vision-with-python/Computer-Vision-with-Python/DATA/watermark_no_copy.png")

img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)

img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)



plt.imshow(img2)
plt.imshow(img1)
img1 = cv2.resize(img1,(1000,1400))

img2 = cv2.resize(img2,(1000,1400))

plt.imshow(img1)
plt.imshow(img2)
blended = cv2.addWeighted(src1 = img1, alpha = 0.4, src2 = img2, beta = 0.5, gamma = 0)

plt.imshow(blended)