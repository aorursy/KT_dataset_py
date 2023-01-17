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

CATEGORIES = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat', 'Fat Hen', 'Loose Silky-bent',
             'Maize', 'Scentless Mayweed', 'Shepherds Purse', 'Small-flowered Cranesbill', 'Sugar beet']

import matplotlib.pyplot as plt
from glob import glob
import cv2
import math


scaleSize = 80 #resize figures to 80 px
seed = 0 #random seed

path='../input/train/*/*.png'
files = glob(path)

trainImage = []
trainLabel = []
j = 1
num = len(files)

for img in files:
    if (j >= num):
        print(str(j)+"/"+str(num), end="\r")
    trainImage.append(cv2.resize(cv2.imread(img), (scaleSize, scaleSize)))
    trainLabel.append(img.split('/')[-2])
    j = j + 1
    
trainImage = np.asarray(trainImage)
trainLabel = pd.DataFrame(trainLabel)
for i in range(8):
    plt.subplot(2,4,i+1)
    plt.imshow(trainImage[i])

clearTrainImage = []
maskTrainImage = []
blurredTrainImage = []
segmentTrainImage = []
sharpTrainImage = []
examples = []
getEx = True
for image in trainImage:
    #Gaussian blur
    #blurImage = cv2.GaussianBlur(image, (5,5), 0)
    
    #Convert to HSV
    hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    #Create mask
    sensitivity = 35
    green_lb = (60 - sensitivity, 100, 50) #lower bound of green range
    green_ub = (60 + sensitivity, 255, 255) #upper bound of green range
    mask = cv2.inRange(hsvImage, green_lb, green_ub)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    #segment image
    segmentImage = cv2.bitwise_and(image, image, mask = mask)
    
    #sharpen
    blurredImage = cv2.GaussianBlur(segmentImage, (5,5), 0)
    sharpImage = cv2.addWeighted(segmentImage, 1.5, blurredImage, -0.5, 0)
    maskTrainImage.append(mask)
    segmentTrainImage.append(segmentImage)
    blurredTrainImage.append(blurredImage)
    sharpTrainImage.append(sharpImage)
    
    
    #Apply the mas
    
fig, axs = plt.subplots(1,5, figsize=(20,20))
axs[0].imshow(trainImage[978])
#plt.subplot(2,3,2)
axs[1].imshow(maskTrainImage[978])
#plt.subplot(2,3,3)
axs[2].imshow(segmentTrainImage[978])
#plt.subplot(2,3,4)
axs[3].imshow(blurredTrainImage[978])
#plt.subplot(2,3,5)
axs[4].imshow(sharpTrainImage[978])