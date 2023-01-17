import numpy as np

import pandas as pd

import cv2

import matplotlib.pyplot as plt
trainImgPath = "/kaggle/input/severstal-steel-defect-detection/train_images/"

trainCsv = "/kaggle/input/severstal-steel-defect-detection/train.csv"

data = pd.read_csv(trainCsv)

encoded_pixels=data.EncodedPixels

splitted_data=str.split(encoded_pixels[1]," ") 

splitted_data=[int(s) for s in splitted_data]

pixels,counts=[],[]

for i in range(len(splitted_data)):

    if(i%2==0):

        pixels.append(splitted_data[i])

    else:

        counts.append(splitted_data[i])

pre_mask=[list(range(pixels[i],pixels[i]+counts[i])) for i in range(len(pixels))]

pre_mask=[list2 for list1 in pre_mask for list2 in list1]

pre_mask

img=cv2.imread(trainImgPath+"/0007a71bf.jpg")

w,h=img.shape[0],img.shape[1]

mask=np.zeros((409600,1),dtype=int)

mask[pre_mask]=1

mask=np.reshape(mask,(h,w)).T

plt.imshow(mask)

#plt.imshow(cv2.imread(trainImgPath+"/0007a71bf.jpg"))
