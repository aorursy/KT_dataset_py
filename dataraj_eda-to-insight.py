# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from skimage import io

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# Any results you write to the current directory are saved as output.
trainData =  pd.read_csv("../input/pku-autonomous-driving/train.csv")

trainData.head()
trainData.shape
trainpath = "../input/pku-autonomous-driving/train_images/" 
img0 = io.imread(trainpath+"ID_8a6e65317.jpg")

io.imshow(img0)
img1 = io.imread(trainpath+"ID_337ddc495.jpg")

io.imshow(img1)
img2 = io.imread(trainpath+"ID_a381bf4d0.jpg")

io.imshow(img2)
img3 = io.imread(trainpath+"ID_7c4a3e0aa.jpg")

io.imshow(img3)
maskpath = "../input/pku-autonomous-driving/train_masks/"
import matplotlib.pyplot as plt
img0 = io.imread(trainpath+"ID_8a6e65317.jpg")

fig = plt.figure(figsize=(20,10))



ax1 = fig.add_subplot(1,2,1)

ax1.imshow(img0)

img0m = io.imread(maskpath+"ID_8a6e65317.jpg")

ax2 = fig.add_subplot(1,2,2)

ax2.imshow(img0m)
trainData.PredictionString[0]
### Every string has data of more than vehicles in Image ID_8a6e65317.jpg. 

numOfCars = len(trainData.PredictionString[0].split(" "))/7

print("Number of cars in image ID_8a6e65317.jpg is : ", numOfCars)
data = trainData.PredictionString[0]

dataList = data.split(" ")

numOfVehicles = len(dataList) /7

variables = ["modeltype", "yaw", "pitch", "roll", "x", "y", "z"]

listOfData = []

for i in range(0,len(dataList),7) :

    

    lastIndex = i+7

    dt = dataList[i:lastIndex:1]

    dct = dict(zip(variables,dt))

    listOfData.append(dct)

    

    
listOfData
img1 = io.imread(trainpath+"ID_337ddc495.jpg")

fig = plt.figure(figsize=(20,10))



ax1 = fig.add_subplot(1,2,1)

ax1.imshow(img0)

img1m = io.imread(maskpath+"ID_337ddc495.jpg")

ax2 = fig.add_subplot(1,2,2)

ax2.imshow(img0m)
img1m.shape
(img1m[0,:,:] == img1m[1,:,:]).all()
numOfCars = len(trainData.PredictionString[1].split(" "))/7

print("Number of cars in image ID_337ddc495.jpg is : ", numOfCars)
data = trainData.PredictionString[0]

def addDictionary(data) :



    dataList = data.split(" ")

    numOfVehicles = len(dataList) /7

    variables = ["modeltype", "yaw", "pitch", "roll", "x", "y", "z"]

    listOfData = []

    for i in range(0,len(dataList),7) :

    

        lastIndex = i+7

        dt = dataList[i:lastIndex:1]

        dct = dict(zip(variables,dt))

        listOfData.append(dct)

    return listOfData

addDictionary(data)
trainData["dictVal"] = trainData.PredictionString.apply(lambda x : addDictionary(x))
trainData.head()
trainData["noOfVehicle"] = trainData.dictVal.apply(lambda x : len(x))
trainData.head()
print("Average number of vehicles in pictures : ",trainData["noOfVehicle"].mean())