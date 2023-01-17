# imports



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import keras.preprocessing.image as kim

from keras.backend import tf as ktf

import os

from PIL import Image

from matplotlib import pyplot as plt

import copy



#helpers



sigLev = 3

pd.options.display.precision = sigLev

figWidth = figHeight = 9

inputDir = "../input"

trainFrame = pd.read_csv(f'{inputDir}/train.csv')
classCountFrame = trainFrame.groupby("Id",as_index = False)["Image"].count()

classCountFrame = classCountFrame.rename(columns = {"Image":"count"})

#then order

classCountFrame = classCountFrame.sort_values("count",ascending = True)

#then just check the head

classCountFrame.head()
chosenClass = classCountFrame["Id"].iloc[1]

consideredImageObs = trainFrame[trainFrame["Id"] == chosenClass]

consideredImageFilename = consideredImageObs["Image"].iloc[0]
fullImageFilename = f'{inputDir}/train/{consideredImageFilename}'

chosenImage = Image.open(fullImageFilename)
plt.imshow(chosenImage)

#get rid of axes

cur_axes = plt.gca()

cur_axes.axes.get_xaxis().set_visible(False)

cur_axes.axes.get_yaxis().set_visible(False)
chosenImage.size
idealWidth = 1050

idealHeight = 600

resizedChosenImage = chosenImage.resize((idealWidth,idealHeight),Image.NEAREST)
plt.imshow(resizedChosenImage)

#get rid of axes

cur_axes = plt.gca()

cur_axes.axes.get_xaxis().set_visible(False)

cur_axes.axes.get_yaxis().set_visible(False)
greyChosenImage = resizedChosenImage.convert("LA")
#then plot

fig, axes = plt.subplots(1,2)

fig.set_size_inches(figWidth,figHeight)

axes[0].imshow(resizedChosenImage)

axes[1].imshow(greyChosenImage)
#convert image to array

imageArray = np.array(resizedChosenImage)
numRotations = 4

rotationSize = 30

#make rotation

rotatedImages = [

    kim.random_rotation(imageArray,rotationSize, 

                        row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest')

    for _ in range(numRotations)]
#then plot each

fig, givenSubplots = plt.subplots(2,2)

fig.set_size_inches(figWidth,figHeight)

for i in range(len(rotatedImages)):

    givenSubplots[int(i / 2),i % 2].imshow(rotatedImages[i])
numShifts = numRotations

widthRange = 0.1

heightRange = 0.3

shiftedImages = [

    kim.random_shift(imageArray, wrg= widthRange, hrg= heightRange, 

                     row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest')

    for _ in range(numShifts)]
#then plot each

fig, givenSubplots = plt.subplots(2,2)

fig.set_size_inches(figWidth,figHeight)

for i in range(len(shiftedImages)):

    givenSubplots[int(i / 2),i % 2].imshow(shiftedImages[i])
numShears = numRotations

givenIntensity = 0.4

shearedImages = [

    kim.random_shear(imageArray, intensity= givenIntensity, 

                 row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest')

    for _ in range(numShears)]
#then plot

fig, givenSubplots = plt.subplots(2,2)

fig.set_size_inches(figWidth,figHeight)

for i in range(len(shearedImages)):

    givenSubplots[int(i / 2),i % 2].imshow(shearedImages[i])
numZooms = numRotations

zoomRangeWidth = 1.5

zoomRangeHeight = .7

zoomedImages = [

    kim.random_zoom(imageArray, zoom_range=(zoomRangeWidth,zoomRangeHeight),

                row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest')

    for _ in range(numZooms)]
#then plot

fig, givenSubplots = plt.subplots(2,2)

fig.set_size_inches(figWidth,figHeight)

for i in range(len(zoomedImages)):

    givenSubplots[int(i / 2),i % 2].imshow(zoomedImages[i])
#collapse to redscale

redImageArray = copy.deepcopy(imageArray)

redImageArray[:,:,1] = 0

redImageArray[:,:,2] = 0
plt.imshow(redImageArray)
colorEnhancer = 30

averageArray = np.mean(imageArray,axis = 2)

redImageArray = copy.deepcopy(imageArray)

redImageArray[:,:,1] = averageArray

redImageArray[:,:,2] = averageArray

redImageArray[:,:,0] += colorEnhancer
plt.imshow(redImageArray)
initialNumImages = 2

numberOfTransformations = 4

numRandomIterationsOfTransformation = 4

numImages = initialNumImages

#then apply sequentially

for i in range(numberOfTransformations):

    numImages += (numImages * numRandomIterationsOfTransformation)

numImages
initialNumImages = 1

numberOfTransformations = 4

numRandomIterationsOfTransformation = 4

numImages = initialNumImages

#then apply sequentially

for i in range(numberOfTransformations):

    numImages += (numImages * numRandomIterationsOfTransformation)

numImages