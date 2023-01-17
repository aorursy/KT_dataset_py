#imports



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import os

import seaborn as sns

import matplotlib.image as mpimg

import random

from PIL import Image

import collections as co

import cv2

import scipy as sp

import copy



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



figWidth = figHeight = 10

whoAmI = 24601

random.seed(whoAmI)

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

sns.set_style("dark")

# Any results you write to the current directory are saved as output.
print(len(os.listdir("../input/train")))

print(len(os.listdir("../input/test")))
trainFrame = pd.read_csv("../input/train.csv")
trainFrame.shape
len(trainFrame["Id"].unique())
idCountFrame = trainFrame.groupby("Id",as_index = False)["Image"].count()
idCountFrame = idCountFrame.rename(columns = {"Image":"numImages"})
idCountFrame["density"] = idCountFrame["numImages"] / np.sum(idCountFrame["numImages"])
idCountFrame = idCountFrame.sort_values("density",ascending = False)
#rank them

idCountFrame["rank"] = range(idCountFrame.shape[0])

idCountFrame["logRank"] = np.log(idCountFrame["rank"] + 1)
plt.plot(idCountFrame["logRank"],idCountFrame["density"])

plt.xlabel("$\log(Rank)$")

plt.ylabel("Density")

plt.title("$\log(Rank)$-Density Plot for our Labels")
topLev = 10

idCountFrame.iloc[0:topLev,:]
trainDir = "../input/train"

testDir = "../input/test"
sampleImageFilename = "../input/train/00022e1a.jpg"

sampleImage = mpimg.imread(sampleImageFilename)

plt.imshow(sampleImage)
iiwcInfo = idCountFrame[idCountFrame["Id"].str.contains("iiwc")]

numberInfo = idCountFrame[idCountFrame["Id"].str.contains("1034")]

print(iiwcInfo)

print(numberInfo)
trainFrame[trainFrame["Id"] == "w_103488f"]
#sample a couple of pictures

numSampled = 4

sampledPicNames = random.sample(os.listdir(trainDir),numSampled)

#then read the images

readImages = [mpimg.imread(trainDir + os.sep + sampledPicNames[i])

             for i in range(len(sampledPicNames))]

#then plot

fig, subplots = plt.subplots(2,2)

fig.set_size_inches(figWidth,figHeight)

for i in range(len(readImages)):

    subplots[int(i / 2),i % 2].imshow(readImages[i])
#then sample the test set

numSampled = 4

sampledPicNames = random.sample(os.listdir(testDir),numSampled)

#then read the images

readImages = [mpimg.imread(testDir + os.sep + sampledPicNames[i])

             for i in range(len(sampledPicNames))]

#then plot

fig, subplots = plt.subplots(2,2)

fig.set_size_inches(figWidth,figHeight)

for i in range(len(readImages)):

    subplots[int(i / 2),i % 2].imshow(readImages[i])
imageSizes = co.Counter([Image.open(f'../input/train/{filename}').size

                        for filename in os.listdir("../input/train")])
imageSizeFrame = pd.DataFrame(list(imageSizes.most_common()),columns = ["imageDim","count"])
#get density

imageSizeFrame["density"] = imageSizeFrame["count"] / np.sum(imageSizeFrame["count"])

#get rank

imageSizeFrame["rank"] = range(imageSizeFrame.shape[0])

imageSizeFrame["logRank"] = np.log(imageSizeFrame["rank"] + 1)
#then plot

plt.plot(imageSizeFrame["logRank"],imageSizeFrame["density"])

plt.xlabel("$\log(Rank)$")

plt.ylabel("Density")

plt.title("$\log(Rank)$-Density Plot for image sizes in the training set")
topLev = 10

imageSizeFrame.iloc[0:topLev,:]
testImageSizesCounter = co.Counter([Image.open(f'../input/test/{filename}').size

                                    for filename in os.listdir("../input/test")])
testImageSizeFrame = pd.DataFrame(list(testImageSizesCounter.most_common()),

                                  columns = ["imageDim","count"])
#get density

testImageSizeFrame["density"] = testImageSizeFrame["count"] / np.sum(testImageSizeFrame["count"])

#get rank

testImageSizeFrame["rank"] = range(testImageSizeFrame.shape[0])

testImageSizeFrame["logRank"] = np.log(testImageSizeFrame["rank"] + 1)
topLev = 10

testImageSizeFrame.iloc[0:topLev,:]
def is_grey_scale(givenImage):

    """Adopted from 

    https://www.kaggle.com/lextoumbourou/humpback-whale-id-data-and-aug-exploration"""

    w,h = givenImage.size

    for i in range(w):

        for j in range(h):

            r,g,b = givenImage.getpixel((i,j))

            if r != g != b: return False

    return True
sampleFrac = 0.1

#get our sampled images

imageList = [Image.open(f'../input/train/{imageName}').convert('RGB')

            for imageName in trainFrame['Image'].sample(frac=sampleFrac)]
isGreyList = [is_grey_scale(givenImage) for givenImage in imageList]
#then get proportion greyscale

np.sum(isGreyList) / len(isGreyList)
#first get filenames

trainImageFilenames = os.listdir("../input/train")

testImageFilenames = os.listdir("../input/test")
#sample 1000 from each

sampleSize = 1000

trainImageFilenamesSample = random.sample(trainImageFilenames,sampleSize)

testImageFilenamesSample = random.sample(testImageFilenames,sampleSize)
#then get images

trainImageSample = [cv2.imread(f'../input/train/{trainImageFilename}',0)

                    for trainImageFilename in trainImageFilenamesSample]

testImageSample = [cv2.imread(f'../input/test/{testImageFilename}',0)

                    for testImageFilename in testImageFilenamesSample]
#then get histograms for each

colorMax = 256

trainImageHists =  [cv2.calcHist([trainImage],[0],None,[colorMax],[0,colorMax]).squeeze()

                    for trainImage in trainImageSample]

testImageHists =  [cv2.calcHist([testImage],[0],None,[colorMax],[0,colorMax]).squeeze()

                    for testImage in testImageSample]
#normalize each

trainImageHists = [trainImageHist / np.sum(trainImageHist) for trainImageHist in trainImageHists]

testImageHists = [testImageHist / np.sum(testImageHist) for testImageHist in testImageHists]
#then get wasserstein distances

wassersteinDistances = [sp.stats.energy_distance(trainImageHists[i],testImageHists[i])

                        for i in range(len(trainImageHists))]
testStatistic = np.mean(wassersteinDistances)
testStatistic
def bootstrapMeanWassersteinDistance(imageList,numSamples):

    """Helper for bootstrapping the mean wasserstein distance from a given filename list"""

    #first get full sample

    fullSampleImages = random.sample(imageList,numSamples * 2)

    #then get train-test split by indices

    fullSampleImageIndices = [i for i in range(len(fullSampleImages))]

    trainImageSampleIndices = random.sample(fullSampleImageIndices,numSamples)

    testImageSampleIndices = list(set(fullSampleImageIndices) - set(trainImageSampleIndices))

    #then actually get said images

    trainImageSample = [fullSampleImages[i] for i in trainImageSampleIndices]

    testImageSample = [fullSampleImages[i] for i in testImageSampleIndices]

    #then get histograms

    colorMax = 256

    trainImageHists =  [cv2.calcHist([trainImage],[0],None,[colorMax],[0,colorMax]).squeeze()

                        for trainImage in trainImageSample]

    testImageHists =  [cv2.calcHist([testImage],[0],None,[colorMax],[0,colorMax]).squeeze()

                        for testImage in testImageSample]

    #normalize each

    trainImageHists = [trainImageHist / np.sum(trainImageHist) 

                       for trainImageHist in trainImageHists]

    testImageHists = [testImageHist / np.sum(testImageHist) 

                      for testImageHist in testImageHists]

    #then get wasserstein distances

    wassersteinDistances = [sp.stats.energy_distance(trainImageHists[i],testImageHists[i])

                        for i in range(len(trainImageHists))]

    #then get test statistic

    return np.mean(wassersteinDistances)
def runSimulations(imageList,numSims,numSamples):

    """Helper that bootstraps our full distribution of mean wasserstein distances"""

    wdDist = [bootstrapMeanWassersteinDistance(imageList,numSamples) for i in range(numSims)]

    return wdDist
#then form filename list

trainImageFilenames = [f'../input/train/{trainImageFilename}'

                       for trainImageFilename in trainImageFilenames]

testImageFilenames = [f'../input/test/{testImageFilename}'

                       for testImageFilename in testImageFilenames]
filenameList = copy.deepcopy(trainImageFilenames)

filenameList.extend(testImageFilenames)
#and because we would crash Kaggle if we loaded in all the images, let's just load in 8000

metaSampleSize = 8000

filenameSample = random.sample(filenameList,metaSampleSize)
imageList = [cv2.imread(filename,0) for filename in filenameSample]
numSims = 100

numSamples = 1000

wdDist = runSimulations(imageList,numSims,numSamples)
wdDistVec = np.array(wdDist)

np.mean(wdDistVec > testStatistic)