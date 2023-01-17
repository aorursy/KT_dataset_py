#importing libraries



import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import os

from os import listdir

import seaborn as sns

from operator import itemgetter 

import matplotlib.image as mpimg

import random

from PIL import Image

import collections as co

import cv2

import scipy as sp

import copy

import plotly.graph_objs as go

import plotly.offline as py
trainDir = "../input/whale-categorization-playground/train/train"

testDir="../input/whale-categorization-playground/test/test/"

valuesFile= "../input/whale-categorization-playground/train.csv"


lntrd=len(listdir(trainDir))

lntsd=len(listdir(testDir))



print("number of train files: "+ str(lntrd))

print("number of test files: " + str(lntsd))

trainPD=pd.read_csv(valuesFile)

if lntrd>0:

    print("lengths of test set to train set: %6.2f" % (lntsd/lntrd))

    if trainPD.shape[0]==lntrd:

        print("number of values and length of train set are consistent")

    else:

        print("number of values and length of train set are inconsistent")

else:

    print("train set is empty")
FrameID = trainPD.groupby("Id",as_index = False)["Image"].count()

sortedID_train = FrameID.sort_values("Image",ascending = False)

idnum=sortedID_train.shape[0]

print(idnum)
sortedID_train.head()
plt.plot(range(idnum),sortedID_train["Image"])



plt.xlabel("sorted index")

plt.ylabel("frequency of occurence")

plt.title("frequency of occurence of labels")
plt.plot(range(1,idnum),sortedID_train["Image"][1:idnum])



plt.xlabel("sorted index")

plt.ylabel("density")

plt.title("Density Plot for Labels")
plt.plot(range(1,idnum+1),sortedID_train["Image"])

plt.yscale("log")

plt.xlabel("ID")

plt.ylabel("Frequency of occurence")

plt.title("Frequency of occurence: log scale")
plt.plot(range(1,idnum),sortedID_train["Image"][1:idnum])

plt.yscale("log")

#plt.yscale("log")

plt.xlabel("ID")

plt.ylabel("Frequency of occurence")

plt.title("Frequency of occurence: log scale")
imnum=25

plt.rcParams["figure.figsize"] = (70,70)

fig, subplots = plt.subplots(5,5)



for i in range(imnum):

    readImg=mpimg.imread(trainDir+"/"+(listdir(trainDir))[i])

    subplots[i // 5,i % 5].imshow(readImg)
readImg=mpimg.imread(trainDir+"/"+(listdir(trainDir))[10])

plt.rcParams["figure.figsize"] = (10,10)

plt.imshow(readImg)

print(listdir(trainDir)[10])
trainPD[trainPD["Image"] == "47841f63.jpg"]
imnum=25

plt.rcParams["figure.figsize"] = (70,70)

fig, subplots = plt.subplots(5,5)



for i in range(imnum):

    readImg=mpimg.imread(testDir+"/"+(listdir(testDir))[i])

    subplots[i // 5,i % 5].imshow(readImg)
sizedict_train=dict()

filelist=listdir(trainDir)

for filename in filelist:

    size=(Image.open(trainDir+"/"+filename)).size

    if size in sizedict_train:

        sizedict_train[size]+=1

    else:

        sizedict_train[size]=1
sortpairs_train= sorted(sizedict_train.items(), key = itemgetter(1), reverse = True)
sortsized_train = [sortpairs_train[i][1] for i in range(len(sortpairs_train))]

sortsized_train = sortsized_train/ np.sum(sortsized_train)
numsizes=len(sizedict_train)

print(numsizes)

plt.rcParams["figure.figsize"] = (5,5)

plt.plot(sortsized_train)



plt.xlabel("index")

plt.ylabel("probability")

plt.title("probability of size")
sortsized_train[0:10]
numsizes=len(sizedict_train)

print(numsizes)



plt.plot(sortsized_train)

plt.yscale("log")

plt.xlabel("index")

plt.ylabel("probability")

plt.title("probability of size")
sizedict_test=dict()

filelist=listdir(testDir)

for filename in filelist:

    size=(Image.open(testDir+"/"+filename)).size

    if size in sizedict_test:

        sizedict_test[size]+=1

    else:

        sizedict_test[size]=1
sortpairs_test= sorted(sizedict_test.items(), key = itemgetter(1), reverse = True)
sortpairs_test[0:10]
sortsized_test = [sortpairs_test[i][1] for i in range(len(sortpairs_test))]

sortsized_test = sortsized_test/ np.sum(sortsized_test)
numsizes=len(sizedict_test)

print(numsizes)



plt.plot(sortsized_test)



plt.xlabel("index")

plt.ylabel("probability")

plt.title("probability of size")
numsizes=len(sizedict_test)

print(numsizes)



plt.plot(sortsized_test)

plt.yscale("log")

plt.xlabel("sorted index")

plt.ylabel("density")

plt.title("Density Plot for Labels")
sortsized_test[0:10]
def checkrgb(rgb):

    

    if len(rgb.shape)==3:

        return 0

    else:

        return 1
lntd=len(listdir(trainDir))

grayscale=[checkrgb(mpimg.imread(trainDir+"/"+(listdir(trainDir))[i])) for i in range(lntd)]
share_grey_train=np.sum(grayscale)/len(grayscale)

share_grey_train
lntd=len(listdir(testDir))

grayscale=[checkrgb(mpimg.imread(testDir+"/"+(listdir(testDir))[i])) for i in range(lntd)]
share_grey_test=np.sum(grayscale)/len(grayscale)

share_grey_test
def rgb2grey(rgb): 

    if len(rgb.shape)==3:

        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140]) 

    else:

        return rgb





def transform_image(img, rsc_dim):

    resized = cv2.resize(img, (rsc_dim, rsc_dim), cv2.INTER_LINEAR)

    

    normalized = cv2.normalize(resized, None, 0.0, 1.0, cv2.NORM_MINMAX)

                         

    trans = normalized.reshape(1, np.prod(normalized.shape))



    return trans/np.linalg.norm(trans)
trainImg=[rgb2grey(mpimg.imread(trainDir+"/"+(listdir(trainDir))[i])) for i in range(400)]
testImg=[rgb2grey(mpimg.imread(testDir+"/"+(listdir(testDir))[i])) for i in range(400)]
rsc_dim=100

gray_all_images_train = [transform_image(img, rsc_dim) for img in trainImg]

gray_all_images_test  = [transform_image(img, rsc_dim) for img in testImg]
gray_imgs_mat_train = np.array(gray_all_images_train).squeeze()

gray_imgs_mat_test= np.array(gray_all_images_test).squeeze()
inputtsne=np.concatenate([gray_imgs_mat_train, gray_imgs_mat_test])
from sklearn.manifold import TSNE

tsne = TSNE(

    n_components=3,

    init='random', # pca

    random_state=101,

    method='barnes_hut',

    n_iter=500,

    verbose=2

).fit_transform(inputtsne)


import matplotlib.pyplot as plt



plt.rcParams["figure.figsize"] = (20,20)



fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')



x =tsne[0:400,0]

y =tsne[0:400,1]

z =tsne[0:400,2]



ax.scatter3D(x, y, z, c='r', marker='o')



x =tsne[400:800,0]

y =tsne[400:800,1]

z =tsne[400:800,2]



ax.scatter3D(x, y, z, c='b', marker='o')





ax.set_xlabel('X Label')

ax.set_ylabel('Y Label')

ax.set_zlabel('Z Label')



plt.show()