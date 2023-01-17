# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input/"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import os

import glob

import cv2

import matplotlib.pyplot as plt

import skimage.feature

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelBinarizer

import keras

from keras.models import Sequential, load_model

from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Lambda, Cropping2D

from keras.utils import np_utils

import random

from random import randint

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import cross_val_score

from sklearn.metrics import precision_score, recall_score

from sklearn.model_selection import cross_val_predict

import pylab as pl

import seaborn as sn

from sklearn.metrics import accuracy_score

import tensorflow as tf

from sklearn.neural_network import MLPClassifier

from collections import Counter

from sklearn.metrics import roc_auc_score





%matplotlib inline
train_data = pd.read_csv('../input/Train/train.csv')

train_imgs = sorted(glob.glob('../input/Train/*.jpg'), key=lambda name: int(os.path.basename(name)[:-4]))

train_dot_imgs = sorted(glob.glob('../input/TrainDotted/*.jpg'), key=lambda name: int(os.path.basename(name)[:-4]))

index = 0



sl_counts = train_data.iloc[index]

print(sl_counts)



print(train_imgs[index])

img = cv2.cvtColor(cv2.imread(train_imgs[index]), cv2.COLOR_BGR2RGB)

img_dot = cv2.cvtColor(cv2.imread(train_dot_imgs[index]), cv2.COLOR_BGR2RGB)



f, ax = plt.subplots(1,2,figsize=(16,8))

(ax1, ax2) = ax.flatten()



ax1.imshow(img)

ax2.imshow(img_dot)
crop_img = img[1350:1900, 3000:3400]

crop_img_dot = img_dot[1350:1900, 3000:3400]



f, ax = plt.subplots(1,2,figsize=(16,8))

(ax1, ax2) = ax.flatten()



ax1.imshow(crop_img)

ax2.imshow(crop_img_dot)



plt.show()
hist = train_data.sum(axis=0)

print(hist)





sea_lions_types = hist[1:]

f, ax1 = plt.subplots(1,1,figsize=(5,5))

sea_lions_types.plot(kind='bar', title='Count of Sea Lion Types (Train)', ax=ax1)

plt.show()
index = 0

sl_counts = train_data.iloc[index]

print(sl_counts)



plt.figure()

sl_counts.plot(kind='bar', title='Count of Sea Lion Types')

plt.show()
# CREDITS GO TO:  Radu Stoicescu

class_names = ['adult_females', 'adult_males', 'juveniles', 'pups', 'subadult_males']



file_names = os.listdir("../input/Train/")

file_names = sorted(file_names, key=lambda 

                    item: (int(item.partition('.')[0]) if item[0].isdigit() else float('inf'), item)) 



# select a subset of files to run on

file_names = file_names[0:3]

print(file_names)

# dataframe to store results in

coordinates_df = pd.DataFrame(index=file_names, columns=class_names)
# CREDITS GO TO:  Radu Stoicescu

for filename in file_names:

    

    # read the Train and Train Dotted images

    image_1 = cv2.imread("../input/TrainDotted/" + filename)

    image_2 = cv2.imread("../input/Train/" + filename)

    

    cut = np.copy(image_2)

    

    # absolute difference between Train and Train Dotted

    image_3 = cv2.absdiff(image_1,image_2)

    

    # mask out blackened regions from Train Dotted

    mask_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)

    mask_1[mask_1 < 20] = 0

    mask_1[mask_1 > 0] = 255

    

    mask_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)

    mask_2[mask_2 < 20] = 0

    mask_2[mask_2 > 0] = 255

    

    image_3 = cv2.bitwise_or(image_3, image_3, mask=mask_1)

    image_3 = cv2.bitwise_or(image_3, image_3, mask=mask_2) 

    

    # convert to grayscale to be accepted by skimage.feature.blob_log

    image_3 = cv2.cvtColor(image_3, cv2.COLOR_BGR2GRAY)

    

    # detect blobs

    blobs = skimage.feature.blob_log(image_3, min_sigma=3, max_sigma=4, num_sigma=1, threshold=0.02)

    

    adult_males = []

    subadult_males = []

    pups = []

    juveniles = []

    adult_females = [] 

    

    image_circles = image_1

    

    for blob in blobs:

        # get the coordinates for each blob

        y, x, s = blob

        # get the color of the pixel from Train Dotted in the center of the blob

        g,b,r = image_1[int(y)][int(x)][:]

        

        # decision tree to pick the class of the blob by looking at the color in Train Dotted

        if r > 200 and g < 50 and b < 50: # RED

            adult_males.append((int(x),int(y)))

            cv2.circle(image_circles, (int(x),int(y)), 20, (0,0,255), 10) 

        elif r > 200 and g > 200 and b < 50: # MAGENTA

            subadult_males.append((int(x),int(y))) 

            cv2.circle(image_circles, (int(x),int(y)), 20, (250,10,250), 10)

        elif r < 100 and g < 100 and 150 < b < 200: # GREEN

            pups.append((int(x),int(y)))

            cv2.circle(image_circles, (int(x),int(y)), 20, (20,180,35), 10)

        elif r < 100 and  100 < g and b < 100: # BLUE

            juveniles.append((int(x),int(y))) 

            cv2.circle(image_circles, (int(x),int(y)), 20, (180,60,30), 10)

        elif r < 150 and g < 50 and b < 100:  # BROWN

            adult_females.append((int(x),int(y)))

            cv2.circle(image_circles, (int(x),int(y)), 20, (0,42,84), 10)  

            

        cv2.rectangle(cut, (int(x)-112,int(y)-112),(int(x)+112,int(y)+112), 0,-1)

            

    coordinates_df["adult_males"][filename] = adult_males

    coordinates_df["subadult_males"][filename] = subadult_males

    coordinates_df["adult_females"][filename] = adult_females

    coordinates_df["juveniles"][filename] = juveniles

    coordinates_df["pups"][filename] = pups
f, ax = plt.subplots(1,1,figsize=(10,16))

ax.imshow(cv2.cvtColor(image_circles, cv2.COLOR_BGR2RGB))

plt.show()
f, ax = plt.subplots(1,1,figsize=(10,16))

ax.imshow(cv2.cvtColor(cut, cv2.COLOR_BGR2RGB))

plt.show()
# CREDITS GO TO:  Radu Stoicescu

x = []

y = []



for filename in file_names:    

    image = cv2.imread("../input/Train/" + filename)

    for lion_class in class_names:

        for coordinates in coordinates_df[lion_class][filename]:

            thumb = image[coordinates[1]-32:coordinates[1]+32,coordinates[0]-32:coordinates[0]+32,:]

            if np.shape(thumb) == (64, 64, 3):

                x.append(thumb)

                y.append(lion_class)
# CREDITS GO TO:  Radu Stoicescu

for i in range(0,np.shape(cut)[0],224):

    for j in range(0,np.shape(cut)[1],224):                

        thumb = cut[i:i+64,j:j+64,:]

        if np.amin(cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY)) != 0:

            if np.shape(thumb) == (64,64,3):

                x.append(thumb)

                y.append("negative")  
# CREDITS GO TO:  Radu Stoicescu

class_names.append("negative")

y.count('negative')
sl_counts = train_data.iloc[0:3]

print(sl_counts)



plt.figure()

sl_counts.plot(kind='bar', title='Count of Sea Lion Types')

plt.show()
# CREDITS GO TO:  Radu Stoicescu

x = np.array(x)

y = np.array(y)
z = []

for img in x:

    img = np.array(img).reshape(12288)

    z.append(img)

z = np.array(z)

x.shape
# CREDITS GO TO:  Radu Stoicescu

for lion_class in class_names:

    f, ax = plt.subplots(1,10,figsize=(12,1.5))

    f.suptitle(lion_class)

    axes = ax.flatten()

    j = 0

    for a in axes:

        a.set_xticks([])

        a.set_yticks([])

        for i in range(j,len(x)):

            if y[i] == lion_class:

                j = i+1

                a.imshow(cv2.cvtColor(x[i], cv2.COLOR_BGR2RGB))

                break
encoder = LabelBinarizer()

encoder.fit(y)

y = encoder.transform(y).astype(float)
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,

                    hidden_layer_sizes=(100, 100), random_state=42)
clf.fit(z,y)
# CREDITS GO TO:  Radu Stoicescu

class_names = ['adult_females', 'adult_males', 'juveniles', 'pups', 'subadult_males']



file_names = os.listdir("../input/Train/")

file_names = sorted(file_names, key=lambda 

                    item: (int(item.partition('.')[0]) if item[0].isdigit() else float('inf'), item)) 



# select a subset of files to run on

file_names = file_names[8:9]



# dataframe to store results in

coordinates_df = pd.DataFrame(index=file_names, columns=class_names)





for filename in file_names:

    

    # read the Train and Train Dotted images

    image_1 = cv2.imread("../input/TrainDotted/" + filename)

    image_2 = cv2.imread("../input/Train/" + filename)

    

    cut = np.copy(image_2)

    

    # absolute difference between Train and Train Dotted

    image_3 = cv2.absdiff(image_1,image_2)

    

    # mask out blackened regions from Train Dotted

    mask_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)

    mask_1[mask_1 < 20] = 0

    mask_1[mask_1 > 0] = 255

    

    mask_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)

    mask_2[mask_2 < 20] = 0

    mask_2[mask_2 > 0] = 255

    

    image_3 = cv2.bitwise_or(image_3, image_3, mask=mask_1)

    image_3 = cv2.bitwise_or(image_3, image_3, mask=mask_2) 

    

    # convert to grayscale to be accepted by skimage.feature.blob_log

    image_3 = cv2.cvtColor(image_3, cv2.COLOR_BGR2GRAY)

    

    # detect blobs

    blobs = skimage.feature.blob_log(image_3, min_sigma=3, max_sigma=4, num_sigma=1, threshold=0.02)

    

    adult_males = []

    subadult_males = []

    pups = []

    juveniles = []

    adult_females = [] 

    

    image_circles = image_1

    

    for blob in blobs:

        # get the coordinates for each blob

        y, x, s = blob

        # get the color of the pixel from Train Dotted in the center of the blob

        g,b,r = image_1[int(y)][int(x)][:]

        

        # decision tree to pick the class of the blob by looking at the color in Train Dotted

        if r > 200 and g < 50 and b < 50: # RED

            adult_males.append((int(x),int(y)))

            cv2.circle(image_circles, (int(x),int(y)), 20, (0,0,255), 10) 

        elif r > 200 and g > 200 and b < 50: # MAGENTA

            subadult_males.append((int(x),int(y))) 

            cv2.circle(image_circles, (int(x),int(y)), 20, (250,10,250), 10)

        elif r < 100 and g < 100 and 150 < b < 200: # GREEN

            pups.append((int(x),int(y)))

            cv2.circle(image_circles, (int(x),int(y)), 20, (20,180,35), 10)

        elif r < 100 and  100 < g and b < 100: # BLUE

            juveniles.append((int(x),int(y))) 

            cv2.circle(image_circles, (int(x),int(y)), 20, (180,60,30), 10)

        elif r < 150 and g < 50 and b < 100:  # BROWN

            adult_females.append((int(x),int(y)))

            cv2.circle(image_circles, (int(x),int(y)), 20, (0,42,84), 10)  

            

        cv2.rectangle(cut, (int(x)-112,int(y)-112),(int(x)+112,int(y)+112), 0,-1)

            

    coordinates_df["adult_males"][filename] = adult_males

    coordinates_df["subadult_males"][filename] = subadult_males

    coordinates_df["adult_females"][filename] = adult_females

    coordinates_df["juveniles"][filename] = juveniles

    coordinates_df["pups"][filename] = pups







x_test = []

y_test = []



for filename in file_names:    

    image = cv2.imread("../input/Train/" + filename)

    for lion_class in class_names:

        for coordinates in coordinates_df[lion_class][filename]:

            thumb = image[coordinates[1]-32:coordinates[1]+32,coordinates[0]-32:coordinates[0]+32,:]

            if np.shape(thumb) == (64, 64, 3):

                x_test.append(thumb)

                y_test.append(lion_class)



for i in range(0,np.shape(cut)[0],224):

    for j in range(0,np.shape(cut)[1],224):                

        thumb = cut[i:i+64,j:j+64,:]

        if np.amin(cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY)) != 0:

            if np.shape(thumb) == (64,64,3):

                x_test.append(thumb)

                y_test.append("negative") 

class_names.append("negative")

x_test = np.array(x_test)

y_test = np.array(y_test)

z_test = []

for img in x_test:

    img = np.array(img).reshape(12288)

    z_test.append(img)

z_test = np.array(z_test)



encoder = LabelBinarizer()

encoder.fit(y_test)

y_test = encoder.transform(y_test)
cvs = cross_val_score(clf, z_test, y_test, cv=3, scoring="accuracy")
cvs
index = 8

sl_counts = train_data.iloc[index]

print(sl_counts)



plt.figure()

sl_counts.plot(kind='bar', title='Count of Sea Lion Types')

plt.show()
cvs = cross_val_score(clf, z_test, y_test, cv=3, scoring="accuracy")

cvs
index = 4

sl_counts = train_data.iloc[index]

print(sl_counts)



plt.figure()

sl_counts.plot(kind='bar', title='Count of Sea Lion Types')

plt.show()