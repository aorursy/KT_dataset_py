# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



#brett ALbrecht-Diego Velazquez





# CHECK THIS OUT! https://www.python-course.eu/python_image_processing.php

# https://medium.com/neuralspace/kaggle-1-winning-approach-for-image-classification-challenge-9c1188157a86



from keras.models import Sequential

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from keras.layers.core import Activation

from keras.layers.core import Flatten

from keras.layers.core import Dense

from keras import backend as K

import sklearn

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# import the necessary packages

from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import Adam

from sklearn.model_selection import train_test_split

from keras.preprocessing.image import img_to_array

from keras.utils import to_categorical

#import pyimagesearch

#from pyimagesearch.lenet import LeNet

#from imutils import paths

import matplotlib.pyplot as plt

import argparse

import random

import cv2

import os

# the following line is only necessary in Python notebook:

%matplotlib inline

from scipy import misc





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

artists = []

IMG_SIZE = 80

DATA_DIR = ".."

CATEGORIES = ["Impressionism", "Surrealism"]



for dirname, _, filenames in os.walk('/kaggle/input/best-artworks-of-all-time/images/images/'):

    if dirname is not "/kaggle/input/best-artworks-of-all-time/images/images/":

        artists.append(dirname)

    #for filename in filenames:

        #print(os.path.join(dirname, filename))



        

training_set = []

for a in artists:

    

    #SOLE impressionists for the most part. no genre crossovers yet

    #we can still reorganize the file structure as a goal, just doing this to get data in to work with quickly.

    if (a == '/kaggle/input/best-artworks-of-all-time/images/images/Edgar_Degas') or (a == '/kaggle/input/best-artworks-of-all-time/images/images/Claude_Monet') or (a == '/kaggle/input/best-artworks-of-all-time/images/images/Alfred_Sisley')or (a == '/kaggle/input/best-artworks-of-all-time/images/images/Camille_Pissarro') or (a == '/kaggle/input/best-artworks-of-all-time/images/images/Paul_Gauguin') or (a == '/kaggle/input/best-artworks-of-all-time/images/images/Pierre-Auguste_Renoir'):

        print(a)

        for img in os.listdir(a):

            #print(img) 

            

            img_array = cv2.imread(os.path.join(a, img))

            #img_array = cv2.imread(os.path.joing(a, img), cv2.IMREAD_GRAYSCALE)

            new_mat = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

                

            training_set.append([new_mat, 0, os.path.join(a, img)]) #arbitrarily 0 for impressionism

    elif (a == '/kaggle/input/best-artworks-of-all-time/images/images/Frida_Kahlo') or (a == '/kaggle/input/best-artworks-of-all-time/images/images/Marc_Chagall') or (a == '/kaggle/input/best-artworks-of-all-time/images/images/Pablo_Picasso') or (a == '/kaggle/input/best-artworks-of-all-time/images/images/Paul_Klee') or (a == '/kaggle/input/best-artworks-of-all-time/images/images/Salvador_Dali'):

        print(a)

        for img in os.listdir(a):

            img_array = cv2.imread(os.path.join(a, img))

            #img_array = cv2.imread(os.path.joing(a, img), cv2.IMREAD_GRAYSCALE)

            new_mat = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

                

            training_set.append([new_mat, 1, os.path.join(a, img)]) #arbitrarily 1 for surrealism

print(len(training_set))
import random

random.shuffle(training_set)

for sample, cat, pathOG in training_set[:5]:

    plt.figure(figsize=(12,12))

    

    

    plt.subplot(1,2,1)

    plt.imshow(sample, cmap='gray')

    

    

    plt.subplot(1,2,2)

    img=cv2.imread(pathOG)

    imgplot = plt.imshow(img)

    

    plt.show()

    

X = []

Y = []



for features, label, path in training_set:

    X.append(features)

    Y.append(label)


# set the matplotlib backend so figures can be saved in the backgro
EPOCHS = 25

INIT_LR = 1e-3

BS = 32







# scale the raw pixel intensities to the range [0, 1]

data = np.array(X, dtype="float") / 255.0

labels = np.array(Y)



# partition the data into training and testing splits using 75% of

# the data for training and the remaining 25% for testing

(trainX, testX, trainY, testY) = train_test_split(data,

	labels, test_size=0.25, random_state=42)



# convert the labels from integers to vectors

trainY = to_categorical(trainY, num_classes=2)

testY = to_categorical(testY, num_classes=2)



# scale the raw pixel intensities to the range [0, 1]

data = np.array(X, dtype="float") / 255.0

labels = np.array(Y)

 

# partition the data into training and testing splits using 75% of

# the data for training and the remaining 25% for testing

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

 

# convert the labels from integers to vectors

trainY = to_categorical(trainY, num_classes=2)

testY = to_categorical(testY, num_classes=2)
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,

    height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,

    horizontal_flip=True, fill_mode="nearest")
%pylab inline





img=imread('../input/best-artworks-of-all-time/images/images/Leonardo_da_Vinci/Leonardo_da_Vinci_78.jpg')

imgplot = plt.imshow(img)

plt.show()




