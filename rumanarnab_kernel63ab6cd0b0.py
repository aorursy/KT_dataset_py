# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import keras

from keras.applications.vgg19 import VGG19

from keras.applications.resnet50 import ResNet50

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ModelCheckpoint

from keras.models import Model

from keras.layers import Dense, Dropout, Flatten,  Conv2D, MaxPooling2D, GlobalAveragePooling2D

from keras import backend as K

from keras.layers.normalization import BatchNormalization

from keras.preprocessing.image import load_img

#from keras.applications.vgg16 import preprocess_input

from keras.applications.resnet50 import preprocess_input

from keras.preprocessing.image import img_to_array

import os

from tqdm import tqdm

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

import cv2

import sys

import bcolz

import random





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Reading the csv data - each ID/picture in train corresponds to ab image

df_train = pd.read_csv('../input/labels.csv')



# Reading the sample of submission file - for each ID/picture, we have to predict the each of the dog probability

df_test = pd.read_csv('../input/sample_submission.csv')
# Printing top 10 rows

df_train.head(10) 
%matplotlib inline

import matplotlib.pyplot as plt

from glob import glob

from mpl_toolkits.axes_grid1 import ImageGrid
# Importing all the pictures for the train and test

train_files = glob('../input/train/*.jpg')

test_files = glob('../input/test/*.jpg')
# Printing the 100th image from train

plt.imshow(plt.imread(train_files[100]))
# Creating dummy flags for the train data and storing them in array

targets_series = pd.Series(df_train['breed'])

one_hot = pd.get_dummies(targets_series, sparse = True)

one_hot_labels = np.asarray(one_hot)
# Sample

print(one_hot_labels)
# Defining the image size for storage

im_size = 300
# Inititalizing a bcolz array of the required size i.e. no. of pics x image size x image size x 3(for RGB)

y_train = []

y_val = []

x_train_raw = bcolz.zeros((0,im_size,im_size,3),np.float32)

x_val_raw = bcolz.zeros((0,im_size,im_size,3),np.float32)
# Running a loop to store each of the pictures in bcolz and splitting the list of image in 80-20

i = 0 

for f, breed in tqdm(df_train.values):

    # load an image from file

    image = load_img('../input/train/{}.jpg'.format(f), target_size=(im_size, im_size))

    image = img_to_array(image)

    # prepare the image for the VGG model

    #image = preprocess_input(image)

    label = one_hot_labels[i]

    if random.randint(1,101) < 80: 

        x_train_raw.append(image)

        y_train.append(label)

    else:

        x_val_raw.append(image)

        y_val.append(label)

    i += 1
# Creating a numpy array for the flags

y_train_raw = np.array(y_train, np.uint8)

y_val_raw = np.array(y_val, np.uint8)

del(y_train,y_val)



# Garbage collection feature of python

import gc

gc.collect()
# Checking the dimension of image arrays(in form of RGB) and the label array (one hot encoded)

print(x_train_raw.shape)

print(y_train_raw.shape)

print(x_val_raw.shape)

print(y_val_raw.shape)
# Defining a function for printing images in defined format

def plotImages( images_arr, n_images = 2):

    fig, axes = plt.subplots(n_images, n_images, figsize=(12,12))

    axes = axes.flatten()

    for img, ax in zip( images_arr, axes):

        ax.imshow(img)

        ax.set_xticks(())

        ax.set_yticks(())

    plt.tight_layout()



# Running the function to print the first 16 pictures

plotImages(x_train_raw[0:4,]/255.)
plotImages(x_train_raw[10:11,]/255.)
# Shape of a picture : image_size x image_size x 3(RGB)

np.shape(x_train_raw[0,])
batch_size = 2
plotImages(x_train_raw[10:11,]/255.)
datagen = ImageDataGenerator()

temp = np.zeros((16,im_size,im_size,3),dtype=np.float32)

image_to_test = 10

for i in range(16):

    if random.randint(1,101) < 50: 

        flip_horizontal = False

    else:

        flip_horizontal = False

    if random.randint(1,101) < 50: 

        flip_vertical = False

    else:

        flip_vertical = False

    

    if random.randint(1,101) < 50:

        tx = im_size*random.randint(1,15)/100.0

    else:

        tx = im_size*random.randint(-15,-1)/100.0

        

    if random.randint(1,101) < 50:

        ty = im_size*random.randint(1,15)/100.0

    else:

        ty = im_size*random.randint(-15,-1)/100.0

    



    zx = 1

    zy = 1

    brightness = random.randint(1,2)/100.0  # Range from 0.01 to 0.02

    channel_shift_intensity = random.randint(1,10)/100.0  # Range from 0.01 to 0.1

    

    temp[i] = datagen.apply_transform(x_train_raw[image_to_test],{

        'tx':tx,

        'ty':ty,

        'shear':shear,

        'zx':zx,

        'zy':zy,

        'flip_horizontal':flip_horizontal,

        'flip_vertical':flip_vertical,

        #'brightness':brightness,

        #'channel_shift_intensity':channel_shift_intensity

        })

    plotImages(temp[0:16,]/255.0) 
