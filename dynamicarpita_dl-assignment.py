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



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/labels.csv')

df_test = pd.read_csv('../input/sample_submission.csv')

targets_series = pd.Series(df_train['breed'])

one_hot = pd.get_dummies(targets_series, sparse = True)

one_hot_labels = np.asarray(one_hot)  
im_size = 300

y_train = []

y_test = []

x_train_raw = bcolz.zeros((0,im_size,im_size,3),np.float32)

x_test_raw = bcolz.zeros((0,im_size,im_size,3),np.float32)

i = 0 

for f in tqdm(df_train.id):

    # load an image from file

    image = load_img('../input/train/{}.jpg'.format(f), target_size=(im_size, im_size))

    image = img_to_array(image)

    # prepare the image for the VGG model

    #image = preprocess_input(image)

    label = one_hot_labels[i]

    x_train_raw.append(image)

    y_train.append(label)

    i += 1
%matplotlib inline

import matplotlib.pyplot as plt

def plotImages( images_arr, n_images=2):

    fig, axes = plt.subplots(1, n_images, figsize=(12,12))

    axes = axes.flatten()

    for img, ax in zip( images_arr, axes):

        ax.imshow(img)

        ax.set_xticks(())

        ax.set_yticks(())

    plt.tight_layout()

plotImages(x_train_raw[0:2,]/255.) 
#i=0

#for f in tqdm(df_test.id):

    # load an image from file

    #image = load_img('../input/test/{}.jpg'.format(f), target_size=(im_size, im_size))

    #image = img_to_array(image)

    # prepare the image for the VGG model

    #image = preprocess_input(image)

    #x_test_raw.append(image)

    #i += 1
#plotImages(x_test_raw[0:2,]/255.)
datagen = ImageDataGenerator(

    #featurewise_center=True,

    #featurewise_std_normalization=True,

    #rotation_range=20,

     width_shift_range=0.2,

     height_shift_range=0.2,

     #zoom_range=0.2,

     horizontal_flip=True,

     vertical_flip=True)



i = 0

for x_batch, y_batch in datagen.flow(x_train_raw, y_train, batch_size=2):

    plotImages(x_batch[0:2,]/255.)

    i += 1

    if i > 0:

        break