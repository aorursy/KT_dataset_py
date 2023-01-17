import gc

import glob

import os

import cv2

import random

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import imageio as im

from keras import models

from keras.models import Sequential

from keras.layers import Activation, Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras.optimizers import adam

from keras.preprocessing import image

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ModelCheckpoint

from keras.utils import np_utils

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

import seaborn as sns

import matplotlib

from matplotlib import pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/"))



# Any results you write to the current directory are saved as output.
# load images dataset

def loadImagesData(glob_path):

    images = []

    names = []

    for img_path in glob.glob(glob_path):

        # load/resize images with cv2

        names.append(os.path.basename(img_path))

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        images.append(img) # already 32x32

    return (images,names)

# map of training label to list of images

trainData = {}

namesData = {}

for label in os.listdir('../input/train/'):

    (images,names) = loadImagesData(f"../input/train/{label}/*.jpg")

    trainData[label] = images

    namesData[label] = names

print("train labels:", ",".join(trainData.keys()))

print(len(trainData['train']))

# show some data

plt.figure(figsize=(4,2))

columns = 4

for i in range(0,8):

    plt.subplot(8 / columns + 1, columns, i + 1)

    plt.imshow(trainData['train'][i])

plt.show()
train_meta = pd.read_csv('../input/train.csv')

print(train_meta.shape)

print(train_meta.has_cactus.value_counts())

# lookup table of name to has_cactus

lookupY = {}

for i in range(0,len(train_meta)):

    row = train_meta.iloc[i,:]

    lookupY[row.id] = row.has_cactus

train_meta.head()
# build x/y dataset

trainList = []

maxCount = 4364 # number of has_cactus = 0

counts = {'0':0,'1':0}

for (i,image) in enumerate(trainData['train']):

    label = lookupY[namesData['train'][i]]

    counts[str(label)] = 1 + counts[str(label)]

    if counts[str(label)] < maxCount:

        trainList.append({

            'label': label,

            'data': image

        })

# shuffle dataset

random.shuffle(trainList)

# dataframe and display

train_df = pd.DataFrame(trainList)

gc.collect()

print(train_df.shape)

print(train_df.label.value_counts())

train_df.head()
# encode training data

data_stack = np.stack(train_df['data'].values)

dfloats = data_stack.astype(np.float)

all_x = np.multiply(dfloats, 1.0 / 255.0)

all_x.shape
all_y = np.array(train_df.label).astype(np.float)

all_y[0:5]
# split test/training data

train_x,test_x,train_y,test_y=train_test_split(all_x,all_y,test_size=0.2,random_state=7)

print(train_x.shape,test_x.shape)
# x,y and rotation data augmentation

datagen = ImageDataGenerator(

    featurewise_center=False,  # set input mean to 0 over the dataset

    samplewise_center=False,  # set each sample mean to 0

    featurewise_std_normalization=False,  # divide inputs by std of the dataset

    samplewise_std_normalization=False,  # divide each input by its std

    rotation_range=60,  # randomly rotate images in the range (degrees, 0 to 180)

    zoom_range=0.2, # zoom images

    horizontal_flip=True,  # randomly flip images

    vertical_flip=True)  # randomly flip images

datagen.fit(train_x)
# create the network

num_filters = 8

input_shape = train_x.shape[1:]

output_shape = 1

# model

m = Sequential()

def tdsNet(m):

    m.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=input_shape))

    m.add(Conv2D(16, kernel_size=3, activation='relu'))

    m.add(Flatten())

    m.add(Dense(units = output_shape, activation='sigmoid'))

tdsNet(m)

# compile adam with decay and use binary_crossentropy for single category dataset

m.compile(optimizer = 'nadam',

          loss = 'binary_crossentropy', 

          metrics = ['accuracy'])

# show summary

m.summary()
# train model

batch_size = 32

history = m.fit_generator(datagen.flow(train_x, train_y,

                          batch_size=batch_size),

                          steps_per_epoch= (train_x.shape[0] // batch_size),

                          epochs = 4,

                          validation_data=(test_x, test_y),

                          workers=4)
# create the network

num_filters = 8

input_shape = train_x.shape[1:]

output_shape = 1

# model

m = Sequential()

def cnnNet(m):

    m.add(Conv2D(30, kernel_size=3, activation='relu', input_shape=input_shape))

    m.add(MaxPooling2D(2,2))

    m.add(Conv2D(15, kernel_size=3, activation='relu'))

    m.add(MaxPooling2D(2,2))

    m.add(Dense(7, activation='relu')) # <7 stops working, but higher values do nothing

    m.add(Flatten())

    m.add(Dense(units = output_shape, activation='sigmoid'))

cnnNet(m)

# compile adam with decay and use binary_crossentropy for single category dataset

m.compile(optimizer = 'nadam',

          loss = 'binary_crossentropy', 

          metrics = ['accuracy'])

# show summary

m.summary()
# train model

batch_size = 32

history = m.fit_generator(datagen.flow(train_x, train_y,

                          batch_size=batch_size),

                          steps_per_epoch= (train_x.shape[0] // batch_size),

                          epochs = 4,

                          validation_data=(test_x, test_y),

                          workers=4)
# build complete x/y dataset

trainList = []

for (i,image) in enumerate(trainData['train']):

    label = lookupY[namesData['train'][i]]

    trainList.append({

        'label': label,

        'data': image

    })

# shuffle dataset

random.shuffle(trainList)

# dataframe and display

train_df = pd.DataFrame(trainList)

gc.collect()

# encode training data

data_stack = np.stack(train_df['data'].values)

dfloats = data_stack.astype(np.float)

all_x = np.multiply(dfloats, 1.0 / 255.0)

all_x.shape

all_y = np.array(train_df.label).astype(np.float)

# split test/training data

train_x,test_x,train_y,test_y=train_test_split(all_x,all_y,test_size=0.2,random_state=7)

print(train_x.shape,test_x.shape)
# continue training model

batch_size = 64

history = m.fit_generator(datagen.flow(train_x, train_y,

                          batch_size=batch_size),

                          steps_per_epoch= (train_x.shape[0] // batch_size),

                          epochs = 4,

                          validation_data=(test_x, test_y),

                          workers=4)
# check sample submission format

pd.read_csv('../input/sample_submission.csv').head()
# output predicted submission csv

(test_images, test_names) = loadImagesData(f"../input/test/test/*.jpg")

data_stack = np.stack(test_images)

dfloats = data_stack.astype(np.float32)

unknown_x = np.multiply(dfloats, 1.0 / 255.0)

# predict

predicted = np.ravel(m.predict(unknown_x))

submission_df = pd.DataFrame({'id':test_names,'has_cactus':predicted})

submission_df.to_csv('submission.csv', index=False)

len(submission_df)