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
df_train = pd.read_csv('../input/labels.csv')

df_test = pd.read_csv('../input/sample_submission.csv')
df_train.head(10)
df_test.head(10)
%matplotlib inline

import matplotlib.pyplot as plt

from glob import glob

from mpl_toolkits.axes_grid1 import ImageGrid
train_files = glob('../input/train/*.jpg')

test_files = glob('../input/test/*.jpg')
plt.imshow(plt.imread(train_files[99]))
image_trial = plt.imread(train_files[99])

def crop_center(img,cropx,cropy):

    y,x = image_trial.shape[0:2]

    startx = x//2-(cropx//2)

    starty = y//2-(cropy//2)    

    plt.imshow(image_trial[starty:starty+cropy,startx:startx+cropx])

    

crop_center(image_trial,150,150)



def crop_left(img,cropx,cropy):

    y,x = image_trial.shape[0:2]

    startx = x//2-(cropx//2)

    starty = y//2-(cropy//2)    

    plt.imshow(image_trial[0:cropy,0:cropx])

    

crop_left(image_trial,150,150)



targets_series = pd.Series(df_train['breed'])

one_hot = pd.get_dummies(targets_series, sparse = True)

one_hot_labels = np.asarray(one_hot)
one_hot.head(10)
im_size = 300
y_train = []

y_val = []

x_train_raw = bcolz.zeros((0,im_size,im_size,3),np.float32)

x_val_raw = bcolz.zeros((0,im_size,im_size,3),np.float32)
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
print(x_train_raw.shape)

print(x_val_raw.shape)
class DataGenerator(keras.utils.Sequence):

    'Generates data for Keras'

    def __init__(self, data, labels, im_size, batch_size, shuffle=True, data_augment = False, test = False):

        'Initialization'

        self.batch_size = batch_size

        self.list_IDs = np.arange(0,data.shape[0])

        self.shuffle = shuffle

        if self.shuffle == True:

            np.random.shuffle(self.list_IDs)        

        self.data = data

        self.data_augment = data_augment

        self.test = test

        if self.test == False:

            self.labels = labels

        self.on_epoch_end()

        

    def __len__(self):

        'Denotes the number of batches per epoch'

        return int(np.floor(len(self.list_IDs) / self.batch_size))



    def __getitem__(self, index):

        'Generate one batch of data'

        # Generate indexes of the batch

        indexes = self.list_IDs[index*self.batch_size:(index+1)*self.batch_size]



        # Generate data

        if self.test == False:

            X, y = self.__data_generation(indexes)

            return preprocess_input(X), y

        else:

            X = self.__data_generation(indexes)

            return preprocess_input(X)

        

    def on_epoch_end(self):

        'Updates indexes after each epoch'

        if self.shuffle == True:

            np.random.shuffle(self.list_IDs)

            

    def __data_generation(self, list_IDs_temp):

        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)

        # Initialization

        X = np.zeros((list_IDs_temp.shape[0],im_size,im_size,3), dtype=np.float32)

        if self.test == False:

            y = np.zeros((list_IDs_temp.shape[0],self.labels.shape[1]), dtype=np.uint8)



        # Generate data

        for i, ID in enumerate(list_IDs_temp):

            

            if self.data_augment == True:

                if random.randint(1,101) < 20:

                    crop_center(x_train_raw[i],150,150)

                    x_train_raw.append(image)

                    y_train.append(label)

                    

                    if random.randint(1,101) < 50: 

                        flip_horizontal = True

                    else:

                        flip_horizontal = False

                    if random.randint(1,101) < 50: 

                        flip_vertical = True

                    else:

                        flip_vertical = False

                    tx = im_size*random.randint(1,2)/100.0

                    ty = im_size*random.randint(1,2)/100.0

                    shear = random.randint(1,10)/100.0

                    zx = random.randint(80,120)/100.0

                    zy = random.randint(80,120)/100.0

                    brightness = random.randint(1,2)/100.0

                    channel_shift_intensity = random.randint(1,10)/100.0

                    

                

                

                    X[i,] = self.data[ID,]

            else:

                # Store sample

                X[i,] = self.data[ID,]



            # Store class

            if self.test == False:

                y[i,] = self.labels[ID,]



        if self.test == False:

            return X, y

        else:

            return X
batch_size = 32





# Parameters

params_trn = {

          'im_size': im_size,

          'batch_size': batch_size,

          'shuffle': True,

          'data_augment' : True,

          'test' : False

         }

params_val = {

          'im_size': im_size,

          'batch_size': batch_size,

          'shuffle': True,

          'data_augment' : False,

          'test' : False

         }



# Generators

training_generator = DataGenerator(x_train_raw, y_train, **params_trn)

validation_generator = DataGenerator(x_val_raw, y_val, **params_val)