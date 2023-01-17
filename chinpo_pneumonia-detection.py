# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import os

import numpy as np

import pandas as pd 

import random

import cv2

import matplotlib.pyplot as plt

import keras.backend as K

import tensorflow as tf

import warnings



from random import shuffle 

from tqdm import tqdm 

from PIL import Image

from keras.models import Model, Sequential

from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization, Conv2D, SeparableConv2D, MaxPool2D, LeakyReLU, Activation, ReLU

from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.preprocessing import MinMaxScaler

from sklearn import svm

from glob import glob

from matplotlib import pyplot as plt



path_train = "../input/chest-xray-pneumonia/chest_xray/train"

path_val = "../input/chest-xray-pneumonia/chest_xray/val"

path_test = "../input/chest-xray-pneumonia/chest_xray/test"
img = glob(path_train+"/PNEUMONIA/*.jpeg")
plt.imshow(np.asarray(plt.imread(img[0])))
classes = ["NORMAL", "PNEUMONIA"]

img_height = 299

img_width = 299



train_data = glob(path_train+"/NORMAL/*.jpeg")

train_data += glob(path_train+"/PNEUMONIA/*.jpeg")



train_datagen = ImageDataGenerator(rescale=1./255,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True) # set validation split



train_generator = train_datagen.flow_from_directory(

    path_train,

    target_size=(img_height, img_width),

    batch_size=24,

    classes=classes,

    class_mode='categorical') # set as training data



validation_generator = train_datagen.flow_from_directory(

    path_val, # same directory as training data

    target_size=(img_height, img_width),

    batch_size=24,

    classes=classes,

    class_mode='categorical') # set as validation data



test_generator = train_datagen.flow_from_directory(

    path_test, # same directory as training data

    target_size=(img_height, img_width),

    batch_size=24,

    classes=classes,

    class_mode='categorical') # set as validation data

train_generator.image_shape
import skimage

from skimage.transform import resize

def plotter(i):

    Pimages = os.listdir(path_train + "/PNEUMONIA")

    Nimages = os.listdir(path_train + "/NORMAL")

    imagep1 = cv2.imread(path_train+"/PNEUMONIA/"+Pimages[i])

    imagep1 = skimage.transform.resize(imagep1, (150, 150, 3) , mode = 'reflect')

    imagen1 = cv2.imread(path_train+"/NORMAL/"+Nimages[i])

    imagen1 = skimage.transform.resize(imagen1, (150, 150, 3))

    pair = np.concatenate((imagen1, imagep1), axis=1)

    print("(Left) - No Pneumonia Vs (Right) - Pneumonia")

    print("-----------------------------------------------------------------------------------------------------------------------------------")

    plt.figure(figsize=(10,5))

    plt.imshow(pair)

    plt.show()

for i in range(5,10):

    plotter(i)
from keras.applications.resnet50 import ResNet50



base_model = ResNet50(weights='imagenet', include_top=False , input_tensor =Input(shape=(299,299,3)), input_shape=(299,299,3))

model = Flatten(name='flatten')(base_model.output)

model = Dense(1024, activation='relu')(model)

model = Dropout(0.7, name='dropout1')(model)

model = Dense(512, activation='relu')(model)

model = Dropout(0.5, name='dropout2')(model)

predictions = Dense(2, activation='softmax')(model)
conv_model = Model(inputs=base_model.input, outputs=predictions)

opt = Adam(lr=0.0001, decay=1e-5)

conv_model.compile(loss='binary_crossentropy', metrics=['accuracy'],optimizer=opt)

print(conv_model.summary())
for layer in conv_model.layers[:-6]:

    layer.trainable = False

conv_model.compile(loss='binary_crossentropy', metrics=['accuracy'],optimizer=opt)

history = conv_model.fit_generator(epochs=5, shuffle=True, validation_data=validation_generator, generator=train_generator, steps_per_epoch=500, validation_steps=10,verbose=2)