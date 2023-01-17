import pandas as pd

import numpy as np

import keras.layers.core as core

import keras.layers.convolutional as conv

import keras.models as models

import keras.utils.np_utils as kutils



# The competition datafiles are in the directory ../input

# Read competition data files:

train = pd.read_csv("../input/train.csv").values

test  = pd.read_csv("../input/test.csv").values
nb_epoch = 1 # Change to 100



batch_size = 128

img_rows, img_cols = 28, 28



nb_filters_1 = 32 # 64

nb_filters_2 = 64 # 128

nb_filters_3 = 128 # 256

nb_conv = 3



trainX = train[:, 1:].reshape(train.shape[0], img_rows, img_cols, 1)

trainX = trainX.astype(float)

trainX /= 255.0



trainY = kutils.to_categorical(train[:, 0])

nb_classes = trainY.shape[1]
trainX.shape