# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

np.random.seed(1984)



import os

import glob

import cv2

import datetime

import pandas as pd

import time

import warnings

warnings.filterwarnings("ignore")



from sklearn.cross_validation import KFold

from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Flatten

from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D

from keras.optimizers import SGD, Adagrad

from keras.callbacks import EarlyStopping

from keras.utils import np_utils

from keras.constraints import maxnorm

from sklearn.metrics import log_loss

from keras import __version__ as keras_version

from collections import Counter



import keras as k

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D
path1 = '/home/amit/Kaggle/planet'

trainPath = '/train-jpg'

testPath = '/test-jpg'



PIC_SIZE = 32
# read Y_train



try:

    Y_train = pd.read_csv(path1+'/train.csv')

except:

    path1 = '/home/amit/Kaggle/planet'

    Y_train = pd.read_csv(path1+'/train.csv')



print (Y_train[0:5])



flatten = lambda l: [item for sublist in l for item in sublist]

labels = list(set(flatten([l.split(' ') for l in Y_train['tags'].values])))

label_map = {l: i for i, l in enumerate(labels)}

inv_label_map = {i: l for l, i in label_map.items()}

print(label_map)

print

print(inv_label_map)



Y_trainDict = {}

for i, row in Y_train.iterrows():

    name = row['image_name']

    tags = row['tags']

    targets = np.zeros(17)

    for t in tags.split(' '):

        targets[label_map[t]] = 1 

    Y_trainDict[name] = targets



print (Y_trainDict['train_0'])

print (Y_trainDict['train_1'])

print (Y_trainDict['train_2'])