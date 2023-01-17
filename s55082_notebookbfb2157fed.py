import numpy as np

import pandas as pd

import tensorflow as tf

import matplotlib.pyplot as plt

from sklearn.model_selection import ShuffleSplit

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

LABELS = 10 # Number of different types of labels (1-10)

WIDTH = 28 # width / height of the image

CHANNELS = 1 # Number of colors in the image (greyscale)

VALID = 10000 # Validation data size

STEPS = 3500 #20000   # Number of steps to run

BATCH = 100 # Stochastic Gradient Descent batch size

PATCH = 5 # Convolutional Kernel size

DEPTH = 8 #32 # Convolutional Kernel depth size == Number of Convolutional Kernels

HIDDEN = 100 #1024 # Number of hidden neurons in the fully connected layer

LR = 0.001 # Learning rate
data = pd.read_csv('../input/train.csv') # Read csv file in pandas dataframe

labels = np.array(data.pop('label')) # Remove the labels as a numpy array from the dataframe

labels = LabelEncoder().fit_transform(labels)[:, None]

labels = OneHotEncoder().fit_transform(labels).todense()

data = StandardScaler().fit_transform(np.float32(data.values)) # Convert the dataframe to a numpy array

data = data.reshape(-1, WIDTH, WIDTH, CHANNELS) # Reshape the data into 42000 2d images

train_data, valid_data = data[:-VALID], data[-VALID:]

train_labels, valid_labels = labels[:-VALID], labels[-VALID:]

print('train data shape = ' + str(train_data.shape) + ' = (TRAIN, WIDTH, WIDTH, CHANNELS)')

print('labels shape = ' + str(labels.shape) + ' = (TRAIN, LABELS)')
data = pd.read_csv('../input/train.csv') 
data