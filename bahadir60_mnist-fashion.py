import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import datetime

import tensorflow as tf



from keras import backend as K

K.set_image_dim_ordering('th')

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Reshape, InputLayer

from keras.layers.normalization import BatchNormalization

from keras.layers.convolutional import Conv2D, MaxPooling2D

from sklearn.metrics import accuracy_score
#check for a GPU

if tf.test.gpu_device_name():

    print('Default GPU device is {}: '.format(tf.test.gpu_device_name()))

else:

    print("No GPU found. Please use GPU for train the model.")
#get data

from tensorflow.examples.tutorials.mnist import input_data

data = input_data.read_data_sets('./MNIST FASHION', one_hot=True, reshape=False)