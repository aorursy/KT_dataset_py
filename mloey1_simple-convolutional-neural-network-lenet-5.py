from __future__ import division, print_function, absolute_import

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import tflearn
from tflearn.data_utils import to_categorical
import tflearn.data_utils as du
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
# read training & testing data

trainx = pd.read_csv("../input/csvTrainImages 60k x 784.csv",header=None)
trainy = pd.read_csv("../input/csvTrainLabel 60k x 1.csv",header=None)

testx = pd.read_csv("../input/csvTestImages 10k x 784.csv",header=None)
testy = pd.read_csv("../input/csvTestLabel 10k x 1.csv",header=None)
trainx.head()
testx.head()
# Split data into training set and validation set
#training images
trainx = trainx.values.astype('float32')
#training labels
trainy = trainy.values.astype('int32')

#testing images
testx = testx.values.astype('float32')
#testing labels
testy = testy.values.astype('int32')
trainx[0]
#One Hot encoding of train labels.
trainy = to_categorical(trainy,10)

#One Hot encoding of test labels.
testy = to_categorical(testy,10)
trainy[0]
print(trainx.shape, trainy.shape, testx.shape, testy.shape)
# reshape input images to 28x28x1
trainx = trainx.reshape([-1, 28, 28, 1])
testx = testx.reshape([-1, 28, 28, 1])
print(trainx.shape, trainy.shape, testx.shape, testy.shape)
#Zero center every sample with specified mean. If not specified, the mean is evaluated over all samples.
trainx, mean1 = du.featurewise_zero_center(trainx)
testx, mean2 = du.featurewise_zero_center(testx)
print(trainx.shape, trainy.shape, testx.shape, testy.shape)
# Building convolutional network

network = input_data(shape=[None, 28, 28, 1], name='input')

network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")

network = max_pool_2d(network, 2)

network = local_response_normalization(network)

network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")

network = max_pool_2d(network, 2)

network = local_response_normalization(network)

network = fully_connected(network, 256, activation='relu')

network = dropout(network, 0.8)

network = fully_connected(network, 128, activation='relu')

network = dropout(network, 0.8)

network = fully_connected(network, 10, activation='softmax')

network = regression(network, optimizer='sgd', learning_rate=0.01,

                     loss='categorical_crossentropy', name='target')
#model complile
model = tflearn.DNN(network, tensorboard_verbose=0)
#model fitting
model.fit({'input': trainx}, {'target': trainy}, n_epoch=20,
           validation_set=({'input': testx}, {'target': testy}),
           snapshot_step=100, show_metric=True, run_id='convnet_arabic_digits')
# Evaluate model
score = model.evaluate(testx, testy)
print('Test accuarcy: %0.2f%%' % (score[0] * 100))
