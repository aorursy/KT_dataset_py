# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tflearn.layers.core import input_data, dropout, fully_connected

from tflearn.layers.conv import conv_2d, max_pool_2d

from tflearn.layers.normalization import local_response_normalization

from tflearn.layers.estimator import regression

from tflearn.data_utils import load_csv

import tflearn





X,Y  = load_csv('../input/train.csv',target_column=0,categorical_labels=True, n_classes=10)

X = np.asarray(X)

testX,testY = load_csv('../input/test.csv',target_column=0,categorical_labels=True, n_classes=10)

testX = np.asarray(testX)

X = X.reshape([-1, 28, 28, 1])



network = input_data(shape=[None, 28, 28, 1], name='input')

network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")

network = max_pool_2d(network, 2)

network = local_response_normalization(network)

network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")

network = max_pool_2d(network, 2)

network = local_response_normalization(network)

network = fully_connected(network, 128, activation='tanh')

network = dropout(network, 0.8)

network = fully_connected(network, 256, activation='tanh')

network = dropout(network, 0.8)

network = fully_connected(network, 10, activation='softmax')

network = regression(network, optimizer='adam', learning_rate=0.01,

                     loss='categorical_crossentropy', name='target')



# Training

model = tflearn.DNN(network, tensorboard_verbose=0)

model.fit({'input': X}, {'target': Y}, n_epoch=3,

           snapshot_step=100, show_metric=True, run_id='convnet_mnist')





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

print(len(X[0]))

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.