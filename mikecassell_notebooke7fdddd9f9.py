# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from PIL import Image

import tensorflow as tf

import tflearn

from tflearn import *

from tflearn.data_utils import image_preloader

import os

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



#Image.open(glob("../input/test/*")[1])

# Any results you write to the current directory are saved as output.
# Generate our index file for reading and processing

TRAIN_DIR = '../input/train/'

TEST_DIR = '../input/test/'

labels = [1 if 'cat' in f else 0 for f in os.listdir(TRAIN_DIR)]

files = os.listdir(TRAIN_DIR)



with open('temp.txt', 'w') as f:

    for i in range(0, len(labels)):

        f.write(files[i] + ' ' + str(labels[i]) + '\n')
x, y = image_preloader(target_path='temp.txt', mode='file', 

                       image_shape=(64,64), categorical_labels=True, normalize=True)
def buildNetwork():

    network = input_data(shape=[None, 64, 64, 3])

    network = conv_2d(network, 32, 3, activation='relu')

    network = max_pool_2d(network, 2)

    network = conv_2d(network, 64, 3, activation='relu')

    network = conv_2d(network, 64, 3, activation='relu')

    network = max_pool_2d(network, 2)

    network = fully_connected(network, 512, activation='relu')

    network = dropout(network, 0.5)

    network = fully_connected(network, 2, activation='softmax')

    network = regression(network, optimizer='adam',

                         loss='categorical_crossentropy',

                         learning_rate=0.001)

    return network
network = buildNetwork()

model = tflearn.DNN(network, tensorboard_verbose=0)



col = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

for x in col:

    tf.add_to_collection(tf.GraphKeys.VARIABLES, x )    

    

model.fit(x, y, n_epoch=50, shuffle=True, validation_set=0.1,

          show_metric=True, batch_size=96, run_id='testrun1')
