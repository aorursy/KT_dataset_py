import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import glob

import os

import cv2

from skimage import color

from skimage import io

import tensorflow as tf

import tflearn

from tflearn.layers.core import input_data, dropout, fully_connected

from tflearn.layers.conv import conv_2d, max_pool_2d

from tflearn.layers.normalization import local_response_normalization

from tflearn.layers.estimator import regression

import random

import csv as csv

import numpy as np

%matplotlib inline 





# Any results you write to the current directory are saved as output.
TRAIN_DIR = '../input/train/'

TEST_DIR = '../input/test/'

NUMBER_OF_TRAIN_DATA = 10



def read_image(file_path):        

    img= cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    img= cv2.resize(img, (150, 150), interpolation=cv2.INTER_CUBIC)

    return img



def prep_data(images):

    count = len(images)

    data = np.ndarray((count,1, 150, 150), dtype=np.uint8)



    for i, image_file in enumerate(images):

        image = read_image(image_file)

        data[i] = image.T

        if i%250 == 0: print('Processed {} of {}'.format(i, count))

    

    return data



train_cats = sorted(glob.glob(os.path.join(TRAIN_DIR, 'cat*.jpg')))

train_dogs = sorted(glob.glob(os.path.join(TRAIN_DIR, 'dog*.jpg')))

train_all = train_dogs+train_cats 



random.Random(4).shuffle(train_all)



test_all = sorted(glob.glob(os.path.join(TEST_DIR, '*.jpg')))



X_train = prep_data([path for path in train_all[0:NUMBER_OF_TRAIN_DATA]] )

Y_train = np.array([1. if 'dog' in name else 0. for name in train_all[0:NUMBER_OF_TRAIN_DATA]])



X_test = [read_image(path) for path in test_all] 

Y_test = np.array([1. if 'dog' in name else 0. for name in test_all])

print(X_train.shape())

print(Y_train.shape())
tf.reset_default_graph()



net = input_data(shape=[None, 1,150, 150], name='input')

net = conv_2d(net, 32, 3, activation='relu', regularizer="L2")

net = max_pool_2d(net, 2)

net = local_response_normalization(net)

net = conv_2d(net, 64, 3, activation='relu', regularizer="L2")

net = max_pool_2d(net, 2)

net = local_response_normalization(net)

net = fully_connected(net, 128, activation='tanh')

net = dropout(net, 0.8)

net = fully_connected(net, 256, activation='tanh')

net = dropout(net, 0.8)

net = fully_connected(net, 10, activation='softmax')

net = regression(net, optimizer='adam', learning_rate=0.01,

                     loss='categorical_crossentropy', name='target')

model = tflearn.DNN(net, tensorboard_verbose=0)

model.fit(X_train, Y_train, n_epoch=1,

           validation_set=({'input': X_test}, {'target': Y_test}),

           snapshot_step=100, show_metric=True, run_id='convnet_mnist')
print(model.predict(np.reshape(X_test[0], (-1, 22500)))[0][0])
myfile = open('submission.csv', 'w')

wr = csv.writer(myfile, quoting=csv.QUOTE_NONE,quotechar='',escapechar='\\')

wr.writerow(["id","label"])



for i in range(0,len(X_test)):

     wr.writerow([i+1,float(model.predict(np.reshape(X_test[i], (-1, 22500)))[0][0])])