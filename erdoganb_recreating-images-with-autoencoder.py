# # This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import os

import math

from PIL import Image

from scipy import misc

from keras.layers import Input, Dense

from keras.models import Model

import matplotlib.pyplot as plt

from random import shuffle



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

# init some vars

train_raw = []

train = []

test = []



img_batch_size = 8

test_ratio = 0.2

# gather all images in cleaned dataset in to the train_raw

files = os.listdir("../input/train_cleaned")

for filename in files:

    if filename.find('.png') > -1:

        img = misc.imread('../input/train_cleaned/' + filename)

        train_raw.append(img)



img = filename = files = None
# iterate images and split them by img_batch_size*img_batch_size pieces, put in to the train var

for img in train_raw:

    d0_batch = math.ceil(img.shape[0] / img_batch_size)

    d1_batch = math.ceil(img.shape[1] / img_batch_size)

    img = img / 255



    for d0_i in range(d0_batch):

        for d1_i in range(d1_batch):

            img_batch = img[d0_i*img_batch_size: (d0_i + 1)*img_batch_size,

                            d1_i*img_batch_size: (d1_i + 1)*img_batch_size]

            if img_batch.size == img_batch_size*img_batch_size:

                train.append(img_batch)

    

img_batch = d0_batch = d1_batch = d0_i = d1_i = None
# shuffle, turn in to a numpy array, split test/train dataset by test_ratio

shuffle(train)

train = np.array(train)



test = train[:int(train.shape[0]*test_ratio),:,:]

train = train[int(train.shape[0]*test_ratio):,:,:]



train = train.reshape((len(train), np.prod(train.shape[1:])))

test = test.reshape((len(test), np.prod(test.shape[1:])))
# size of hidden layer of autoencoder

encoding_dim = int(img_batch_size*img_batch_size / 4)

# input placeholder

input_img = Input(shape=(img_batch_size*img_batch_size,))

# hidden layer

encoded = Dense(encoding_dim, activation='relu')(input_img)

# output layer

decoded = Dense(img_batch_size*img_batch_size, activation='sigmoid')(encoded)

# put them in a Model

autoencoder = Model(input_img, decoded)



# compile and train

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.fit(train, train,

                verbose=0,

                epochs=100,

                batch_size=8192,

                validation_data=(test,test))

# Uncomment this if you want to observe losses

# print(autoencoder.history.history)
# gather dirty test data

test_dirty_raw = []

files = os.listdir("../input/test")

i = 0

for filename in files:

    if (filename.find('.png') > -1 and i < 1):

        img = misc.imread('../input/test/' + filename)

        test_dirty_raw.append(img)



img = filename = files = None
test_img = np.array(test_dirty_raw[np.random.randint(len(test_dirty_raw))]) / 255



d0_batch = math.ceil(test_img.shape[0] / img_batch_size)

d1_batch = math.ceil(test_img.shape[1] / img_batch_size)



test_img_p = np.ones(test_img.shape)



#

for d0_i in range(d0_batch):

    for d1_i in range(d1_batch):

        img_batch = test_img[d0_i*img_batch_size: (d0_i + 1)*img_batch_size,

                             d1_i*img_batch_size: (d1_i + 1)*img_batch_size]



        if img_batch.size == img_batch_size*img_batch_size:

            img_batch = np.array(img_batch.reshape(1,img_batch.size))



            img_batch_p = autoencoder.predict(img_batch)

            img_batch_p = img_batch_p.reshape((img_batch_size,img_batch_size))



            test_img_p[d0_i*img_batch_size: (d0_i + 1)*img_batch_size,

                       d1_i*img_batch_size:(d1_i + 1)*img_batch_size] = img_batch_p



img_batch = d0_batch = d1_batch = d0_i = d1_i = None



# draw test and recreated images

plt.figure(figsize=(10, 10))



ax = plt.subplot(2, 1, 1)

plt.imshow(test_img * 255)

plt.gray()

ax.get_xaxis().set_visible(False)

ax.get_yaxis().set_visible(False)



ax = plt.subplot(2, 1, 2)

plt.imshow(test_img_p * 255)

plt.gray()

ax.get_xaxis().set_visible(False)

ax.get_yaxis().set_visible(False)



plt.show()