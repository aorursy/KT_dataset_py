# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

# print(os.listdir("../input"))



work_dir = '/kaggle/input'

data_dir = '/kaggle/input'

print(os.listdir(data_dir))



# Any results you write to the current directory are saved as output.
import sys

import os

import glob

import pickle

import numpy as np

#from time import time



import tensorflow as tf



from tensorflow import keras



from sklearn.model_selection import train_test_split
def extract_pickle(file0):

    with open(file0,'rb') as file1:

        file2 = pickle.load(file1, encoding='bytes')



    # Convert bytes of dictionary to string and numbers

    file3 = {ff.decode('ascii'):file2.get(ff) for ff in file2.keys()}



    return file3
def collect_all_pickles(fileZ):

    # Read the first batch

    dbi = extract_pickle(f0[0])



    # Aggregate of all the training batches starts with the first batch

    X_read, y_read, f_read = dbi['data'], dbi['labels'], dbi['filenames']



    # Add all the remaining batches to the append/concatenate

    for i in range(1,5):

        dbi = extract_pickle(f0[i])

        #

        X_read = np.concatenate((X_read, dbi['data']))

        y_read += dbi['labels']

        f_read += dbi['filenames']



    return X_read, y_read, f_read
def normalize_images(X_img):

    X_gim = np.zeros_like(X_img, dtype=np.float64)

    ni = X_img.shape[0]



    for i in range(ni):

        b0 = X_img[i,:]

        bmin, bmax = np.min(b0), np.max(b0)



        X_gim[i,:] = (b0 - bmin) / (bmax - bmin)



    X_gim = X_gim.reshape((ni,32,32,3), order='F')



    return X_gim
def one_hot_encode(y_img):

    y_gim = np.zeros((len(y_img),10), dtype=np.int8)



    for idx,val in enumerate(y_img):

        y_gim[idx,val] = 1



    return y_gim
# Number of cases per batch

mi = 10000



# Make label_names as a separate list

labnam = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']



# array to store filenames of train batch files

f0 = glob.glob(os.path.join(data_dir, 'data*'))



# Collect all the pickled batched together in X and y train samples

X_trn, y_trn, f_trn = collect_all_pickles(f0)



m, n = X_trn.shape





os.chdir(data_dir)



# array to store filenames of test batch files

g0 = glob.glob(os.path.join('test*'))



# # Extract the pickled file as a dictionary, and generate X and y test samples

dbi = extract_pickle(g0[0])

X_tst, y_tst, f_tst = dbi['data'], dbi['labels'], dbi['filenames']



os.chdir(work_dir)
# Normalize all images and convert those from uint8 to float

X_trn = normalize_images(X_trn)

X_tst = normalize_images(X_tst)



# one-hot-encode the labels

y_trn = one_hot_encode(y_trn)

y_tst = one_hot_encode(y_tst)
# Hyperparameters

epoch = 10         # number of epochs

alpha = 0.001      # learning rate, alpha

batch = 128        # size of each batch

kprob = 0.3        # keep_prob for Dropout in a layer
model = tf.keras.models.Sequential()



# Convolution layer with 64 filters of size 3x3

model.add(tf.keras.layers.Conv2D(64, (3,3), input_shape=[32,32,3], \

            strides=(1,1), padding='same', activation=tf.nn.relu))

model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), padding='same'))

model.add(tf.keras.layers.BatchNormalization())



# Convolution layer with 128 filters of size 3x3

model.add(tf.keras.layers.Conv2D(128, (3,3), 

            strides=(1,1), padding='same', activation=tf.nn.relu))

model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), padding='same'))

model.add(tf.keras.layers.BatchNormalization())



# Convolution layer with 256 filters of size 3x3

model.add(tf.keras.layers.Conv2D(256, (3,3), 

            strides=(1,1), padding='same', activation=tf.nn.relu))

model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), padding='same'))

model.add(tf.keras.layers.BatchNormalization())



# Convolution layer with 512 filters of size 3x3

model.add(tf.keras.layers.Conv2D(512, (3,3), 

            strides=(1,1), padding='same', activation=tf.nn.relu))

model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), padding='same'))

model.add(tf.keras.layers.BatchNormalization())



# Flatten

model.add(tf.keras.layers.Flatten())



# Dense layer with 128 output neurons

model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

model.add(tf.keras.layers.Dropout(kprob))

model.add(tf.keras.layers.BatchNormalization())



# Dense layer with 256 output neurons

model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))

model.add(tf.keras.layers.Dropout(kprob))

model.add(tf.keras.layers.BatchNormalization())



# Dense layer with 512 output neurons

model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu))

model.add(tf.keras.layers.Dropout(kprob))

model.add(tf.keras.layers.BatchNormalization())



# Dense layer with 1024 output neurons

model.add(tf.keras.layers.Dense(1024, activation=tf.nn.relu))

model.add(tf.keras.layers.Dropout(kprob))

model.add(tf.keras.layers.BatchNormalization())



model.add(tf.keras.layers.Dense(10, activation='softmax'))
model.summary()
coms = model.compile(keras.optimizers.SGD(lr=alpha), loss='categorical_crossentropy', \

              metrics=['accuracy'])
fits = model.fit(X_trn, y_trn, validation_split=0.20, batch_size=32, epochs=100, verbose=1)
print(fits.history.keys())
import matplotlib.pyplot as plt

fig = plt.figure()

fig.set_size_inches(9,4)

ax = fig.add_subplot(1,2,1)

plt.plot(fits.history['loss'], 'r-', label='loss');  plt.plot(fits.history['val_loss'], 'b-', label='val. loss')

plt.legend()

ax = fig.add_subplot(1,2,2)

plt.plot(fits.history['acc'], 'r-', label='accuracy');  plt.plot(fits.history['val_acc'], 'b-', label='val. accuracy')

plt.legend()