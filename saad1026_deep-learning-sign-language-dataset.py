import math

import numpy as np

import h5py

import matplotlib.pyplot as plt

import scipy

from PIL import Image

from scipy import ndimage

import tensorflow as tf

from tensorflow.python.framework import ops

from keras import layers

from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D

from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D

from keras.models import Model

from keras.preprocessing import image

from keras.utils import layer_utils

from keras.utils.data_utils import get_file

from keras.applications.imagenet_utils import preprocess_input

import pydot

from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot

from keras.utils import plot_model

#from kt_utils import *

import keras.backend as K

K.set_image_data_format('channels_last')





%matplotlib inline

np.random.seed(1)

# Input data files are available in the "../input/" directory.

from subprocess import check_output

print(check_output(["ls","../input/Sign-language-digits-dataset"]).decode('utf8'))

X_1 = np.load('../input/Sign-language-digits-dataset/X.npy')

Y_1 = np.load('../input/Sign-language-digits-dataset/Y.npy')

print(X_1.shape)

img_size=64

plt.subplot(1,2,1)

plt.imshow(X_1[201].reshape(img_size,img_size))

plt.axis('off')

plt.subplot(1,2,2)

plt.imshow(X_1[900].reshape(img_size,img_size))

plt.axis('off')
# Join a sequence of arrays along an row axis.

X=np.concatenate((X_1[204:409],X_1[822:1027]),axis=0) # from 0 to 204 is zero sign and from 205 to 410 is one sign 

z= np.zeros(205)

o= np.ones(205)

Y= np.concatenate((z,o),axis=0).reshape(X.shape[0],1)

print('X shape: ',X.shape)

print('Y shape: ',Y.shape)
X_1[822:1027].shape
# Then lets create x_train, y_train, x_test, y_test arrays

from sklearn.model_selection import train_test_split



X_train, X_test, Y_train, Y_test = train_test_split(X_1, Y_1, test_size= 0.20, random_state=0)

#X_dev, X_test, Y_dev, Y_test = train_test_split(X_test1, Y_test1, test_size= 0.20, random_state=0)

num_of_train= X_train.shape[0]

#num_of_dev = X_dev.shape[0]

num_of_test= X_test.shape[0]

X_train = X_train/255

#X_dev = X_dev/255

X_test = X_test/255

X_train = X_train.reshape(num_of_train,img_size,img_size,1)

#X_dev = X_dev.reshape(num_of_dev,img_size,img_size,1)

X_test = X_test.reshape(num_of_test,img_size,img_size,1)

print(X_test.shape)
def HappyModel(input_shape):

    X_input = Input(input_shape)



    # Zero-Padding: pads the border of X_input with zeroes

    X = X_input



    # CONV -> BN -> RELU Block applied to X

    X = Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0')(X)

    X = BatchNormalization(axis = 3, name = 'bn0')(X)

    X = Activation('relu')(X)



    # MAXPOOL

    X = MaxPooling2D((2, 2), name='max_pool0')(X)

    

    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED

    X = Flatten()(X)

    X = Dense(10, activation='sigmoid', name='fc')(X)



    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.

    model = Model(inputs = X_input, outputs = X, name='HappyModel')

    

    ### END CODE HERE ###

    

    return model
happyModel = HappyModel(X_train.shape[1:])
happyModel.compile(optimizer = "adam", loss = "categorical_hinge", metrics = ["accuracy"])
happyModel.fit(x = X_train, y = Y_train, epochs = 40, batch_size = 60)
preds = happyModel.evaluate(x = X_test, y = Y_test)

print()

print ("Loss = " + str(preds[0]))

print ("Test Accuracy = " + str(preds[1]))