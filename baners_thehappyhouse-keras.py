# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math

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



import keras.backend as K

K.set_image_data_format('channels_last')

import matplotlib.pyplot as plt

from matplotlib.pyplot import imshow



%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import h5py

import os

for dirname, _, filenames in os.walk('/kaggle/input/happy-house-dataset'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
def load_dataset():

    train_dataset = h5py.File('/kaggle/input/happy-house-dataset/train_happy.h5', "r")

    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features

    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels



    test_dataset = h5py.File('/kaggle/input/happy-house-dataset/test_happy.h5', "r")

    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features

    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels



    classes = np.array(test_dataset["list_classes"][:]) # the list of classes

    

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))

    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes



X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
# Normalize image vectors

X_train = X_train_orig/255.

X_test = X_test_orig/255.



# Reshape

Y_train = Y_train_orig.T

Y_test = Y_test_orig.T



print ("number of training examples = " + str(X_train.shape[0]))

print ("number of test examples = " + str(X_test.shape[0]))

print ("X_train shape: " + str(X_train.shape))

print ("Y_train shape: " + str(Y_train.shape))

print ("X_test shape: " + str(X_test.shape))

print ("Y_test shape: " + str(Y_test.shape))
def HappyModel(input_shape):

    """

    Implementation of the HappyModel.

    

    Arguments:

    input_shape -- shape of the images of the dataset



    Returns:

    model -- a Model() instance in Keras

    """



    X_input = Input(input_shape)



    # Zero-Padding: pads the border of X_input with zeroes

    X = ZeroPadding2D((3, 3))(X_input)



    # CONV -> BN -> RELU Block applied to X

    X = Conv2D(32, (7, 7), strides=(1, 1), name='conv0')(X)

    X = BatchNormalization(axis=3, name='bn0')(X)

    X = Activation('relu')(X)



    # MAXPOOL

    X = MaxPooling2D((2, 2), name='max_pool')(X)



    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED

    X = Flatten()(X)

    X = Dense(1, activation='sigmoid', name='fc')(X)



    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.

    model = Model(inputs=X_input, outputs=X, name='HappyModel')



    return model





happyModel = HappyModel(X_train.shape[1:])
happyModel.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
happyModel.fit(X_train, Y_train, epochs=40, batch_size=50)
preds = happyModel.evaluate(X_test, Y_test, batch_size=32, verbose=1, sample_weight=None)

print()

print ("Loss = " + str(preds[0]))

print ("Test Accuracy = " + str(preds[1]))
img_path = '/kaggle/input/happysuvro/suvro_run.jpg'



img = image.load_img(img_path, target_size=(64, 64))

imshow(img)



x = image.img_to_array(img)

x = np.expand_dims(x, axis=0)

x = preprocess_input(x)



print(happyModel.predict(x))
happyModel.summary()
plot_model(happyModel, to_file='HappyModel.png')

SVG(model_to_dot(happyModel).create(prog='dot', format='svg'))