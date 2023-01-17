# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# import the necessary packages

from keras.models import Sequential

from keras.layers.normalization import BatchNormalization

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from keras.layers.core import Activation

from keras.layers.core import Flatten

from keras.layers.core import Dropout

from keras.layers.core import Dense

from keras import backend as K



class SmallerVGGNet:

    @staticmethod

    def build(width, height, depth, classes, finalAct="softmax"):

        # initialize the model along with the input shape to be

        # "channels last" and the channels dimension itself

        model = Sequential()

        inputShape = (height, width, depth)

        chanDim = -1



        # if we are using "channels first", update the input shape

        # and channels dimension

        if K.image_data_format() == "channels_first":

            inputShape = (depth, height, width)

            chanDim = 1



        # CONV => RELU => POOL

        model.add(Conv2D(32, (3, 3), padding="same",input_shape=inputShape))

        model.add(Activation("relu"))

        model.add(BatchNormalization(axis=chanDim))

        model.add(MaxPooling2D(pool_size=(3, 3)))

        model.add(Dropout(0.25))



        # (CONV => RELU) * 2 => POOL

        model.add(Conv2D(64, (3, 3), padding="same"))

        model.add(Activation("relu"))

        model.add(BatchNormalization(axis=chanDim))

        model.add(Conv2D(64, (3, 3), padding="same"))

        model.add(Activation("relu"))

        model.add(BatchNormalization(axis=chanDim))

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Dropout(0.25))



        # (CONV => RELU) * 2 => POOL

        model.add(Conv2D(128, (3, 3), padding="same"))

        model.add(Activation("relu"))

        model.add(BatchNormalization(axis=chanDim))

        model.add(Conv2D(128, (3, 3), padding="same"))

        model.add(Activation("relu"))

        model.add(BatchNormalization(axis=chanDim))

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Dropout(0.25))



        # first (and only) set of FC => RELU layers

        model.add(Flatten())

        model.add(Dense(1024))

        model.add(Activation("relu"))

        model.add(BatchNormalization())

        model.add(Dropout(0.5))



        # softmax classifier

        model.add(Dense(classes))

        model.add(Activation(finalAct))



        # return the constructed network architecture

        return model