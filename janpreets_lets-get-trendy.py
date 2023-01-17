# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import keras



from keras.models import Sequential

from keras.layers.core import Flatten, Dense, Dropout

from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D

from keras.optimizers import SGD

from keras.layers import Dense, Dropout, Flatten, Input

from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D

from keras.models import Model

from keras.utils.layer_utils import print_summary

import cv2, numpy as np

from keras import backend as K

K.set_image_dim_ordering('tf')
train = pd.read_csv("../input/fashionmnist/fashion-mnist_train.csv")

test = pd.read_csv("../input/fashionmnist/fashion-mnist_test.csv")
label = train.label
train = train.drop("label",axis = 1)
train.head()
import cv2

pix = np.array(train.iloc[1].values)

pixels = np.array(pix, dtype='uint8')

pixels = pixels.reshape((28, 28))

plt.imshow(pixels, cmap='gray')

plt.show()

import keras.utils.np_utils as kutils
trainX = train.values.reshape(train.shape[0], 28, 28, 1)

trainX = trainX.astype(float)

trainX /= 255.0



trainY = kutils.to_categorical(label)

nb_classes = trainY.shape[1]
label = label.values
label = label.reshape((60000,1))
label.shape
model = Sequential()

model.add(Convolution2D(32, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=(28,28,1)))

model.add(Convolution2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(ZeroPadding2D((1,1)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))



model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer=keras.optimizers.Adadelta(),

              metrics=['accuracy'])
model.fit(trainX, trainY, batch_size=4, nb_epoch=2, verbose=1)
vis_input = Input(shape=(28,28,1), name="vis_input")





x = Convolution2D(64, (3, 3), activation='relu')     (vis_input)

x = Convolution2D(64, (3, 3), activation='relu')     (x)

x = MaxPooling2D((2,2))             (x)



x = ZeroPadding2D((1,1))                           (x)

x = Convolution2D(128, (3, 3), activation='relu')    (x)

x = ZeroPadding2D((1,1))                           (x)

x = Convolution2D(128, (3, 3), activation='relu')    (x)

x = MaxPooling2D((2,2))             (x)



x = ZeroPadding2D((1,1))                           (x)

x = Convolution2D(256, (3, 3), activation='relu')    (x)

x = ZeroPadding2D((1,1))                           (x)

x = Convolution2D(256, (3, 3), activation='relu')    (x)

x = ZeroPadding2D((1,1))                           (x)

x = Convolution2D(256, (3, 3), activation='relu')    (x)

x = MaxPooling2D((2,2))             (x)



x = ZeroPadding2D((1,1))                           (x)

x = Convolution2D(512, (3, 3), activation='relu')    (x)

x = ZeroPadding2D((1,1))                           (x)

x = Convolution2D(512, (3, 3), activation='relu')    (x)

x = ZeroPadding2D((1,1))                           (x)

x = Convolution2D(512, (3, 3), activation='relu')    (x)

x = MaxPooling2D((2,2))             (x)





x = Flatten()                                      (x)

x = Dense(4096, activation='relu')                 (x)

x = Dropout(0.5)                                   (x)

x = Dense(4096, activation='relu')                 (x)

x = Dropout(0.5)                                   (x)

x = Dense(10, activation='relu')                 (x)



model = Model(input= vis_input, output=x)

model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer=keras.optimizers.Adadelta(),

              metrics=['accuracy'])
print(model.summary())
model.fit(trainX, trainY, batch_size=4, nb_epoch=2, verbose=1)