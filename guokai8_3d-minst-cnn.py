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
from keras.layers import Conv3D, MaxPool3D, Flatten, Dense

from keras.layers import Dropout, Input, BatchNormalization, Activation

from keras.optimizers import adam, adadelta

from keras.models import Model

from keras.utils.np_utils import to_categorical

from sklearn.metrics import confusion_matrix, accuracy_score

from keras.losses import categorical_crossentropy

from matplotlib.pyplot import cm

import matplotlib.pyplot as plt

import numpy as np

import keras

import h5py

init_notebook_mode(connected=True)

%matplotlib inline
with h5py.File('../input/3d-mnist/full_dataset_vectors.h5', 'r') as dataset:

    x_train = dataset["X_train"][:]

    x_test = dataset["X_test"][:]

    y_train = dataset["y_train"][:]

    y_test = dataset["y_test"][:]
print ("x_train shape: ", x_train.shape)

print ("y_train shape: ", y_train.shape)



print ("x_test shape:  ", x_test.shape)

print ("y_test shape:  ", y_test.shape)

###trasform to 3d

xtrain = np.ndarray((x_train.shape[0], 4096, 3))

xtest = np.ndarray((x_test.shape[0], 4096, 3))
def add_rgb_dimention(array):

    scaler_map = cm.ScalarMappable(cmap="Oranges")

    array = scaler_map.to_rgba(array)[:, : -1]

    return array

for i in range(x_train.shape[0]):

    xtrain[i] = add_rgb_dimention(x_train[i])

for i in range(x_test.shape[0]):

    xtest[i] = add_rgb_dimention(x_test[i])
add_rgb_dimention(x_train[1]).shape

xtrain = xtrain.reshape(x_train.shape[0], 16, 16, 16, 3)

xtest = xtest.reshape(x_test.shape[0], 16, 16, 16, 3)



## convert target variable into one-hot

y_train = keras.utils.to_categorical(y_train, 10)

y_test = keras.utils.to_categorical(y_test, 10)
y_train.shape
## input layer

input_layer = Input((16, 16, 16, 3))



## convolutional layers

conv_layer1 = Conv3D(filters=16, kernel_size=(3, 3, 3),strides=1,padding ='same',activation='relu')(input_layer)

conv_layer1 = BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer="uniform")(conv_layer1)

conv_layer1 = Dropout(0.1)(conv_layer1)

conv_layer2 = Conv3D(filters=32, kernel_size=(3, 3, 3),strides=1,padding ='same',activation='relu')(conv_layer1)

conv_layer2 = BatchNormalization(momentum=0.15, epsilon=1e-5, gamma_initializer="uniform")(conv_layer2)

conv_layer2 = Dropout(0.2)(conv_layer2)

## add max pooling to obtain the most imformatic features

pooling_layer1 = MaxPool3D(pool_size=2,strides=2,padding='same')(conv_layer2)



conv_layer3 = Conv3D(filters=64, kernel_size=(5, 5, 5),strides=1,padding ='same',activation='relu')(pooling_layer1)

conv_layer3 = BatchNormalization(momentum=0.15, epsilon=1e-5, gamma_initializer="uniform")(conv_layer3)

conv_layer3 = Dropout(0.2)(conv_layer3)

conv_layer4 = Conv3D(filters=32, kernel_size=(3, 3, 3),strides=1,padding ='same',activation='relu')(conv_layer3)

conv_layer4 = BatchNormalization(momentum=0.15, epsilon=1e-5, gamma_initializer="uniform")(conv_layer4)

conv_layer4 = Dropout(0.2)(conv_layer4)

##pooling_layer3 = MaxPool3D(pool_size=(2, 2, 2))(conv_layer4)



## perform batch normalization on the convolution outputs before feeding it to MLP architecture

flatten_layer = Flatten()(conv_layer4)



## create an MLP architecture with dense layers : 4096 -> 512 -> 10

## add dropouts to avoid overfitting / perform regularization

dense_layer1 = Dense(units=4096, activation='relu')(flatten_layer)

dense_layer1 = Dropout(0.05)(dense_layer1)

dense_layer2 = Dense(units=1024, activation='relu')(dense_layer1)

dense_layer2 = Dropout(0.05)(dense_layer2)

dense_layer3 = Dense(units=256, activation='relu')(dense_layer2)

dense_layer3 = Dropout(0.05)(dense_layer3)

output_layer = Dense(units=10, activation='softmax')(dense_layer3)



## define the model with input layer and output layer

model = Model(inputs=input_layer, outputs=output_layer,name='3DCNN')
adam = adam(learning_rate=0.002, beta_1=0.9, beta_2=0.999, amsgrad=True)

model.compile(loss=categorical_crossentropy, optimizer=adam, metrics=['acc'])



model.summary()
model.fit(x=xtrain, y=y_train, batch_size=128, epochs=50, validation_split=0.2)

pred = model.predict(xtest)

pred = np.argmax(pred, axis=1)

pred