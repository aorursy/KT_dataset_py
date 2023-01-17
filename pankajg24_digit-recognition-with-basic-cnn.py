# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf

from keras import layers

from keras.layers import Dense, Input, Activation, Conv2D, ZeroPadding2D, Flatten, BatchNormalization

from keras.layers import MaxPooling2D

from keras.models import Model

from keras.preprocessing import image

from keras.applications.imagenet_utils import preprocess_input

import keras.backend as K

from keras.utils import to_categorical



K.set_image_data_format('channels_last')

import matplotlib.pyplot as plt

from matplotlib.pyplot import imshow
train_data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test_data = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
train_data.shape
test_data.shape
train_data_np  = np.array(train_data)
y_train = train_data_np[:,0]

y_train.shape
y_train_1 = y_train.reshape(y_train.shape[0],1)

y_train_1.shape
x_train = train_data_np[:, 1:]

x_train.shape


x_train = x_train.reshape(x_train.shape[0],28,28)

x_train.shape
x_test = np.array(test_data)

x_test.shape
x_test = x_test.reshape(x_test.shape[0], 28,28,1)
x_test.shape
#display few random images

imshow(x_train[10])
#Create the model

def digit_model(x_input):

    

    x_input = Input(x_input)

    

    X = ZeroPadding2D((1,1))(x_input)

    

    X = Conv2D(32,(3,3), strides = (1,1), name = 'conv0')(X)

    X = BatchNormalization(axis = 3, name = 'bn0')(X)

    X = Activation('relu')(X)

    

    X = MaxPooling2D(pool_size = (2,2), name = 'mp0')(X)

    

#    X = Conv2D(8,(3,3), strides = (1,1), name = 'conv1')(X)

#    X = Activation('relu')(X)

    

#    X = MaxPooling2D(pool_size = (2,2), name = 'mp1')(X)

    X = Flatten()(X)

    X = Dense(10, activation = 'sigmoid', name = 'fc')(X)

    

    model = Model(inputs = x_input, outputs = X, name = 'digit_model')

    

    return model

    
X_train_1 = x_train.reshape(x_train.shape[0], 28,28,1) / 255

X_test_1 = x_test / 255
#Covert Y to 10 columns

y_train_oh = to_categorical(y_train_1, num_classes=10)

#help(to_categorical)

y_train_oh.shape
y_train_oh[30]
dmodel = digit_model(X_train_1.shape[1:])
dmodel.compile(optimizer= 'adam', loss= 'binary_crossentropy', metrics= ['accuracy'])

dmodel.fit(X_train_1, y_train_oh, batch_size= 256, epochs= 50)
help(dmodel.predict)
Y_pred = dmodel.predict(X_test_1, batch_size= 256)
Y_pred.shape
Y_pred[10]
x_test_f = np.array(test_data)

x_test_f.shape

x_test_f = x_test_f.reshape(x_test_f.shape[0], 28,28)
imshow(x_test_f[10])
help(np.argmax)
y_pred_final = np.argmax(Y_pred, axis= 1)
y_pred_final.shape
y_pred_final[0:10]