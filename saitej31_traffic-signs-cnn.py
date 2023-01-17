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
data = pd.read_pickle('/kaggle/input/traffic-signs-preprocessed/data8.pickle')
data
print(len(data['x_train']),len(data['y_train']),len(data['x_validation']),len(data['y_validation']),len(data['x_test']),len(data['y_test']))

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
import tensorflow as tf

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

%matplotlib inline
# Normalize image vectors
X_train = data['x_train']/255.
X_test = data['x_test']/255.
X_validation = data['x_validation']/.255

# Reshape
Y_train = data['y_train'].T
Y_test = data['y_test'].T
Y_validation = data['y_validation'].T

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of Validation examples = " + str(X_validation.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))

print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))

print ("X_validation shape: " + str(X_validation.shape))
print ("Y_validation shape: " + str(Y_validation.shape))

print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))

x_validation = X_validation.transpose(0, 2, 3, 1)
x_train = X_train.transpose(0, 2, 3, 1)
x_test = X_test.transpose(0, 2, 3, 1)
print(x_train.shape,x_validation.shape,x_test.shape)
def model(input_shape):

    X_input = Input(input_shape)

    # Zero-Padding: pads the border of X_input with zeroes
    X = ZeroPadding2D((3, 3))(X_input)

    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(16, (3, 3), strides = (1, 1), name = 'conv0')(X)
    print(X)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2), name='max_pool0')(X)
    
    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(32, (3, 3), strides = (1, 1), name = 'conv1')(X)
    X = BatchNormalization(axis = 3, name = 'bn1')(X)
    X = Activation('relu')(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2), name='max_pool1')(X)
    
    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(64, (3, 3), strides = (1, 1), name = 'conv2')(X)
    X = BatchNormalization(axis = 3, name = 'bn2')(X)
    X = Activation('relu')(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2), name='max_pool2')(X)
    

    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = X)

    
    
    return model
Model = model(x_train.shape[1:])
Model.compile(optimizer='Adadelta', loss = 'binary_crossentropy', metrics = ["accuracy"])
Model.fit(x = x_train , y = Y_train, epochs = 50, batch_size = 1000)
preds = Model.evaluate(x = x_validation, y = Y_validation )
print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
preds = Model.evaluate(x = x_test, y = Y_test)
print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
