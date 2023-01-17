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
import h5py

import matplotlib.pyplot as plt

import numpy as np

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
#Datasets

test = h5py.File("../input/happy-house-dataset/test_happy.h5")

train = h5py.File("../input/happy-house-dataset/train_happy.h5")
train_x_orig = np.array(train['train_set_x'][:])

train_y_orig = np.array(train['train_set_y'][:])

test_x_orig = np.array(test['test_set_x'][:])

test_y_orig = np.array(test['test_set_y'][:])



print(train_x_orig.shape)

print(train_y_orig.shape)

print(test_x_orig.shape)

print(test_y_orig.shape)



#Flatten the image

train_x = train_x_orig/255

test_x = test_x_orig/255



#Reshape train_y_orig and test_y_orig

train_y = train_y_orig.reshape(1,train_y_orig.shape[0]).T

test_y = test_y_orig.reshape(1,test_y_orig.shape[0]).T



print("Number of training example " +str(train_x.shape[0]))

print("Number of test example " +str(test_x.shape[0]))

print("train_X shape "+str(train_x.shape))

print("train_Y shape "+str(train_y.shape))

print("test_X shape "+str(test_x.shape))

print("test_Y shape "+str(test_y.shape))
print(train_y)
plt.imshow(train_x[599])
def HappyModel(input_size):

    #Define input with shape input_size

    input_x = Input(input_size)

    

    #Zero Padding

    X = ZeroPadding2D((3,3),name='Zero_pad')(input_x)

    

    #Conv2D, BatchNormalization and Activation. BatchNormalizartion is a technique for training very deep NNs that standardize the inputsto a layer for each mini-batch.It increases the learning rate that increases the speed at which NNs train.Makes weights easier to initialize.

    X = Conv2D(32,(5,5),strides=(1,1),name='conv2d')(X)

    X = BatchNormalization(axis=3,name='BN')(X)

    X = Activation("relu")(X)

    

    #MaxPool

    X = MaxPooling2D((2,2),name='max-pool')(X)

    

    #Flatten X +FullyConnected

    X = Flatten()(X)

    X = Dense(1,activation='sigmoid',name='FC')(X)

    

    #create Model

    model = Model(inputs = input_x, outputs = X, name='model')

    

    return model

    
# First create a model by calling the function above.

model = HappyModel(train_x.shape[1:])

# Compile the model

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
# Fit the model into training data

history = model.fit(train_x,train_y,epochs=30,batch_size=30)
# Evaluate the model on Test Set

evals = model.evaluate(test_x,test_y)
print("Loss "+str(evals[0]))

print("accuracy "+str(evals[1]))
model.summary()