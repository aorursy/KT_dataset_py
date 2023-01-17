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
import pandas as pd
import numpy as np
from matplotlib import pyplot  as plt 
%matplotlib inline
import cv2 as cv

import os
from keras.layers import Dense, Convolution2D, UpSampling2D, MaxPooling2D, ZeroPadding2D, Flatten, Dropout, Reshape
from keras.models import Sequential
from keras.utils import np_utils
dataset = pd.read_csv('../input/facial-expression-recognitionferchallenge/fer2013/fer2013/fer2013.csv')
dataset.head()
train_data = dataset[["emotion", "pixels"]][dataset["Usage"] == "Training"]
test_data = dataset[["emotion", "pixels"]][dataset["Usage"] == "PrivateTest"]       
# val_data = dataset[["emotion", "pixels"]][dataset["Usage"] == "PublicTest"]      
x_train = train_data['pixels'].apply(lambda x:np.fromstring(x, sep = ' ').reshape(48,48))
x_test = test_data['pixels'].apply(lambda x:np.fromstring(x, sep = ' ').reshape(48,48))
# x_val = val_data['pixels'].apply(lambda x:np.fromstring(x, sep = ' ').reshape(48,48))
x_train.shape
from math import ceil

X_train = np.zeros((x_train.shape[0],48,48,3))
for a,x in enumerate(x_train):
    X_train[a] = np.expand_dims(x,axis=2)
X_train = np.array(X_train)
print(X_train.shape)

X_test = np.zeros((x_test.shape[0],48,48,3))
for a,x in enumerate(x_test):
    X_test[a] = np.expand_dims(x,axis=2)
X_test = np.array(X_test)
print(X_test.shape)
y_train = train_data['emotion']
y_test = test_data['emotion']
# y_val = val_data['emotion']
label_dict = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Sad', 5:'Surprise', 6:'Neutral'}

plt.imshow(X_valdata[10][0],cmap='bone')
plt.title(label_dict[Y_valdata[10].argmax()])
plt.show()
from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# y_val = to_categorical(y_val)
val_split = 0.8
split = ceil(x_train.shape[0]*val_split)
X_traindata = X_train[0:split]
X_valdata = X_train[split:]
Y_traindata = y_train[0:split]
Y_valdata = y_train[split:]
print(X_traindata.shape,X_valdata.shape)
print(Y_traindata.shape,Y_valdata.shape)
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
        featurewise_center=False,  
        samplewise_center=False,  
        featurewise_std_normalization=False,  
        samplewise_std_normalization=False,  
        zca_whitening=False,  
        rotation_range=10,  
        zoom_range = 0.0,  
        width_shift_range=0.1,  
        height_shift_range=0.1,  
        horizontal_flip=False, 
        vertical_flip=False)  
from keras.callbacks import ReduceLROnPlateau
lr_reduce = ReduceLROnPlateau(monitor='val_acc', factor=0.1, epsilon=0.0001, patience=1, verbose=1)
!pip install keras
from keras.layers import Dense , Activation
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD , Adam
from keras.layers import Input, Conv2D , BatchNormalization
from keras.layers import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.common.set_image_dim_ordering('th')
from keras.applications.mobilenet_v2 import MobileNetV2
mnetv2 = MobileNetV2(include_top=False,input_shape=(48,48,3), weights='imagenet')
mnetv2.summary()
topLayerModel = Sequential()

topLayerModel.add(mnetv2)
topLayerModel.add(Flatten())
topLayerModel.add(Dense(256, activation='relu'))
topLayerModel.add(Dense(256, activation='relu'))
topLayerModel.add(Dropout(0.5))
topLayerModel.add(Dense(128, activation='relu'))
topLayerModel.add(Dense(7, activation='softmax'))
print(topLayerModel.summary())
topLayerModel.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])
history = topLayerModel.fit_generator(datagen.flow(X_traindata, Y_traindata, batch_size=128),
                    steps_per_epoch=ceil(X_train.shape[0] / 128) ,
                    callbacks=[lr_reduce,],
                    validation_data=(X_valdata, Y_valdata),
                    epochs = 10, verbose = 1)
import tensorflow as tf
print(tf.__version__)
X_test = np.zeros((x_test.shape[0],48,48,3))
for a,x in enumerate(x_test):
    X_test[a] = np.expand_dims(x,axis=2)
X_test = np.array(X_test)
print(X_test.shape)
#Select id for manual checking
idx = 511
print("Output: ",label_dict[topLayerModel.predict(np.array([X_test[idx]])).argmax()])
plt.imshow(x_test.iloc[idx],cmap='bone')
plt.title(label_dict[y_test[idx].argmax()])
plt.show()
