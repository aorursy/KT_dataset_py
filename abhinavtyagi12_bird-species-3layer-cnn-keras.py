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
import matplotlib.pyplot as plt

from matplotlib.pyplot import imshow

import numpy as np

import pandas as pd

import PIL

import tensorflow as tf

import keras

from keras import backend as K

from keras.layers import Conv2D, MaxPool2D, BatchNormalization

from keras.layers import Activation,Dropout, Flatten, Dense, Input

from keras.models import load_model, Model, Sequential

from keras.preprocessing.image import ImageDataGenerator
train_data_dir = '/kaggle/input/100-bird-species/180/train/'

test_data_dir = '/kaggle/input/100-bird-species/180/test/'

valid_data_dir='/kaggle/input/100-bird-species/180/valid/'

epochs = 25

batch_size = 32
train_datagen=ImageDataGenerator()
valid_datagen=ImageDataGenerator()
valid_data=valid_datagen.flow_from_directory(directory=valid_data_dir)
train_data=train_datagen.flow_from_directory(directory=train_data_dir)
train_data.image_shape
test_datagen=ImageDataGenerator()
test_data=test_datagen.flow_from_directory(test_data_dir)
len(train_data),len(test_data)
(train_data[1][0]).shape[1:]
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
tf.debugging.set_log_device_placement(True)
# #testing cnn

# def model(input_shape):

#     X_input=Input(input_shape)

#     X=Conv2D(32,(5,5),strides=(1,1), name='conv1')(X_input)

#     X=BatchNormalization(axis=3, name='bn1')(X)

#     X=Activation('relu')(X)

    

#     X=MaxPool2D((2,2),name='maxpool1')(X)

    

#     X=Flatten()(X)

#     X=Dense(180,activation='softmax', name='fc1')(X)

#     model=Model(inputs=X_input,outputs=X,name='bird_classifier')

    

#     return model
# bird_model=model(train_data[1][0].shape[1:])
# bird_model.summary()
# bird_model.compile(loss='mean_squared_error',optimizer='sgd',metrics=['accuracy'])
# bird_model.fit(train_data,epochs=25)
def model2(input_shape):

    X_input=Input(input_shape)

    X=Conv2D(32,(7,7),strides=(1,1), name='conv1')(X_input)

    X=BatchNormalization(axis=3, name='bn1')(X)

    X=Activation('relu')(X)

    

    X=MaxPool2D((2,2),name='maxpool1')(X)

    

    X=Conv2D(64,(7,7),strides=(1,1), name='conv2')(X)

    X=BatchNormalization(axis=3, name='bn2')(X)

    X=Activation('relu')(X)

    

    X=MaxPool2D((2,2),name='maxpool2')(X)

    

    X=Conv2D(128,(5,5),strides=(1,1), name='conv3')(X)

    X=BatchNormalization(axis=3, name='bn3')(X)

    X=Activation('relu')(X)

    

    X=MaxPool2D((2,2),name='maxpool3')(X)

    

#     X=Conv2D(512,(5,5),strides=(1,1), name='conv4')(X)

#     X=BatchNormalization(axis=3, name='bn4')(X)

#     X=Activation('relu')(X)

    

#     X=MaxPool2D((2,2),name='maxpool4')(X)

    

    X=Flatten()(X)

    X=Dense(180,activation='softmax', name='fc1')(X)

    model2=Model(inputs=X_input,outputs=X,name='bird_classifier')

    

    return model2
bird_model_2=model2(train_data[1][0].shape[1:])
bird_model_2.summary()
sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

adam=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
bird_model_2.compile(loss='binary_crossentropy',optimizer=sgd,metrics=['accuracy'])
bird_model_2.fit(train_data,epochs=25)
#run for more epochs to get better accuracy
preds_valid=bird_model_2.evaluate(valid_data)

print()

print("loss= "+str(preds_valid[0]))

print("test accuracy= "+str(round(preds_valid[1]*100,2))+"%")
preds_test=bird_model_2.evaluate(test_data)

print()

print("loss= "+str(preds_test[0]))

print("test accuracy= "+str(round(preds_test[1]*100,2))+"%")