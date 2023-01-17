
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/chest_xray/"))

# Any results you write to the current directory are saved as output.
train_path = "../input/chest_xray/chest_xray/train"
test_path = "../input/chest_xray/chest_xray/test"
val_path = "../input/chest_xray/chest_xray/val"
import keras
from keras.preprocessing.image import ImageDataGenerator
train_batch = ImageDataGenerator().flow_from_directory(train_path , target_size = (224,224) , classes =['NORMAL','PNEUMONIA'] , batch_size=50)
test_batch = ImageDataGenerator().flow_from_directory(test_path , target_size = (224,224) , classes =['NORMAL','PNEUMONIA'] , batch_size=50)
val_batch = ImageDataGenerator().flow_from_directory(val_path , target_size = (224,224) , classes =['NORMAL','PNEUMONIA'] , batch_size=4)
X_train , y_train = next(train_batch)
X_test , y_test = next(test_batch)
X_val , y_val = next(val_batch)
print(X_train.shape," ",y_train.shape)
print(X_test.shape," ",y_test.shape)
print(X_val.shape," ",y_val.shape)
X_train = X_train/255
X_test = X_train/255
X_test = X_train/255
X_train.shape
from keras.models import Sequential
from keras.layers import Dense , Conv2D , BatchNormalization , MaxPooling2D ,Dropout,Flatten
model = Sequential()
model.add(Conv2D(filters=32 , kernel_size=(5,5), strides=(1,1),padding='same' , activation='relu' , input_shape = (224,224,3)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
model.add(Conv2D(filters=64 , kernel_size=(5,5), strides=(1,1),padding='same' , activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
model.add(Flatten())
model.add(Dense(1, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2 , activation='softmax'))
model.summary()
from keras.optimizers import SGD
optimizer = SGD(lr = 0.01 , momentum=0.09)
model.compile(optimizer=optimizer , loss = 'categorical_crossentropy' , metrics=['accuracy'])
traing = model.fit(X_train , y_train , epochs=10 , verbose=2 , validation_data=(X_test , y_test))

