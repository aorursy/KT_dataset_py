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
! wget https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip
! unzip dogImages.zip
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator, load_img
train_data= ImageDataGenerator(
    rescale=.1/255)
train_generator=train_data.flow_from_directory(
    'dogImages/test',
    target_size=(256,256),
    batch_size=34,
    shuffle=True,
    class_mode='categorical')


val_data= ImageDataGenerator(
    rescale=.1/255)
val_generator=val_data.flow_from_directory(
    'dogImages/valid',
    target_size=(256,256),
    batch_size=34)
test_data= ImageDataGenerator(
    rescale=.1/255)
test_generator=test_data.flow_from_directory(
    'dogImages/test',
    target_size=(256,256),
    batch_size=34)
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation,Dropout,Dense,BatchNormalization,Flatten
model = Sequential()
model.add(Conv2D(16,(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Dropout(0.2))

model.add(Conv2D(32,(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Dropout(0.2))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Dropout(0.2))

model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(133,activation='relu'))
model.add(Dense(133,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='rmsprop', metrics=['accuracy'])
epochs=5
given= model.fit_generator(
    train_generator,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=34)
model.summary()
predict = model.predict(test_generator)