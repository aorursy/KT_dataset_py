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
import keras
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
from keras.models import Sequential
from keras.losses import binary_crossentropy
import pickle
import tensorflow as tf
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)

# instantiate a distribution strategy
tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
with tpu_strategy.scope():
    model = Sequential()
    model.add(Conv2D(32,kernel_size=(3,3),input_shape=(224,224,3),activation='relu'))
    model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(128,kernel_size=(3,3),activation='relu'))
    model.add(Conv2D(128,kernel_size=(3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(rate=0.25))
    model.add(Conv2D(256,kernel_size=(3,3),activation='relu'))
    model.add(Dropout(rate=0.25))
    model.add(Conv2D(256,kernel_size=(3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(rate=0.25))
    model.add(Flatten())
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy'])
model.summary()
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        '../input/covid19-xray-dataset-train-test-sets/xray_dataset_covid19/train',
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary')
validation_generator = test_datagen.flow_from_directory(
        '../input/covid19-xray-dataset-train-test-sets/xray_dataset_covid19/test',
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary')
model.fit(
        train_generator,
        steps_per_epoch=148,
        epochs=5,
        validation_data=validation_generator
        )
model.save('covid19_97.5%.m5')



