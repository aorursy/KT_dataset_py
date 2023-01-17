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
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale= 1./255)

td = train_datagen.flow_from_directory(directory='/kaggle/input/covid19-xray-dataset-train-test-sets/xray_dataset_covid19/train', target_size=(224, 224), batch_size=256, shuffle=True, class_mode= 'binary')

validation_datagen = ImageDataGenerator(rescale =1./255)

vd = validation_datagen.flow_from_directory(directory = '/kaggle/input/covid19-xray-dataset-train-test-sets/xray_dataset_covid19/test', target_size=(224,224), batch_size=256, shuffle=True, class_mode='binary')
import tensorflow as tf

from tensorflow.keras.models import Model

from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import Input

from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization

from tensorflow.keras.layers import Dropout

from tensorflow.keras.initializers import RandomNormal

from tensorflow.keras.regularizers import l2

from tensorflow.keras import optimizers

from tensorflow.keras.layers import Flatten
input = Input(shape=(224,224,3))

conv1 = Conv2D(filters=64,kernel_size=(3,3),strides=(1,1), padding='same', kernel_initializer= 'random_normal', bias_initializer= 'zeros')(input)

conv2 = Conv2D(filters=64,kernel_size=(3,3),strides=(1,1), padding='same', kernel_initializer= 'random_normal', bias_initializer= 'zeros')(conv1)

pool1 = MaxPooling2D(pool_size=(2,2), strides=2)(conv2)

conv3 = Conv2D(filters=128, kernel_size=(3,3),strides=(1,1), padding='same', kernel_initializer= 'random_normal', bias_initializer= 'zeros')(pool1)

conv4 = Conv2D(filters=128, kernel_size=(3,3),strides=(1,1), padding='same', kernel_initializer= 'random_normal', bias_initializer= 'zeros')(conv3)

pool2 = MaxPooling2D(pool_size=(2,2), strides=2)(conv4)

conv5 = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same', kernel_initializer= 'random_normal', bias_initializer= 'zeros')(pool2)

conv6 = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same', kernel_initializer= 'random_normal', bias_initializer= 'zeros')(conv5)

conv7 = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same', kernel_initializer= 'random_normal', bias_initializer= 'zeros')(conv6)

conv8 = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same', kernel_initializer= 'random_normal', bias_initializer= 'zeros')(conv7)

pool3 = MaxPooling2D(pool_size=(2,2), strides=2)(conv8)

conv9 = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same', kernel_initializer= 'random_normal', bias_initializer= 'zeros')(pool3)

conv10 = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same', kernel_initializer= 'random_normal', bias_initializer= 'zeros')(conv9)

conv11 = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same', kernel_initializer= 'random_normal', bias_initializer= 'zeros')(conv10)

conv12 = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same', kernel_initializer= 'random_normal', bias_initializer= 'zeros')(conv11)

pool4 = MaxPooling2D(pool_size= (2,2), strides= 2)(conv12)

conv13 = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same', kernel_initializer= 'random_normal', bias_initializer= 'zeros')(pool4)

conv14 = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same', kernel_initializer= 'random_normal', bias_initializer= 'zeros')(conv13)

conv15 = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same', kernel_initializer= 'random_normal', bias_initializer= 'zeros')(conv14)

conv16 = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same', kernel_initializer= 'random_normal', bias_initializer= 'zeros')(conv15)

pool5 = MaxPooling2D(pool_size= (2,2), strides= 2)(conv16)

flatten = Flatten()(pool5)

dense1 = Dense(units=4096,activation='relu', kernel_regularizer= l2(0.0005), kernel_initializer= 'random_normal', bias_initializer= 'zeros')(flatten)

dropout1 = Dropout(rate=0.5)(dense1)

dense2 = Dense(units=4096, activation='relu', kernel_regularizer= l2(0.0005), kernel_initializer= 'random_normal', bias_initializer= 'zeros')(dropout1)

dropout2 = Dropout(rate=0.5)(dense2)

output = Dense(units=1, activation='sigmoid', kernel_regularizer= l2(0.0005), kernel_initializer= 'random_normal', bias_initializer= 'zeros')(dropout2)

A = Model(inputs = input, outputs = output)

A.summary()

optimize = tf.keras.optimizers.Adam(learning_rate = 0.0001)

A.compile(optimizer= optimize, metrics=['accuracy'], loss='binary_crossentropy')

model = A.fit(x=td, validation_data = vd, epochs=30)