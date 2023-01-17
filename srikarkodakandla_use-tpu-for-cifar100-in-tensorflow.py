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
from tensorflow.keras import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar100
(a,b),(c,d)=cifar100.load_data()
plt.imshow(a[0])
plt.imshow(c[0])
b_cat=to_categorical(b)
d_cat=to_categorical(d)
#we have to convert to tensors before giving the data to the TPU\
#preferably we use float32
a=tf.cast(a,tf.float32)
b_cat=tf.cast(b_cat,tf.float32)
c=tf.cast(c,tf.float32)
d_cat=tf.cast(d_cat,tf.float32)
#Run this to initlize the TPU
# detect and init the TPU
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)

# instantiate a distribution strategy
tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

a.shape
"""construct neurons using with  tpu_strategy.scope(): it means we are constructing the neurons 
to be compactable for the TPU Architecture"""
with tpu_strategy.scope():
    model = Sequential()
    model.add(Convolution2D(512,3,input_shape=(32,32,3)))
    model.add(BatchNormalization())
    model.add(Convolution2D(512, 3,activation='relu'))
    model.add(BatchNormalization())
    model.add(Convolution2D(512, 3,  activation='relu'))
    model.add(BatchNormalization())
    model.add(Convolution2D(512, 3,  activation='relu'))
    model.add(BatchNormalization())
    model.add(Convolution2D(512, 3,  activation='relu'))
    model.add(BatchNormalization())
    model.add(Convolution2D(512, 3,  activation='relu'))
    model.add(BatchNormalization())
    model.add(Convolution2D(512, 3,  activation='relu'))
    model.add(Convolution2D(512, 3,  activation='relu'))
    model.add(BatchNormalization())
    model.add(Convolution2D(512, 3,  activation='relu'))
    model.add(BatchNormalization())
    model.add(Convolution2D(512, 3,  activation='relu'))
    model.add(BatchNormalization())
    model.add(Convolution2D(512, 3,  activation='relu'))
    model.add(BatchNormalization())
    model.add(Convolution2D(512, 3,  activation='relu'))
    model.add(BatchNormalization())
    model.add(Convolution2D(512, 3,  activation='relu'))
    model.add(Convolution2D(512, 3,  activation='relu'))
    model.add(BatchNormalization())
    model.add(Convolution2D(512, 3,  activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='softmax'))
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
model.fit(a,b_cat,validation_data=(c,d_cat),epochs=50,batch_size = 128)
tpu_strategy.num_replicas_in_sync
16*8
model.summary()
import pandas as pd
pd.DataFrame(model.history.history)
pd.DataFrame(model.history.history).plot()
