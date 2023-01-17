# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np 

import pandas as pd 

import tensorflow as tf

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from sklearn.model_selection import train_test_split

from tensorflow.keras import layers, models

from tensorflow import keras



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from __future__ import absolute_import, division, print_function, unicode_literals



tf.keras.backend.clear_session() 
train_data = pd.read_csv("/kaggle/input/Kannada-MNIST/train.csv")



target = train_data['label']

train_vars = train_data.drop(['label'],axis=1)



X_train = train_vars/255

y = target



X_train = X_train.values.reshape(X_train.shape[0],28,28,1)
inputs = keras.Input(shape=(28, 28, 1), name='img')

x = layers.Conv2D(32, 3, activation='relu')(inputs)

x = layers.BatchNormalization()(x)

x = layers.Conv2D(64, 3, activation='relu')(x)

x = layers.BatchNormalization()(x)

block_1_output = layers.MaxPooling2D(3)(x)



x = layers.Conv2D(64, 3, activation='relu', padding='same')(block_1_output)

x = layers.BatchNormalization()(x)

x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)

x = layers.BatchNormalization()(x)

block_2_output = layers.add([x, block_1_output])



x = layers.Conv2D(64, 3, activation='relu', padding='same')(block_2_output)

x = layers.BatchNormalization()(x)

x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)

x = layers.BatchNormalization()(x)

block_3_output = layers.add([x, block_2_output])



x = layers.Conv2D(64, 3, activation='relu', padding='same')(block_3_output)

x = layers.BatchNormalization()(x)

x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)

block_4_output = layers.add([x, block_3_output])







x = layers.Conv2D(64, 3, activation='relu')(block_4_output)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dense(256, activation='relu')(x)

x = layers.Dropout(0.5)(x)

outputs = layers.Dense(10, activation='softmax')(x)



model = keras.Model(inputs, outputs, name='kannada_resnet')

model.summary()
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])



es = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=8)

lr_reduction = ReduceLROnPlateau(monitor='val_loss',patience=5,verbose=1,factor=0.2)
model.fit(X_train,np.array(y),

          epochs=50,validation_split=0.2,

         batch_size=128, shuffle=True,callbacks =[lr_reduction,es])
test_data = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')

ImageId = test_data['id']

NN_test = test_data.drop(['id'],axis=1)

NN_test = NN_test/255

NN_test = NN_test.values.reshape(NN_test.shape[0],28,28,1)



predictions = np.argmax(model.predict(NN_test),axis=1)
sub = pd.DataFrame({'id':ImageId, 'label':predictions})

sub.to_csv("submission.csv",index=False)