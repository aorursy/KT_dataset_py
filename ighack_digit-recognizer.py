# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow.keras import models,layers

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
train_size = int(len(data)*0.8)
train_dataset = data.iloc[:train_size,:]
test_dataset = data.iloc[train_size:]
train_dataset.isna().any()
train_dataset.isnull().any()
train_y = train_dataset.pop('label')
train_y = train_y.values
train_x = train_dataset.values
train_x
train_y
test_y = test_dataset.pop('label')
test_y = test_y.values
test_x = test_dataset.values
train_x.shape
train_y.shape
model = models.Sequential()
model.add(layers.Dense(800,activation='relu',input_shape=(train_x.shape[1],)))
model.add(layers.Dense(10,activation='softmax'))
model.summary()
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=10,mode='max')
model.fit(train_x,train_y,epochs=100,batch_size=20,validation_split=0.2,callbacks=[early_stopping])