# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow import keras
from keras import models
from keras import layers
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("../input/digit-recognizer/train.csv")
test = pd.read_csv("../input/digit-recognizer/test.csv")
network=models.Sequential()
network.add(layers.Dense(512,activation='relu',input_shape=(28*28,)))
network.add(layers.Dense(10,activation='softmax'))
train_images=train.iloc[:,1:]
train_labels=train.iloc[:,0]

test
network.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
train_images=train_images.astype('float32')/255
#test=test.astype('float32')/255
from keras.utils import to_categorical
train_labels=to_categorical(train_labels)
network.fit(train_images,train_labels,epochs=5,batch_size=128)
print(test.shape)
preds = network.predict(test)
for i in range(len(test)):
    for j in range(10):
        if(preds[i][j]==1.0):
            print(i+1,',',j)


