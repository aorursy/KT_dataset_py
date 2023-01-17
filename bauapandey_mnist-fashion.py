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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import keras
train_path = '/kaggle/input/fashionmnist/fashion-mnist_train.csv'
test_path = '/kaggle/input/fashionmnist/fashion-mnist_test.csv'
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
Y_train = train_df['label']
X_train = train_df.drop(labels='label',axis=1)
X_train = X_train.values.reshape(-1,28,28,1)

Y_test = test_df['label']
X_test = test_df.drop(labels='label',axis=1)
X_test = X_test.values.reshape(-1,28,28,1)
plt.imshow(X_train[59][:,:,0])
# X_train.shape
# Y_train.shape
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam,RMSprop,Adamax,SGD
from keras.models import Sequential
from keras.layers import Activation,Conv2D,MaxPool2D,Dense,Dropout,Flatten
model = Sequential()

model.add(Conv2D(filters=32,kernel_size=3,strides=1,padding='Same',activation='relu',input_shape=(28,28,1)))
model.add(Conv2D(filters=32,kernel_size=3,strides=1,padding='Same',activation='relu'))
model.add(MaxPool2D(pool_size=2,strides=2))
model.add(Dropout(0.30))


model.add(Conv2D(filters=64,kernel_size=3,strides=1,padding='Same',activation='relu'))
# model.add(Conv2D(filters=64,kernel_size=5,strides=1,padding='Same',activation='relu'))
model.add(MaxPool2D(pool_size=2,strides=2))


model.add(Conv2D(filters=128,kernel_size=3,strides=1,padding='Same',activation='relu'))
# model.add(Conv2D(filters=128,kernel_size=5,strides=1,padding='Same',activation='relu'))
model.add(MaxPool2D(pool_size=2,strides=2))
model.add(Dropout(0.30))


model.add(Flatten())
model.add(Dense(units=100,activation='relu'))
model.add(Dropout(0.50))

model.add(Dense(units=64,activation='relu'))
model.add(Dropout(0.50))


model.add(Dense(units=10,activation='softmax'))
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)


# model.compile(optimizer='adagrad',loss='categorical_crossentropy',metrics=['accuracy'])

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
Y_train = to_categorical(Y_train,num_classes=10)
Y_test = to_categorical(Y_test,num_classes=10)
from sklearn.model_selection import train_test_split

train_X,valid_X,train_Y,valid_Y = train_test_split(X_train,Y_train,test_size=0.4,random_state=10)
Y_train.shape
epochs=20
batch_size=200

history = model.fit(train_X,train_Y, batch_size = batch_size, epochs = epochs, 
         validation_data = (valid_X,valid_Y), verbose = 2)
results = model.evaluate(X_test,Y_test,verbose=0)
print("score:",results[0])
print("accuracy:",results[1])