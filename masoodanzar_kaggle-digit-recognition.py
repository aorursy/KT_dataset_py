# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import matplotlib.pyplot as plt

from keras import models

from keras import layers

from keras.utils import to_categorical

from keras.datasets import mnist



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data_train=pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

X=data_train.iloc[:,1:]

y=data_train.iloc[:,0]

print('X=',X.shape, '     y_train =',len(y))

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

data_test=pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

data_train.head()

K_test=data_test.iloc[:, 0:]

print('K_test Size  =',K_test.shape)
#******************Model No 1*****************

X1_train = X_train

X1_test  = X_test

X1_train = X1_train.astype('float32') / 255

X1_test = X1_test.astype('float32') / 255

y1_train = to_categorical(y_train)

y1_test = to_categorical(y_test)



print('X1_train Size =',X1_train.shape, '     y1_train =',len(y1_train))

print('X1_test Size  =',X1_test.shape)



model1 = models.Sequential()

model1.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))

model1.add(layers.Dense(10, activation='softmax'))



model1.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

model1.fit(X1_train, y1_train, epochs=3) #, batch_size=28

test_loss1, test_acc1 = model1.evaluate(X1_test, y1_test)



model1.summary()

print('Model 1 Aest_Acc:', test_acc1)
X2_train = X_train

X2_test  = X_test

X2_train = X2_train.astype('float32') / 255

X2_test  = X2_test.astype('float32') / 255

y2_train = to_categorical(y_train)

y2_test  = to_categorical(y_test)



print('X1_train Size =',X2_train.shape, '     y1_train =',len(y2_train))

print('X2_test Size  =',X2_test.shape,  '     y1_train =',len(y2_test))
#******************Model No 2*****************

#X2_train = X2_train.to_numpy()

#X2_test = X2_test.to_numpy()

model2=tf.keras.models.Sequential()

model2.add(tf.keras.layers.Flatten())

model2.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

model2.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

model2.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))



model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model2.fit(X2_train,y2_train, epochs=5)

test_loss2, test_acc2 = model2.evaluate(X2_test, y2_test)

model2.summary()

print('Model 2 Aest_Acc:', test_acc2)
# Model 3

X3_train = X_train.to_numpy()

X3_test = X_test.to_numpy()



X3_train = X3_train.reshape((-1, 28, 28, 1))

X3_test = X3_test.reshape((-1, 28, 28, 1))

X3_train = X3_train.astype('float32') / 255

X3_test = X3_test.astype('float32') / 255

y3_train = to_categorical(y_train)

y3_test = to_categorical(y_test)



model3 = models.Sequential()

model3.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))

model3.add(layers.MaxPooling2D((2, 2)))

model3.add(layers.Conv2D(64, (3, 3), activation='relu'))

model3.add(layers.MaxPooling2D((2, 2)))

model3.add(layers.Conv2D(64, (3, 3), activation='relu'))

model3.add(layers.Flatten())

model3.add(layers.Dense(64, activation='relu'))

model3.add(layers.Dense(10, activation='softmax'))



model3.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

model3.fit(X3_train, y3_train,epochs=5) #, , batch_size=64

test_loss3, test_acc3 = model3.evaluate(X3_test, y3_test)

# Test Result

K3_test = K_test.to_numpy()

K3_test = K3_test.reshape((-1, 28, 28, 1))

K3_test = K3_test.astype('float32') / 255

y_predict=model3.predict(K3_test)



y_p=y_predict.argmax(axis=1)

yp = pd.DataFrame(y_p, columns=['Label'])

yp.index.name='ImageId'

yp.head()

#yp.to_csv('resultr.csv')