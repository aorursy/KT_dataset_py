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
train_data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test_data = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
train_data.shape
train_data['label'].value_counts()
x = train_data.drop('label',axis=1)
y = train_data['label']
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)
x_train = x_train.astype('float32')

x_test = x_test.astype('float32')

x_train/= 255

x_test/= 255
import keras

y_train = keras.utils.to_categorical(y_train,num_classes=10)

y_test = keras.utils.to_categorical(y_test,num_classes=10)
x_train.shape[0]
x_train = np.asarray(x_train)

for a in x_train:

    a = np.arange(784)

    a.reshape(28, 28)
x_train = np.reshape(x_train,(33600,28,28,1))
x_test.shape
x_test = np.asarray(x_test)

for a in x_test:

    a = np.arange(784)

    a.reshape(28, 28)

x_test = np.reshape(x_test,(8400,28,28,1))
x_train.shape[1:]
# from keras.layers import Conv2D,Dense,Flatten,Dropout,MaxPooling2D

# from keras import Sequential

from keras.models import Sequential

from keras.layers.normalization import BatchNormalization

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from keras.layers.core import Activation, Flatten, Dropout, Dense
inputShape = (28, 28, 1)



model = Sequential()



model.add(Conv2D(32, (3, 3), padding="same",input_shape=inputShape))

model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Dropout(0.25))



model.add(Conv2D(64, (3, 3), padding="same"))

model.add(Activation("relu"))



model.add(Conv2D(64, (3, 3), padding="same"))

model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(128, (3, 3), padding="same"))

model.add(Activation("relu"))



model.add(Flatten())



model.add(Dense(1024))

model.add(Activation("relu"))

model.add(Dropout(0.5))

model.add(Dense(10))

model.add(Activation("softmax"))



# model.add(Conv2D(32, (3, 3), activation='relu',input_shape=(28,28,1)))

# model.add(MaxPooling2D(pool_size=(2,2)))

# model.add(Dropout(0.2))



# model.add(Conv2D(64,(3,3),padding='same',activation='relu'))

# model.add(MaxPooling2D(pool_size=(2,2)))

# model.add(Dropout(0.2))



# model.add(Conv2D(64,3,3,input_shape = x_train.shape[1:],activation='relu'))

# model.add(MaxPooling2D(pool_size=(2,2)))

# model.add(Dropout(0.2))



# model.add(Flatten())



# model.add(Dense(128,activation='relu'))

# model.add(Dropout(0.2))

# model.add(Dense(10,activation='softmax'))



model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(x_train,y_train,epochs=30,batch_size=32,validation_data=(x_test,y_test))



scores = model.evaluate(x_test,y_test)

print('loss:',scores[0])

print('acc:',scores[1])
test_data = test_data.astype('float32')

test_data/= 255
test_data.shape
test = np.asarray(test_data)

for a in test:

    a = np.arange(784)

    a.reshape(28, 28)

test = np.reshape(test,(28000,28,28,1))
pred = model.predict_classes(test)
pred
sub = pd.DataFrame({"Label":pred})

sub.index.name = "ImageId"

sub.index+=1

sub
sub.to_csv('sub1.csv')