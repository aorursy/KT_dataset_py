# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_path = "/kaggle/input/mnist-in-csv/mnist_train.csv"

data = pd.read_csv(train_path)

data.describe()
data = data.values

y = data[:,0]

x = data[:,1:]

print(y.shape,x.shape)
x = x.reshape((-1,28,28,1))
x = x/255

print(np.max(x))
print(x.shape)
import cv2

import matplotlib.pyplot as plt



def display(image):

    plt.imshow(image)

    plt.show()

    

sample  = x[0].reshape((28,28))

display(sample)
import keras

from keras.models import Sequential

from keras.layers import Dense,ReLU,Conv2D,MaxPooling2D,Activation,Flatten,Dropout,BatchNormalization
model = Sequential()

model.add(Conv2D(filters=16,kernel_size=(3,3),strides=(1,1),padding='same',input_shape=(28,28,1),name='0'))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(2,2))

model.add(Conv2D(filters=32,kernel_size=(3,3),strides=(1,1),padding='same',name='1'))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(2,2))

model.add(Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same',name='2'))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(2,2))

model.add(Conv2D(filters=128,kernel_size=(3,3),strides=(1,1),padding='same',name='3'))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(2,2))

model.add(Flatten())

model.add(Dense(64))

model.add(Activation('relu'))

model.add(Dropout(0.29))

model.add(Dense(32))

model.add(Activation('relu'))

model.add(Dropout(0.25))

model.add(Dense(16))

model.add(Activation('relu'))

model.add(Dropout(0.21))

model.add(Dense(10))

model.add(Activation('softmax'))

model.summary()
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
from keras.utils import to_categorical

y = to_categorical(y)
y.shape
history = model.fit(x,y,validation_split=0.2,epochs = 18,verbose=1)
model.save('model.h5')

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()
def get_result(image):

    # image should be 28x28x1 with [0-1] values

    images = np.array([image])

    res = list(model.predict(images)[0])

    mx = max(res)

    return res.index(mx)

idx = 956

sample = x[idx]

print(y[idx])

display(sample.reshape(28,28))

print(get_result(sample))