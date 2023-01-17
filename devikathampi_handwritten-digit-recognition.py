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

from keras.datasets import mnist

from keras.models import Sequential

from keras.layers import Dense,Dropout,Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras import backend as K

import matplotlib.pyplot as plt
(x_train,y_train),(x_test,y_test) = mnist.load_data()

print(x_train.shape,y_train.shape)
num_classes=10

x_train = x_train.reshape(x_train.shape[0],28,28,1)

x_test = x_test.reshape(x_test.shape[0],28,28,1)

input_shape=(28,28,1)



y_train =keras.utils.to_categorical(y_train,num_classes)

y_test = keras.utils.to_categorical(y_test,num_classes)



x_train = x_train.astype("float32")

x_test = x_test.astype("float32")

x_train/=255

x_test/=255

print("x_train shape is",x_train.shape)

print("x_train samples:",x_train.shape[0])

print("x_test samples:",x_test.shape[0])
batch_size=128



epochs=10



model=Sequential()

model.add(Conv2D(32, kernel_size=(5,5),activation='relu',input_shape=input_shape))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3,3), activation="relu"))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(64, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))



model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=["accuracy"])

model.summary()
model_hist = model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test, y_test))



print("The model has successfully trained")

model.save('mnist.h5')
score = model.evaluate(x_test,y_test,verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
image_index = 5574

plt.imshow(x_test[image_index].reshape(28, 28),cmap='Greys')

pred = model.predict(x_test[image_index].reshape(1, 28, 28, 1))

print('The number predicted is : ', pred.argmax())