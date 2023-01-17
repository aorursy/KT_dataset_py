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
file = '/kaggle/input/sudoku/sudoku.csv'

df = pd.read_csv(file)

print(df.describe())

print(df.head())
size = 1000000  

x = np.zeros((size,9,9),dtype='uint8')

y = np.zeros((size,9,9),dtype='uint8')
df = df.values



def process_matrix(series):

    series = str(series)

    np_small = np.array([int (x) for x in list(series)])

    return np_small.reshape((9,9))



for i in range(size):

    quiz = df[i,0]

    soln = df[i,1]

    x[i] = process_matrix(quiz)

    y[i] = process_matrix(soln)

    

print(x[200])

print(y[200])
print(y.shape,x.shape)
x = x.reshape((-1,9,9,1))

y = y.reshape((-1,9,9,1))
print(x.shape,y.shape)
import keras

from keras.layers import *

from keras.models import *
model = Sequential()

model.add(Conv2D(128,kernel_size=(3,3),strides=(2,2),activation='relu',padding='same',input_shape=(9,9,1)))

model.add(BatchNormalization())

model.add(Conv2D(32,kernel_size=(3,3),strides=(2,2),activation='relu',padding='same'))

model.add(BatchNormalization())

model.add(Conv2D(64,kernel_size=(3,3),strides=(2,2),activation='relu',padding='same'))

model.add(BatchNormalization())

model.add(Conv2D(128,kernel_size=(3,3),strides=(2,2),activation='relu',padding='same'))

model.add(BatchNormalization())

model.add(Conv2D(256,kernel_size=(3,3),strides=(2,2),activation='relu',padding='same'))

model.add(BatchNormalization())

model.add(Conv2D(512,kernel_size=(3,3),strides=(2,2),activation='relu',padding='same'))

model.add(BatchNormalization())

model.add(Conv2D(1024,kernel_size=(3,3),strides=(2,2),activation='relu',padding='same'))

model.add(Flatten())

model.add(Dense(512,activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(256,activation='relu'))

model.add(Dropout(0.1))

model.add(Dense(128,activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(9*9*1))

model.add(Reshape((9,9,1)))

model.add(Activation('relu'))



model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])

model.summary()
history=model.fit(x,y,batch_size=256,epochs=30,validation_split=0.2)

model.save('model.h5')
import matplotlib.pyplot as plt

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()