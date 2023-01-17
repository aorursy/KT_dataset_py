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
from keras.models import Sequential

from keras.layers import Conv2D,Dense,Flatten,Dropout,MaxPool2D

from keras.optimizers import Adam,RMSprop

from keras.initializers import RandomNormal

import matplotlib.pyplot as plt

import h5py



import seaborn as sns

import os

#import Files

trainFile = h5py.File('../input/happy-house-dataset/train_happy.h5')

testFile = h5py.File('../input/happy-house-dataset/test_happy.h5')
train_x = np.array(trainFile['train_set_x'][:])

train_y = np.array(trainFile['train_set_y'][:])



test_x = np.array(testFile['test_set_x'][:])

test_y = np.array(testFile['test_set_y'][:])

train_y = train_y.reshape((1, train_y.shape[0]))

test_y = test_y.reshape((1, test_y.shape[0]))

#Normilization 

X_train = train_x / 255.0

X_test = test_x / 255.0



y_train = train_y.T

y_test = test_y.T
#create Model

model = Sequential()



model.add(Conv2D(filters=24, kernel_size=(5,5), activation='relu', padding='Same', input_shape=(64,64,3)))

model.add(Conv2D(filters=48, kernel_size=(5,5), activation='relu', padding='Same'))

model.add(Conv2D(filters=64, kernel_size=(5,5), activation='relu', padding='Same'))



model.add(MaxPool2D(pool_size=(2,2)))



model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='Same'))



model.add(MaxPool2D(pool_size=(2,2)))



model.add(Flatten())



model.add(Dense(128, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

epochs = 30

batch_size = 30

history = model.fit(x=X_train, y=y_train, epochs=epochs, verbose=2,batch_size=batch_size)
test_score = model.evaluate(X_test, y_test, verbose=1)

print('test loss:', test_score[0])

print('test accuracy:', test_score[1])