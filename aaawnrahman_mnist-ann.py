# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#import dataset.....

dataset_train = pd.read_csv("../input/mnist_train.csv")

print(dataset_train)

dataset_test = pd.read_csv("../input/mnist_test.csv")

print(dataset_test)
#feature initializing...

y_train = np.array(dataset_train.iloc[:,0])

X_train = np.array(dataset_train.iloc[:,1:])

X_test = np.array(dataset_test.iloc[:,1:]) 

y_test = np.array(dataset_test.iloc[:,0])

X_train = X_train.reshape(60000, 784)

X_test = X_test.reshape(10000, 784)

print(X_train , y_train)

print(X_train.shape)

print(y_test.shape)

#onehot encoding.....

from keras.utils import np_utils

y_train = np_utils.to_categorical(y_train, 10)

y_test = np_utils.to_categorical(y_test, 10)

print(y_train.shape)
#import libraries....



from keras.datasets import mnist

from keras.models import Sequential, load_model

from keras.layers.core import Dense, Dropout, Activation





#model building ....



#layer 1.....

model = Sequential()

model.add(Dense(512, input_shape=(784,)))

model.add(Activation('relu'))                            

model.add(Dropout(0.2))

#layer 2......

model.add(Dense(128))

model.add(Activation('relu'))

model.add(Dropout(0.2))



#final......

model.add(Dense(10))

model.add(Activation('softmax'))

model.summary()
#compiling model....

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

#fit the model...



fit_model = model.fit(X_train, y_train,epochs=20,batch_size=10000)