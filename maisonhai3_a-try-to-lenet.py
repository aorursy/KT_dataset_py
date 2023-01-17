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

from keras.layers import Dense, Conv2D, AveragePooling2D, MaxPooling2D, Activation,Flatten

import keras
# LeNet 



# Make the model

'''

lenetHaiModel = Sequential()

lenetHaiModel.add(Conv2D(6,(5,5),strides=1,))

lenetHaiModel.add(AveragePooling2D(6,(2,2),strides=2))

lenetHaiModel.add(Conv2D(16,(5,5),strides=1))

lenetHaiModel.add(AveragePooling2D(16,(2,2),strides=2))

lenetHaiModel.add(Convo2D(120,(5,5), strides=1))

lenetHaiModel.add(Dense(units=84,activation='tanh'))

lenetHaiModel.add(Dense(units=10,activation='softmax',name='output'))



# Compile but not yet fit the model

lenetHaiModel.compile(optimizer='rmsprop',

                     loss='categorical_crossentropy',

                     metrics=['accuracy'])

'''

lenetHaiModel = Sequential()

# Convo2D 1.

lenetHaiModel.add(Conv2D(6, kernel_size=(5,5), strides=1,activation='tanh', input_shape=(28,28,1), padding='same'))

#  Aver Pooling 2D 2

lenetHaiModel.add(AveragePooling2D(pool_size=(2,2), strides=(1,1), padding='valid'))

# Convo2D 3

lenetHaiModel.add(Conv2D(16, kernel_size=(10,10), strides=1, activation='tanh', padding='valid')) # input_shape?

# Aver Pooling 2D 4

lenetHaiModel.add(AveragePooling2D(pool_size=(2,2), strides=(1,1), padding='valid'))

# Convo2D 5

lenetHaiModel.add(Conv2D(120, kernel_size=(5,5), strides=1, activation='tanh', padding='valid'))

# Flatten before Dense

lenetHaiModel.add(Flatten())

# Dense 6

lenetHaiModel.add(Dense(84, activation='tanh'))

# Output 

lenetHaiModel.add(Dense(10, activation='softmax'))

# Compile

lenetHaiModel.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])

# Import the data

train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')



## Take a look before any manipulation.

print('The shape of train.csv: ' + str(train.shape))

print('The shape of test.csv: ' + str(test.shape))

# Assign X_train, X_test..

Y_train = train['label']

X_train = train.drop('label',axis=1)



X_test = test



# Normalization

X_train /= 255

X_test /= 255



# Reshape back to images.

X_train = X_train.values.reshape(-1,28,28,1)

X_test = X_test.values.reshape(-1,28,28,1)



# Do one-hot on the Y_train

from keras.utils.np_utils import to_categorical

Y_train = to_categorical(Y_train, num_classes =10)



# Splitting

from sklearn.model_selection import train_test_split

X_train, X_val, Y_train, Y_val = train_test_split(X_train,Y_train,test_size=0.33,random_state=42) # random_state??

print(Y_train.shape)
lenetHaiModel.fit(X_train,Y_train,batch_size=128,epochs=2, verbose=1,validation_data=(X_val,Y_val))
results = lenetHaiModel.predict(X_test)



results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("LeNet-5.csv",index=False)