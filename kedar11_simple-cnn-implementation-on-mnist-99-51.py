# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from keras.layers import Flatten,Dense

from keras.models import Sequential

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

import matplotlib.pyplot as plt

import seaborn as sb

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))





train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')

# Any results you write to the current directory are saved as output.
X_train=train.drop('label',axis=1)

X_train = X_train / 255.0

test = test / 255.0

X_train = X_train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)



y_train=train['label']

y_train=to_categorical(y_train,num_classes=10)

y_train[:5]
print(X_train.shape)

print(y_train.shape)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.1, random_state=2)
plt.imshow(X_train[12][:,:,0])
model = Sequential()



model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',activation ='relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.1))



model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',activation ='relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.2))



model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.4))

model.add(Dense(10, activation = "softmax"))



model.compile(optimizer = 'adam' , loss = "categorical_crossentropy", metrics=["accuracy"])



model.fit(X_train, y_train, batch_size =136,epochs =10,validation_data = (X_val, y_val), verbose = 2)
model.summary()
prediction= model.predict(test)



# select the indix with the maximum probability

prediction= np.argmax(prediction,axis = 1)



prediction= pd.Series(prediction,name="Label")
print(prediction[0])

plt.imshow(test[0][:,:,0])