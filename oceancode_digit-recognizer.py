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
train = pd.read_csv("../input/train.csv")

test= pd.read_csv("../input/test.csv")

train.head()
test.head()
X_train=train.drop('label',axis=1)

y_train=train['label']
X_train = X_train / 255.0

test = test / 255.0

X_train = X_train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)
plt.imshow(X_train[1][:,:,0])
plt.imshow(X_train[2][:,:,0])
plt.imshow(X_train[3][:,:,0])
from keras.utils import to_categorical



y_train= to_categorical(y_train, num_classes=10)
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.1, random_state=2)
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



model.summary()

model.fit(X_train, y_train, batch_size =136,epochs =7,validation_data = (X_val, y_val), verbose = 2)
prediction= model.predict(test)



prediction= np.argmax(prediction,axis = 1)



prediction= pd.Series(prediction,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),prediction],axis = 1)



submission.to_csv("digit_recognizer.csv",index=False)