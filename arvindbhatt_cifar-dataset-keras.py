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
import keras
from keras.datasets import cifar10
data=cifar10.load_data()
data=np.array(data)
X_train=data[0][0]
y_train=data[0][1]
X_test=data[1][0]
y_test=data[1][0]


X_train
y_train=keras.utils.to_categorical(y_train,10)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
X_train=X_train.reshape((50000,32,32,3))
import keras
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Dropout,Flatten,MaxPooling2D



model=Sequential()
model.add(Conv2D(32, kernel_size=(3,3),padding="same",input_shape=(32,32,3),activation="relu"))
model.add(Conv2D(64,kernel_size=(3,3),padding="valid",activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(32,kernel_size=(3,3),padding="valid",activation="relu"))
model.add(Flatten())
model.add(Dense(30,activation="relu"))
model.add(Dense(10,activation="softmax"))
opt = keras.optimizers.rmsprop()

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
model.fit(X_train,y_train,batch_size=100,epochs=5,verbose=1)
model.summary()