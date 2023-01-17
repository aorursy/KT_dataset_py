# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import keras 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_df=pd.read_csv("../input/fashion-mnist_train.csv")

train_df.head()
test_df=pd.read_csv("../input/fashion-mnist_test.csv")

test_df.head()
train_df.shape,test_df.shape
train_df=train_df.astype("float32")

test_df=test_df.astype("float32")
#training dataset

y_train= X=train_df["label"]

X_train=train_df.drop(columns="label",axis=1)
#testing dataset

y_test= X=test_df["label"]

X_test=test_df.drop(columns="label",axis=1)
X_train.head()
y_train.head()
X_train.shape
X_train=np.array(X_train)

X_train=X_train.reshape(X_train.shape[0],28,28)



X_test=np.array(X_test)

X_test=X_test.reshape(X_test.shape[0],28,28)
X_train[0]
X_train=X_train/255

X_test=X_test/255
X_train.shape
X_test.shape
from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

model=Sequential()

model.add(Flatten(input_shape=[28,28])) #input layer

#hand rule for hidden layer neurons, it should be less than shapeof data (28*28) and max 2 layers are enough in genral

model.add(Dense(200,activation="relu")) #hidden layer  

model.add(Dense(200,activation="relu")) #hidden layer 

model.add(Dense(10,activation="softmax")) #softamx is used in probability #output layer (0-9)

model.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
model.fit(X_train,y_train,epochs=3)#epochs is iteration
model.evaluate(X_test,y_test)
model.save("MNIST_FASHION_CNN_model")
model.summary()  # gives us details about model neurons,weights(trainable parameters)