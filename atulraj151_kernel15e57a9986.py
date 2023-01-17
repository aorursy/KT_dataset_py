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
train_orig=pd.read_csv("../input/train.csv")

test_orig=pd.read_csv("../input/test.csv")
y_train=train_orig['label'].values

x_train=train_orig.drop(labels='label',axis=1)
x_train=x_train.values

x_test=test_orig.values

x_train=x_train.reshape(x_train.shape[0],28,28,1)

x_test=x_test.reshape(x_test.shape[0],28,28,1)
print("dimension of training set",x_train.shape)

print("dimension of test set",x_test.shape)

#normalisation of the data

x_train=x_train.astype("float32")

x_test=x_test.astype("float32")
x_train=x_train/255

x_test=x_test/255
import tensorflow as tf

from keras.models import Sequential

from keras.layers import Dense,Conv2D,Dropout,Flatten,MaxPool2D
model=Sequential()

model.add(Conv2D(filters=30,kernel_size=(3,3),input_shape=(28,28,1)))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(128,activation=tf.nn.relu))

model.add(Dropout(.2))

model.add(Dense(10,activation=tf.nn.softmax))

model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=['accuracy'])

model.fit(x=x_train,y=y_train,epochs=30)
y_pred=model.predict(x_test)
l=[]

for i in range(y_pred.shape[0]):

    l.append(y_pred[i].argmax())
with open('predictions.csv', 'w') as predictions_file:

    predictions_file.write('ImageId,Label' + '\n')

    for i in range(len(x_test)):

        predictions_file.write(f'{i + 1},{l[i]}' + '\n')

