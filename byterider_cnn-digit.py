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
import tensorflow as tf

from tensorflow import keras
(train_X,train_Y),(test_X,test_Y) = tf.keras.datasets.mnist.load_data()

ques = pd.read_csv("../input/test.csv")

train_X = tf.keras.utils.normalize(train_X,axis=1)

test_X = tf.keras.utils.normalize(test_X,axis=1)

test = ques.values

print (train_X.shape,test_X.shape)



train_X = train_X.reshape(-1,28,28,1)

test_X = train_X.reshape(-1,28,28,1)

test = test.reshape(-1,28,28,1)
model = keras.models.Sequential()

model.add(keras.layers.Conv2D(filters=32,kernel_size=(5,5),activation=tf.nn.relu,padding='SAME'))



model.add(keras.layers.Conv2D(filters=64,kernel_size=(5,5),activation='relu',padding="SAME"))

#model.add(keras.layers.BatchNormalization())

model.add(keras.layers.MaxPooling2D(pool_size=(3,3)))

model.add(keras.layers.Dropout(0.25))



model.add(keras.layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu',padding="SAME"))

#model.add(keras.layers.BatchNormalization())

model.add(keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)))

model.add(keras.layers.Dropout(0.5))



model.add(keras.layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding="SAME"))

model.add(keras.layers.MaxPooling2D(pool_size=(3,3),strides=(2,2)))

model.add(keras.layers.Dropout(0.5))







model.add(keras.layers.Flatten())



model.add(keras.layers.Dense(128,activation='relu'))

model.add(keras.layers.Dense(256,activation='relu'))

model.add(keras.layers.Dense(128,activation='relu'))

model.add(keras.layers.Dense(10,activation='softmax'))



model.compile(metrics=['accuracy'],loss='sparse_categorical_crossentropy',optimizer='adam')

model.fit(train_X,train_Y,epochs=15)

print (model.evaluate(test_X,test_Y))
answer = model.predict(test)



f = open("answer.txt","w")

f.write("ImageId,Label\n")

i = 1

for j in answer:

    f.write(str(i)+","+str(np.argmax(j))+"\n")

    i += 1

f.close()