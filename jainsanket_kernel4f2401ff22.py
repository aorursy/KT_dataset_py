# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import tensorflow as tf

from tensorflow import keras

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import keras

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

data=pd.read_csv("../input/digit-recognizer/train.csv")

sdata=data.iloc[np.random.permutation(len(data))]

x_t=sdata

x_t=x_t.drop("label",axis=1)

x_t/=255

y_t=sdata["label"]

x_t=np.array(x_t)

x_t = x_t.reshape(x_t.shape[0], 28, 28, 1)

y_t = keras.utils.to_categorical(y_t, 10)

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))

model.compile(optimizer='adam',loss=keras.losses.categorical_crossentropy,metrics=['accuracy'])

model.fit(x_t,y_t,epochs=3)

print("trained")

tdata=pd.read_csv("../input/digit-recognizer/test.csv")

X_t=tdata

X_t/=255

X_t=np.array(X_t)

X_t = X_t.reshape(X_t.shape[0], 28, 28, 1)

predict=model.predict([X_t])

print(predict)

a=[]

for i in range(len(predict)):

    a.append(np.argmax(predict[i]))

out=pd.DataFrame()

out["ImageId"]=range(1,len(predict)+1)

out["Label"]=a

out.to_csv("output.csv",index=False)

print(y_t[:1])

# Any results you write to the current directory are saved as outputting 