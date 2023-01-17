# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import tensorflow as tf

from tensorflow import keras

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plot





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/ex1data1.txt")

print(df.shape)
print(df.head())
print(df.iloc[0])
x=np.array(df['6.1101'])

print(len(x))

print(x)
y=np.array(df['17.592'])

print(len(y))

print(y)
model=tf.keras.Sequential([keras.layers.Dense(units=4, input_shape=[1])])
model.compile(optimizer='sgd',loss='mean_squared_error')

train=model.fit(x,y,epochs=1000)

print(model.predict([7.0032]))
print(len(model.predict(x)))