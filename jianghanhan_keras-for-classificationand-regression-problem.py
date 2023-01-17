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
import tensorflow as tf 

import pandas as pd

import matplotlib.pyplot as plt

import tensorflow.keras as keras 

import tensorflow.keras.layers as layers

print(tf.__version__) ##查看版本
##导入数据

from sklearn.datasets import load_boston

data = load_boston()

x = data.data

y = data.target

#print(x.shape,y.shape)



x_train = x[0:404]

x_test = x[404:]

x_train.shape,x_test.shape



y_train = y[0:404]

y_test = y[404:]

#(x_train,y_train),(x_test,y_test) = load_boston()



##查看数据维度

print(x_train.shape,y_train.shape)

print(y_test.shape,x_test.shape)
##用 tensorflow.keras 搭建网络



model = keras.Sequential( [ 

     layers.Dense(32,activation = 'sigmoid',input_shape =(13,))

    ,layers.Dense(32,activation = 'sigmoid')

    ,layers.Dense(32,activation = 'sigmoid')

    ,layers.Dense(1)

     ])



##全连接层
# 定义训练模型

model.compile(optimizer=keras.optimizers.Adam(),

             loss='mean_squared_error',  # keras.losses.mean_squared_error

             metrics=['mse'])

model.summary()

##训练模型

history = model.fit(x_train,y_train,batch_size=50,epochs=50,validation_split=0.2,verbose = 1)
##评估模型



scores = model.evaluate(x_test,y_test)

print(model.metrics_names)   ##查看metrics 

scores
##学习曲线

def plot_learning_curves(history):

    pd.DataFrame(history.history).plot(figsize=(8, 5))

    plt.grid(True)

    plt.gca().set_ylim(0, 100)

    plt.show()



plot_learning_curves(history)



##这里我画曲线，不知道为啥会这样..
from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split

import tensorflow as tf 

import pandas as pd

import matplotlib.pyplot as plt

import tensorflow.keras as keras 

import tensorflow.keras.layers as layers

print(tf.__version__) ##查看版本







data = load_breast_cancer()

x = data.data

y = data.target



##分训练集与测试集

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 7)



print(x_train.shape,y_train.shape)

print(x_test.shape,y_test.shape)
##建立模型

model = keras.Sequential( [ 

     layers.Dense(32,activation = 'relu',input_shape =(30,))

    ,layers.Dense(32,activation = 'relu')

    ,layers.Dense(1,activation = 'sigmoid')

     ])
# 配置模型

model.compile(optimizer=keras.optimizers.Adam(),

             loss='binary_crossentropy',  # keras.losses.mean_squared_error 

              ##二分类问题用的loss

             metrics=['accuracy'])

model.summary()
##训练模型

history = model.fit(x_train, y_train,validation_split=0.1, batch_size=64, epochs=100, verbose=0)
##评估模型

scores = model.evaluate(x_test,y_test)

print(model.metrics_names)

scores
##学习曲线

def plot_learning_curves(history):

    pd.DataFrame(history.history).plot(figsize=(8, 5))

    plt.grid(True)

    plt.gca().set_ylim(0, 1)

    plt.show()



plot_learning_curves(history)