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
from __future__ import print_function

import keras

from keras.datasets import cifar10

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Conv2D, MaxPooling2D

import os



import matplotlib.pyplot as plt
# The data, split between train and test sets:

(x_train, y_label_train), (x_test, y_label_test) = cifar10.load_data()  ##利用keras里头的包就有资料

print('x_train shape:', x_train.shape)       ##查看维度

print(x_train.shape[0], 'train samples')

print(x_test.shape[0], 'test samples')



# Normalize the data. Before we need to connvert data type to float for computation.

x_train = x_train.astype('float32')     ##转换字符类型

x_test = x_test.astype('float32')

x_train /= 255                          ##压缩到0-1之间

x_test /= 255



from keras.utils import np_utils 

y_label_train_onehot = np_utils.to_categorical(y_label_train)   ##标签分类问题 变成one_hot的形式

y_label_test_onehot = np_utils.to_categorical(y_label_test)
##用keras.models 模块搭建网络 方便快速!!



model = Sequential()



model.add(Conv2D(filters = 32,input_shape = (32,32,3),kernel_size=(3,3),padding = 'same',activation='relu'))

model.add(Dropout(0.2))

model.add(Conv2D(filters = 32,kernel_size=(3,3),padding = 'same',activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Conv2D(filters = 64,kernel_size=(3,3),padding = 'same',activation='relu'))

model.add(Dropout(0.2))

model.add(Conv2D(filters = 64,kernel_size=(3,3),padding = 'same',activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Conv2D(filters = 128,kernel_size=(3,3),padding = 'same',activation='relu'))

model.add(Dropout(0.2))

model.add(Conv2D(filters = 128,kernel_size=(3,3),padding = 'same',activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Flatten())

model.add(Dropout(0.2))

model.add(Dense(2500,activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(1500,activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(10,activation='softmax'))



##定义模型训练!!

model.compile(loss = 'categorical_crossentropy'

             ,optimizer = 'adam',metrics = ['accuracy'])



##查看模型!!

model.summary()







##训练模型

history = model.fit(x_train,y_label_train_onehot,validation_split=0.2,epochs = 5,batch_size=300,verbose = 1)
##模型评估

scores = model.evaluate(x_test,y_label_test_onehot,verbose = 0)

scores[1]
def plot_learning_curves(history):

    pd.DataFrame(history.history).plot(figsize=(8, 5))

    plt.grid(True)

    plt.gca().set_ylim(0, 1)

    plt.show()



plot_learning_curves(history)
## 花太长时间计算了 