# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_train.csv')
data_test = pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_test.csv')
data_test.head()
X_train = data.drop(columns=['label']).values
X_test = data_test.drop(columns=['label']).values
print(X_train.shape)
print(X_test.shape)
X_train = X_train.reshape(X_train.shape[0],28,28,1)
X_test = X_test.reshape(X_test.shape[0],28,28,1)
input_shape = (28,28,1)
y_train = data[['label']].values
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Conv2D, MaxPool2D, Activation, Flatten
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.initializers import he_normal,he_uniform
model = Sequential()
model.add(Conv2D(28,kernel_size=(3,3),input_shape=input_shape))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(28,kernel_size=(3,3)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(28,kernel_size=(3,3)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128,activation='relu',kernel_initializer=he_normal(32)))
model.add(Dropout(0.2))
model.add(Dense(10,activation=tf.keras.activations.softmax))
model.summary()
model.compile(optimizer=tf.keras.optimizers.Adadelta(0.001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train,epochs=30)
y_pred = model.predict_classes(X_test)
y_ans = pd.DataFrame(y_pred,columns=['Label'])
y_ans.head()
y_ans.set_index([list(range(1,10001))],inplace=True)
y_ans.index.name='ImageId'
y_ans.head()
y_ans.to_csv('solution.csv')
