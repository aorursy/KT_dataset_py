# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import tensorflow

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from tensorflow.keras import Sequential

from tensorflow.keras.models import load_model

from tensorflow.keras.layers import Flatten, Dense, Dropout

from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import to_categorical

from tensorflow.keras import layers

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        pass

        #print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_ = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')
y = train_['label']
Y = to_categorical(y)
train_.drop('label',axis=1,inplace=True)
x = np.array(train_)

X = x/255
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.3)
model_dense = Sequential()

model_dense.add(layers.Dense(300, input_dim=x_train.shape[1],activation='relu'))

model_dense.add(layers.Dropout(0.2))

model_dense.add(layers.Dense(100,activation= 'relu'))

model_dense.add(layers.Dense(10,activation= 'softmax'))

model_dense.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model_dense.fit(x_train,y_train, batch_size=1000, epochs=20)
model_dense.evaluate(x_test,y_test)
raw_data_y = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')
raw_data_y.drop('id',axis=1,inplace=True)
test_data = np.array(raw_data_y)

test_data = test_data/255
y_pred = model_dense.predict_classes(test_data)
sub_ = pd.read_csv('/kaggle/input/Kannada-MNIST/sample_submission.csv')
sub_.head(5)
sub_['label'] = y_pred
sub_.to_csv('submission.csv')