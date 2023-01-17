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

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

from tensorflow.keras.callbacks import TensorBoard



data = np.array(pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv'))

#print(data.shape)

y=data[:,-1];

#print(y.shape)

#print(y)

x = np.delete(data, -1, axis=1)

#print(x.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)



#print(x_train.shape)

#print(x_test.shape)

#print(y_train.shape)

#print(y_test.shape)



model= Sequential([Flatten(),Dense(16,activation='sigmoid',input_shape=(199364,30)),Dense(16,activation='sigmoid'),Dense(1,activation='sigmoid')])

model.compile(optimizer='sgd', loss='mean_squared_error',metrics=['accuracy'])

model.fit(x_train,y_train,validation_data=(x_test,y_test), epochs=10, verbose = 1)
