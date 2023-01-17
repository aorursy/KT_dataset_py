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
train = pd.read_csv('../input/digit-recognizer/train.csv')

test = pd.read_csv('../input/digit-recognizer/test.csv')

train.head()
import numpy as np

Y_train = train['label']

X_train = train.drop('label',axis=1)

X_train_tf = X_train.to_numpy()

Y_train_tf = Y_train.to_numpy()

Y_train_tf
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X_train_tf,Y_train_tf,test_size=0.15,random_state=42)

x_train,x_test = x_train/255.0,x_test/255.0
model = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(784,)),

                            tf.keras.layers.Dense(128,activation='relu'),

                            tf.keras.layers.Dense(128,activation='relu'), 

                            tf.keras.layers.Dense(10,activation='softmax')])

model.compile(optimizer='rmsprop',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=15)
model.evaluate(x_test,y_test)