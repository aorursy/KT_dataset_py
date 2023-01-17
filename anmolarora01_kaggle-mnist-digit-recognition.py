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
df_train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

df_test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
y_train = df_train['label']

x_train = df_train.drop('label', axis = 1)
x_train = x_train/255

temp = x_train.to_numpy().reshape(42000,28,28,1)
import tensorflow as tf

model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(28,(3,3), input_shape=(28,28,1)))

model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))

model.compile(optimizer='adam', 

              loss='sparse_categorical_crossentropy', 

              metrics=['accuracy'])

model.fit(x=temp,y=y_train, epochs=10)
x_test = df_test/255

x_test = x_test.to_numpy().reshape(28000,28,28,1)

y_test = model.predict(x_test)

out = pd.DataFrame(list(map(lambda x: x.argmax(),y_test)))

out.to_csv('submit.csv')