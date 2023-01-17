# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import tensorflow as tf

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/train.csv')

label = data.label.values

#label = pd.get_dummies(data.label).values.reshape([42000,10,1])

data = data.drop('label', axis=1).values
data = data.reshape([42000, 28, 28])

data = data/255
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.CuDNNLSTM(128, input_shape=(28,28), return_sequences=True))

model.add(tf.keras.layers.CuDNNLSTM(128))

model.add(tf.keras.layers.Dense(10, activation='relu'))

model.add(tf.keras.layers.Dense(10, activation='softmax'))



model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(data, label, epochs=10)
data = pd.read_csv('../input/test.csv').values

data = data.reshape([data.shape[0], 28, 28])

data = data/255
pred = model.predict(data)
r=[]

for p in pred:

    r.append(np.argmax(p))

res = pd.DataFrame()

res['Label'] = r

res.index = res.index+1

res.index.names = ['ImageId']
res.to_csv('result.csv')