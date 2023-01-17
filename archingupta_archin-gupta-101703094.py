# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load







# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
trainset = pd.read_csv('/kaggle/input/cat-in-the-dat/train.csv')

testset = pd.read_csv('/kaggle/input/cat-in-the-dat/test.csv')

X = trainset.iloc[:,1:24].values

Y=trainset.iloc[:,24].values

test_X = testset.iloc[:,1:].values
from sklearn.preprocessing import OneHotEncoder,LabelEncoder

for i in range(3,15):

  labelEncoder=LabelEncoder()

  X[:,i]=labelEncoder.fit_transform(X[:,i]) 

for i in range(16,21):

  labelEncoder=LabelEncoder()

  X[:,i]=labelEncoder.fit_transform(X[:,i]) 



for i in range(3,15):

  labelEncoder=LabelEncoder()

  test_X[:,i]=labelEncoder.fit_transform(test_X[:,i]) 

for i in range(16,21):

  labelEncoder=LabelEncoder()

  test_X[:,i]=labelEncoder.fit_transform(test_X[:,i]) 
import tensorflow as tf

from tensorflow import keras

model = keras.models.Sequential([

    tf.keras.layers.Dense(128,activation='relu'),

    tf.keras.layers.Dense(1,activation='sigmoid'),

])
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['binary_accuracy'])

model.fit(np.asarray(X).astype(np.int32), np.asarray(Y).astype(np.int32), epochs=30, batch_size=100, verbose=1)
predictions = model.predict(np.asarray(test_X).astype(np.int32))
submit = pd.concat([testset['id'], pd.Series(predictions[:,0]).rename('target')], axis=1)

submit.to_csv('output.csv', index=False, header=True)

submit