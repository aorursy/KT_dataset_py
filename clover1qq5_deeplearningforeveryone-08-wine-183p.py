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
from keras.models import Sequential

from keras.layers import Dense

from keras.callbacks import ModelCheckpoint, EarlyStopping





import tensorflow as tf

import matplotlib.pyplot as plt





seed = 0

np.random.seed(seed)

tf.random.set_seed(3)



# 데이터 입력

df_pre = pd.read_csv('../input/winecsv/wine.csv', header=None)

df = df_pre.sample(frac=1)



dataset = df.values

X = dataset[:,0:12]

Y = dataset[:,12]



model = Sequential()

model.add(Dense(30,  input_dim=12, activation='relu'))

model.add(Dense(12, activation='relu'))

model.add(Dense(8, activation='relu'))

model.add(Dense(1, activation='sigmoid'))





model.compile(loss='binary_crossentropy',

           optimizer='adam',

           metrics=['accuracy'])





model.fit(X, Y, epochs=200, batch_size=200)





print("\n Accuracy: %.4f" % (model.evaluate(X, Y)[1]))
