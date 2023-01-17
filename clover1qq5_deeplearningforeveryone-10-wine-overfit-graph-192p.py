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

from keras.callbacks import ModelCheckpoint



import os

import matplotlib.pyplot as plt

import tensorflow as tf


np.random.seed(3)

tf.random.set_seed(3)



df_pre = pd.read_csv('../input/winecsv/wine.csv', header=None)

df = df_pre.sample(frac=0.15)



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


MODEL_DIR = './model/'

if not os.path.exists(MODEL_DIR):

   os.mkdir(MODEL_DIR)


modelpath="./model/{epoch:02d}-{val_loss:.4f}.hdf5"

checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=1, save_best_only=True)


history = model.fit(X, Y, validation_split=0.33, epochs=3500, batch_size=500)


y_vloss=history.history['val_loss']


y_acc=history.history['accuracy'] # 책에서는 'acc'라고 돼있는데 이걸 줄이면 안 되고 'accuracy'라고 명확히 해줘야한다. 


x_len = np.arange(len(y_acc))

plt.plot(x_len, y_vloss, "o", c="red", markersize=3)

plt.plot(x_len, y_acc, "o", c="blue", markersize=3)



plt.show()