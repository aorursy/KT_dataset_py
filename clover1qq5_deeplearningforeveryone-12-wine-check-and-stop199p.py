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
#책이랑 다르게 에러. keras부분 tensorflow.keras로 수정



from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping





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


early_stopping_callback = EarlyStopping(monitor='val_loss', patience=100)



model.fit(X, Y, validation_split=0.2, epochs=3500, batch_size=500, verbose=0, callbacks=[early_stopping_callback,checkpointer])