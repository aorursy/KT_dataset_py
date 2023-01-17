# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from keras import optimizers

import lightgbm as lgb

from keras.metrics import binary_accuracy

from keras.models import Sequential

from keras.layers import Dense

from keras.utils import to_categorical

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df1 = pd.read_csv('../input/export3.csv')
target = df1['t123']

target = to_categorical(target, num_classes=2)

df1 = df1.drop(['t123'], axis=1)
df1.head(3)
model = Sequential()

model.add(Dense(128, input_shape=(122,), activation='relu'))

model.add(Dense(64, activation='relu'))

model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.fit(df1, target, epochs=20, batch_size=64)