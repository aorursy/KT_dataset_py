# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import keras
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
print(len(df_train)), print(len(df_test))
df_train.head()
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential

from keras.layers import Dense, Activation
model = Sequential()

model.add(Dense(32, input_dim=784))

model.add(Activation('relu'))



model.add(Dense(10))

model.add(Activation('softmax'))
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)



# Let's train the model using RMSprop

model.compile(loss='categorical_crossentropy',

              optimizer=opt,

              metrics=['accuracy'])



np_train = df_train.values

x_train = np_train[:, 1:]

y_train = np_train[:, 0]
np_test = df_test.values

x_test = np_test[:, :]

y_train.shape
y_train_bin = LabelBinarizer().fit_transform(y_train)

# y_test_bin = LabelBinarizer().fit_transform(y_test)
y_train_bin.shape
model.fit(x_train, y_train_bin,

          batch_size=128,

          epochs=20,

          shuffle=True)
res = model.predict_classes(x_test)
pd.DataFrame(res).to_csv("result.csv")