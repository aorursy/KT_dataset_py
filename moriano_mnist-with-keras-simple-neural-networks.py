# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import keras

from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D, Activation

from keras.models import Sequential

from sklearn.model_selection import train_test_split

from keras.optimizers import RMSprop
train_raw = pd.read_csv("../input/train.csv")

test_raw = pd.read_csv("../input/test.csv")
print(train_raw.shape)

print(test_raw.shape)

train, val = train_test_split(train_raw, train_size=0.8)
train.shape
val.shape
y_train = train["label"]

del train["label"]

X_train = train



y_val = val["label"]

del val["label"]

X_val = val
X_train = X_train.astype("float32")

X_val = X_val.astype("float32")

X_train = X_train / 255

X_val = X_val / 255
y_train = keras.utils.to_categorical(y_train, 10)

y_val = keras.utils.to_categorical(y_val, 10)
model = Sequential()

model.add(Dense(128, activation='relu', input_shape=(784,)))

model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy',

              optimizer=RMSprop(),

metrics=['accuracy'])
model.summary()
history = model.fit(X_train, y_train,

                    batch_size=128,

                    epochs=10,

                    verbose=1,

                    validation_data=(X_val, y_val))
history.history
to_plot = pd.DataFrame([history.history["acc"], history.history["val_acc"]])

to_plot = to_plot.T

to_plot
to_plot.columns=["acc", "val_acc"]
to_plot
to_plot.plot(figsize=(15, 5))
model = Sequential()

model.add(Dense(512, activation='relu', input_shape=(784,)))

model.add(Dense(512, activation='relu', input_shape=(784,)))

model.add(Dense(256, activation='relu', input_shape=(784,)))

model.add(Dense(10, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',

              optimizer=RMSprop(),

metrics=['accuracy'])
history = model.fit(X_train, y_train,

                    batch_size=128,

                    epochs=10,

                    verbose=1,

                    validation_data=(X_val, y_val))
to_plot = pd.DataFrame([history.history["acc"], history.history["val_acc"]])

to_plot = to_plot.T

to_plot.columns=["acc", "val_acc"]
to_plot.plot(figsize=(15, 5))