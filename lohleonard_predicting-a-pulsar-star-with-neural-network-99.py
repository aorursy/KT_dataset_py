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
import pandas as pd

import numpy as np
df = pd.read_csv("../input/pulsar_stars.csv")

df.head()
df.info()
data = df.values; data
data.shape
feature = data[:,:-1]

label = data[:,-1]
print("shape of feature is: ", feature.shape)

print("shape of label is: ", label.shape)
for col_idx in range(8):

    col_data = feature[:,col_idx]

    col_data -= np.min(col_data, axis=0)

    col_data /= (np.max(col_data, axis=0) - np.min(col_data, axis=0))
feature
x_train = feature[:12528, :]

y_train = label[:12528]

x_test = feature[12528:, :]

y_test = label[12528:]



print("x_train shape: ", x_train.shape)

print("y_train shape: ", y_train.shape)

print("x_test shape: ", x_test.shape)

print("y_test shape: ", y_test.shape)
from keras.utils import to_categorical



y_train = to_categorical(y_train, num_classes=2)

y_test = to_categorical(y_test, num_classes=2)
from keras import Sequential

from keras.layers import Dense



model = Sequential()

model.add(Dense(50, activation="relu", input_dim=8))

model.add(Dense(50, activation="relu"))

model.add(Dense(2, activation="softmax"))

model.compile(optimizer='rmsprop',

              loss='binary_crossentropy',

              metrics=['accuracy'])
model.summary()
model.fit(x_train, y_train, validation_data=(x_test, y_test) ,epochs=50, batch_size=32)
from sklearn.metrics import classification_report



y_pred = model.predict(x_test, batch_size=64, verbose=1)

y_pred_bool = np.argmax(y_pred, axis=1)

y_true = np.argmax(y_test, axis=1)

print(classification_report(y_true, y_pred_bool))