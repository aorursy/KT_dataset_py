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
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.layers.core import Dense, Dropout, Flatten
train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")
train_data.head()
train_data.info()
test_data.info()
train_data["label"].value_counts()
srand = 112
X = train_data.drop("label", axis=1)
Y = train_data["label"]

X = X / 255.0
X_pred = test_data / 255.0
X = X.values.reshape(-1, 28, 28, 1)
X_pred = X_pred.values.reshape(-1, 28, 28, 1)
Y = to_categorical(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=srand)
# plt.imshow(X_train[0][:,:,0])
# plt.title(Y_train[0])
# plt.show()
fig, ax = plt.subplots(1, 5, figsize = (15,3), sharey=True)
for i in range(len(ax)):
    ax[i].set_title(Y_train[i])
    ax[i].imshow(X_train[i][:,:,0])
plt.show()
model = Sequential()
# In:28=>Padding:28=>(5,5)conv:24=>pool:12=>(5,5)conv:8=>pool:4=>flatten=>dense=>Out(dim10 One-Hot)

model.add(Conv2D(filters=20, kernel_size=5, input_shape=(28, 28, 1), activation="relu"))
model.add(MaxPool2D(pool_size=2))
model.add(Conv2D(filters=20, kernel_size=5, activation="relu"))
model.add(MaxPool2D(pool_size=2))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))
model.compile(optimizer="RMSprop", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, Y_train, epochs=10, validation_split=0.2)
model.evaluate(X_test, Y_test)
pred = model.predict_classes(X_pred)
fig, ax = plt.subplots(1, 10, figsize = (20,2), sharey=True)
for i in range(len(ax)):
    ax[i].set_title(pred[i])
    ax[i].imshow(X_pred[i][:,:,0])
plt.show()
submission = pd.read_csv("../input/sample_submission.csv")
submission["Label"] = pred
submission.head()
submission.to_csv("my_submission.csv", index=False)