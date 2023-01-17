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
import tensorflow as tf

from tensorflow import keras
train = pd.read_csv("../input/digit-recognizer/train.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")
train.head()
train_y_full = train.label

train_X_full = train.drop(labels=['label'], axis=1)
train_X_full = train_X_full / 255.

X_test = test / 255.

X_train, X_valid = train_X_full[:-5000], train_X_full[-5000:]

y_train, y_valid = train_y_full[:-5000], train_y_full[-5000:]
X_train = X_train.values.reshape(-1,28,28,1)

X_valid = X_valid.values.reshape(-1,28,28,1)

X_test = X_test.values.reshape(-1,28,28,1)
X_train.shape, X_valid.shape, X_test.shape
model = keras.models.Sequential([

    keras.layers.Conv2D(32, kernel_size=5, padding="same", activation="relu"),

    keras.layers.Conv2D(64, kernel_size=5, padding="same", activation="relu"),

    keras.layers.MaxPool2D(),

    keras.layers.Flatten(),

    keras.layers.Dropout(0.25),

    keras.layers.Dense(128, activation="relu"),

    keras.layers.Dropout(0.4),

    keras.layers.Dense(10, activation="softmax")

])



model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam",

              metrics=["accuracy"])





model.fit(X_train, y_train, epochs=25, validation_data=(X_valid, y_valid))
results = model.predict(X_test)



results = np.argmax(results, axis = 1)



results = pd.Series(results, name="Label")
submission = pd.concat([pd.Series(range(1, 28001), name = "ImageId"), results], axis = 1)



submission.to_csv("my_submission.csv", index=False)