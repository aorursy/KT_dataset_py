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

fashion_mnist = keras.datasets.fashion_mnist

(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0

y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",

 "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
X_train = X_train.reshape(55000, 28, 28, 1)

X_valid = X_valid.reshape(5000, 28, 28, 1)
model = keras.models.Sequential([

 keras.layers.Conv2D(64, 7, activation="relu", padding="same", input_shape=(28, 28, 1)),

 keras.layers.MaxPooling2D(2),

 keras.layers.Conv2D(128, 3 , activation="relu", padding="same"),

 keras.layers.Conv2D(128, 3 , activation="relu", padding="same"),

 keras.layers.MaxPooling2D(2),

 keras.layers.Conv2D(256, 3, activation="relu", padding="same"),

 keras.layers.Conv2D(256, 3, activation="relu", padding="same"),

 keras.layers.MaxPooling2D(2),

 keras.layers.Flatten(),

 keras.layers.Dense(128, activation="relu"),

 keras.layers.Dropout(0.5),

 keras.layers.Dense(64, activation="relu"),

 keras.layers.Dropout(0.5),

 keras.layers.Dense(10, activation="softmax"),

])

# Compiling the model

opt = keras.optimizers.SGD(lr=0.01, momentum=0.9)

model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# fit model

history = model.fit(X_train, y_train, epochs=1, batch_size=64, validation_data=(X_valid, y_valid))
model.evaluate(X_valid, y_valid)
X_new = X_test[:3]

X_new.shape
X_new = X_new.reshape(3, 28, 28, 1)

y_proba = model.predict(X_new)

y_proba.round(2)
y_pred = model.predict_classes(X_new)

y_pred
np.array(class_names)[y_pred]