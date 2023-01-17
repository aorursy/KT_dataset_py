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
# TensorFlow and tf.keras

import tensorflow as tf

from tensorflow import keras



# Helper libraries

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import pandas as pd
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

trn_images, valid_images, trn_labels, valid_labels = train_test_split(train_images, train_labels,test_size=0.2)
trn_images = trn_images / 255.0

test_images = test_images / 255.0

valid_images = valid_images / 255.0
model1 = keras.models.Sequential()

model1.add(keras.layers.Flatten(input_shape=[28, 28]))

model1.add(keras.layers.Dense(300, activation="relu"))

model1.add(keras.layers.Dense(10, activation="softmax"))
model2 = keras.models.Sequential()

model2.add(keras.layers.Flatten(input_shape=[28, 28]))

model2.add(keras.layers.Dense(300, activation="relu"))

model2.add(keras.layers.Dense(10, activation="softmax"))
model1.compile(loss="sparse_categorical_crossentropy",

optimizer="adam",

metrics=["accuracy"])

model2.compile(loss="sparse_categorical_crossentropy",

optimizer="adam",

metrics=["accuracy"])
history = model1.fit(trn_images, trn_labels, epochs=50,validation_data=(valid_images, valid_labels))
import pandas as pd

import matplotlib.pyplot as plt

df= pd.DataFrame(history.history)

ind=[1,3]

df.iloc[:,ind].plot(figsize=(8, 5))

plt.grid(True)

plt.gca().set_ylim(0.8, 0.98) # set the vertical range to [0.5-1]

plt.show()
model1.evaluate(test_images, test_labels)
callback = tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)

history1 = model2.fit(trn_images, trn_labels, epochs=50,validation_data=(valid_images, valid_labels),callbacks=[callback])
model2.evaluate(test_images, test_labels)