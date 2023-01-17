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
import matplotlib.pyplot as plt

import tensorflow as tf

import numpy as np

import pandas as pd

import os



from sklearn.model_selection import train_test_split

from tensorflow import keras

from keras.utils.np_utils import to_categorical
train = pd.read_csv('../input/digit-recognizer/train.csv')

test = pd.read_csv('../input/digit-recognizer/test.csv')
# for labels 

train_y = train['label']

# for pixel values

train_x = train.drop(labels=['label'], axis=1)
# normalize the data

train_x = train_x / 255.0

test = test / 255.0
train_x = train_x.values.reshape(-1, 28, 28, 1)

test = test.values.reshape(-1, 28, 28, 1)
train_x = np.pad(train_x, ((0,0),(2,2),(2,2),(0,0)), 'constant')

test = np.pad(test, ((0,0),(2,2),(2,2),(0,0)), 'constant')
train_y = tf.keras.utils.to_categorical(train_y, num_classes=10)
train_X, val_X, train_Y, val_Y = train_test_split(train_x, train_y, test_size = 0.2)
model_2 = keras.models.Sequential ([

    keras.layers.Conv2D(6, 5, activation="relu", input_shape=(32,32,1)),

    keras.layers.AvgPool2D(pool_size=2, strides=2),

    keras.layers.Conv2D(16, 5, activation="relu"),

    keras.layers.AvgPool2D(pool_size=2, strides=2),

    keras.layers.Conv2D(120, 5, activation="relu"),

    keras.layers.Flatten(),

    keras.layers.Dense(units=84, activation="relu"),

    keras.layers.Dense(units=10, activation="softmax")

])
model_2.compile(loss="categorical_crossentropy", 

              optimizer="adam",

              metrics=["accuracy"])
checkpoint_cb_2 = keras.callbacks.ModelCheckpoint("model_2.h5", save_best_only=True)

history_2 = model_2.fit(train_X, train_Y, epochs=30, 

                    validation_data=(val_X, val_Y),

                    callbacks=[checkpoint_cb_2])
model_2 = keras.models.load_model("model_2.h5")

results_2 = model_2.predict(test)

results_2 = np.argmax(results_2,axis = 1)

results_2 = pd.Series(results_2,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"), results_2], axis = 1)

submission.to_csv("submission_2.csv",index=False)