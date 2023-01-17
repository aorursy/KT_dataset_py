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
import tensorflow as tf
import pandas as pd
path = "../input/Iris.csv"
dataset = pd.read_csv(path)
dataset[:10]
inputs = dataset[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
outputs_pre = dataset[["Species"]]
outputs = pd.get_dummies(outputs_pre).values
model = tf.keras.Sequential(name="model")
model.add(tf.keras.layers.Dense(32, input_shape=(4,), name="input_layer", activation="relu"))
model.add(tf.keras.layers.Dense(64, name="hidden_ly1", activation="relu"))
model.add(tf.keras.layers.Dense(128, name="hidden_ly2", activation="relu"))
model.add(tf.keras.layers.Dense(3, name="output_layer", activation="softmax"))
model.summary()
model.compile("rmsprop", "categorical_crossentropy", metrics=["accuracy"])
model.fit(inputs, outputs, epochs=100)
print("Success rate : {}%".format(int(model.evaluate(inputs, outputs, 128)[1]*100)))
