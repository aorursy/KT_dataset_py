# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import tensorflow as tf


gpus = tf.config.experimental.list_physical_devices('GPU')

for gpu in gpus:

    print("Name:", gpu.name, "  Type:", gpu.device_type)



from tensorflow.python.client import device_lib



device_lib.list_local_devices()



tf.test.is_gpu_available()
mnist = tf.keras.datasets.mnist



(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0
model = tf.keras.models.Sequential([

  tf.keras.layers.Flatten(input_shape=(28, 28)),

  tf.keras.layers.Dense(128, activation='relu'),

  tf.keras.layers.Dropout(0.2),

  tf.keras.layers.Dense(10)

])
predictions = model(x_train[:1]).numpy()

predictions
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam',

              loss=loss_fn,

              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)


model.evaluate(x_test,  y_test, verbose=2)
