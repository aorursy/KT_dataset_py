!pip install -U tensorflow-gpu==2.0.0-beta1
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

print(tf.__version__)
!pip install tensorflow-datasets
import tensorflow_datasets as tfds
from __future__ import absolute_import, division, print_function, unicode_literals



import numpy as np



import tensorflow_hub as hub



print("Version: ", tf.__version__)

print("Eager mode: ", tf.executing_eagerly())

print("Hub version: ", hub.__version__)

print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")
# 将训练集按照 6:4 的比例进行切割，从而最终我们将得到 15,000

# 个训练样本, 10,000 个验证样本以及 25,000 个测试样本

train_validation_split = tfds.Split.TRAIN.subsplit([6, 4])



(train_data, validation_data), test_data = tfds.load(

    name="imdb_reviews", 

    split=(train_validation_split, tfds.Split.TEST),

    as_supervised=True)
train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))

train_examples_batch
train_labels_batch
embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"

hub_layer = hub.KerasLayer(embedding, input_shape=[], 

                           dtype=tf.string, trainable=True)

hub_layer(train_examples_batch[:3])
model = tf.keras.Sequential()

model.add(hub_layer)

model.add(tf.keras.layers.Dense(16, activation='relu'))

model.add(tf.keras.layers.Dense(1, activation='sigmoid'))



model.summary()
model.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['accuracy'])
history = model.fit(train_data.shuffle(10000).batch(512),

                    epochs=20,

                    validation_data=validation_data.batch(512),

                    verbose=1)
results = model.evaluate(test_data.batch(512), verbose=2)

for name, value in zip(model.metrics_names, results):

  print("%s: %.3f" % (name, value))