import tensorflow as tf

from tensorflow import keras

import tensorflow_hub as hub



import numpy as np

import pandas as pd
df = pd.read_csv("../input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv")
df.loc[df["sentiment"] == "positive", "sentiment"] = 1

df.loc[df["sentiment"] == "negative", "sentiment"] = 0

df
target = df.pop("sentiment")
dataset = tf.data.Dataset.from_tensor_slices((df.values, target))
train_val_data = dataset.take(25000)

test_data = dataset.skip(25000)
train_val = train_val_data.take(10000) # 40% for validation

train_data = train_val_data.skip(10000) # 60% for training
train_data_batch, train_label_batch = next(iter(train_data.batch(10)))

train_data_batch
train_label_batch
embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"

model = keras.Sequential([

    hub.KerasLayer(embedding, input_shape=[], dtype=tf.string, trainable=True),

    keras.layers.Dense(16, activation='relu'),

    keras.layers.Dense(1)

])

model.summary()
model.compile(optimizer='adam',

              loss=keras.losses.BinaryCrossentropy(from_logits=True),

              metrics=['accuracy'])
train_data.shuffle(15000)

model.fit(x=tf.reshape(next(iter(train_data.batch(512)))[0], [-1,]),

          y=next(iter(train_data.batch(512)))[1],

          epochs=20,

          verbose=1)
test_loss, test_acc = model.evaluate(x=tf.reshape(next(iter(test_data.batch(512)))[0], [-1,]),

                                     y=next(iter(test_data.batch(512)))[1],

                                     verbose=2)