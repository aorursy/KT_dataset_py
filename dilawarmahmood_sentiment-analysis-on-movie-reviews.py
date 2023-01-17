import tensorflow as tf

from tensorflow import keras

import tensorflow_hub as hub



import numpy as np

import pandas as pd
train = pd.read_csv("../input/sentiment-analysis-on-movie-reviews/train.tsv.zip", sep='\t')

train
test = pd.read_csv("../input/sentiment-analysis-on-movie-reviews/test.tsv.zip", sep='\t')

test
sampleSubmission = pd.read_csv("../input/sentiment-analysis-on-movie-reviews/sampleSubmission.csv")

sampleSubmission
X = train.pop("Phrase")

y = train.pop("Sentiment")
dataset = tf.data.Dataset.from_tensor_slices((X, y))
train_data_batch, train_label_batch = next(iter(dataset.batch(10)))

train_data_batch
train_label_batch
embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"

model = keras.Sequential([

    hub.KerasLayer(embedding, input_shape=[], dtype=tf.string, trainable=True),

    keras.layers.Dense(128, activation='relu'),

    keras.layers.Dense(5)

])

model.summary()
model.compile(optimizer='adam',

              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),

              metrics=['accuracy'])
model.fit(dataset.shuffle(10000).batch(512),

          epochs=20,

          verbose=1)
test_x = test.pop("Phrase")

test_x
test_x = tf.constant(test_x, dtype=tf.string)
prob_model = keras.Sequential([

    model,

    keras.layers.Softmax()

])
preds = prob_model.predict(test_x)
preds