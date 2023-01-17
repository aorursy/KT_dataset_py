import os

import random

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split
import tensorflow as tf

print("Version: ", tf.__version__)

print("Eager mode: ", tf.executing_eagerly())

print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")
SEED = 42
os.environ['PYTHONHASHSEED']=str(SEED)

random.seed(SEED)

np.random.seed(SEED)

tf.random.set_seed(SEED)
pd.options.display.max_colwidth = 150
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
dataset_raw = pd.read_csv('/kaggle/input/twitter-sentiment-analysis-hatred-speech/train.csv')

dataset_raw.shape
dataset_raw
print("Number of True labels: %d " % dataset_raw['label'].astype(bool).sum(axis=0))
test_raw = pd.read_csv('/kaggle/input/twitter-sentiment-analysis-hatred-speech/test.csv')

test_raw.shape
test_raw
data_tweets = dataset_raw['tweet']

data_labels = dataset_raw['label']
train_tweets, test_tweets, train_labels, test_labels = train_test_split(

    data_tweets, data_labels, test_size=0.2,random_state=SEED)

train_tweets, val_tweets, train_labels, val_labels = train_test_split(

    train_tweets, train_labels, test_size=0.2,random_state=SEED)
print(train_tweets.shape, val_tweets.shape, test_tweets.shape)
train_dataset = tf.data.Dataset.from_tensor_slices((train_tweets, train_labels))

val_dataset = tf.data.Dataset.from_tensor_slices((val_tweets, val_labels))

test_dataset = tf.data.Dataset.from_tensor_slices((test_tweets, test_labels))
# Check training batches

train_examples_batch, train_labels_batch = next(iter(train_dataset.batch(10)))

train_examples_batch
import tensorflow_hub as hub

print("Hub version: ", hub.__version__)
# load tfhub skipgram version of word2vec with 1 out-of-vocabulary bucket; maps to 500-dimensional vectors

embed = hub.load("https://tfhub.dev/google/Wiki-words-500-with-normalization/2")
# testing

test = embed(['Shall I compare thee to a summer\'s day'])

test_2 = embed(['Thou art more lovely and more temperate'])



print(tf.keras.losses.cosine_similarity(

    test,

    test_2,

    axis=-1

    ))
hub_embedding = hub.KerasLayer("https://tfhub.dev/google/Wiki-words-500-with-normalization/2",

                              input_shape=[], dtype=tf.string, trainable=False)
hub_embedding(train_examples_batch[:3]) # test
model = tf.keras.Sequential()

model.add(hub_embedding)

model.add(tf.keras.layers.Dense(16, activation='relu'))

model.add(tf.keras.layers.Dropout(.2))

model.add(tf.keras.layers.Dense(8, activation='relu'))

model.add(tf.keras.layers.Dense(1))



model.summary()
model.compile(optimizer='adam',

              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),

              metrics=['accuracy'])
history = model.fit(train_dataset.shuffle(10000).batch(512),

                    epochs=10,

                    validation_data=val_dataset.batch(512),

                    verbose=1)
results = model.evaluate(test_dataset.batch(512), verbose=2)



for name, value in zip(model.metrics_names, results):

  print("%s: %.3f" % (name, value))
predictions = (model.predict_classes(test_raw['tweet'])).astype(bool)

print("%d tweets out of %d labeled True" % ((predictions.sum(axis=0)), predictions.size))
# Construct a dataframe of test tweets and predicted labels

predictions_df = pd.DataFrame(predictions, columns=['label'])

predictions_df['tweet'] = test_raw['tweet']

predictions_df
# Take a look at tweet labeled True

predictions_df[predictions_df['label']]