import pandas as pd

import numpy as np



import tensorflow as tf



!pip install -q tensorflow-hub

!pip install -q tfds-nightly

import tensorflow_hub as hub

import tensorflow_datasets as tfds



print("Version: ", tf.__version__)

print("Eager mode: ", tf.executing_eagerly())

print("Hub version: ", hub.__version__)

print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")
train_set = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test_set = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

train_set = train_set.drop(['id', 'keyword', 'location'], axis=1)

test_set = test_set.drop(['id', 'keyword', 'location'], axis=1)

train_set
f=0

r=0

for i in range(0, len(train_set)):

    if train_set.target[i] == 0:

        f+=1

    else:

        r+=1

print(100*f/len(train_set), "% fake in train set")

print(100*r/len(train_set), "% real in train set")
# len_train_set = len(train_set)

train_set = tf.data.Dataset.from_tensor_slices((train_set.text.values, train_set.target.values))

test_set = tf.data.Dataset.from_tensor_slices((test_set.text.values))

# validation_data = train_set.shard(num_shards=int(len_train_set / 10), index=0)

# train_set = train_set.shard(num_shards=int(9 * len_train_set / 10), index=1)
model = tf.keras.Sequential()

model.add(hub.KerasLayer("https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1", input_shape=[], dtype=tf.string, trainable=True))

model.add(tf.keras.layers.Dense(64, activation='relu'))

model.add(tf.keras.layers.Dense(32, activation='relu'))

model.add(tf.keras.layers.Dense(16, activation='relu'))

model.add(tf.keras.layers.Dense(1, activation='sigmoid'))



model.summary()
# with tf.device('/GPU:0'):

model.compile(optimizer='adam',

              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),

              metrics=['accuracy'])

history = model.fit(train_set.shuffle(7613).batch(128),

                    epochs=10)#,

#                     validation_data=validation_data.batch(512))
results = model.predict(test_set.batch(512))

results = np.round(results)
perfect_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

perfect_submission['target'] = results

perfect_submission.to_csv('perfect_submission.csv', index=False)

perfect_submission.describe()