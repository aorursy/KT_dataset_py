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
train_data = pd.read_csv('../input/nlp-getting-started/train.csv')

test_data = pd.read_csv('../input/nlp-getting-started/test.csv')
train_data.head()
test_data.head()
slicing_size = int(7612*0.6)

val_data = train_data[slicing_size:]

train_data = train_data[:slicing_size]
val_data
train_data
import tensorflow as tf
!pip install -q tensorflow-hub

import tensorflow_hub as hub
target = train_data.pop('target')

target
tweets = train_data.pop('text')

tweets
dataset = tf.data.Dataset.from_tensor_slices((tweets.values, target.values))

dataset
val_tweets = val_data.pop('text')

val_target = val_data.pop('target')
val_tweets
val_target
val_dataset = tf.data.Dataset.from_tensor_slices((val_tweets.values, val_target.values))

val_dataset
test_tweets = test_data.pop('text')

test_tweets
test_dataset = tf.data.Dataset.from_tensor_slices((test_tweets.values))

test_dataset
embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"

hub_layer = hub.KerasLayer(embedding, input_shape=[], 

                           dtype=tf.string, trainable=True)

model = tf.keras.Sequential()

model.add(hub_layer)

model.add(tf.keras.layers.Dense(16,activation='relu'))

model.add(tf.keras.layers.Dense(1,activation='sigmoid'))

model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(dataset.shuffle(len(train_data)).batch(512),

                    epochs=20,

                    validation_data = val_dataset.batch(512),

                     verbose=1)
import matplotlib.pyplot as plt

def plotting(history, metrics):

    plt.plot(history.history[metrics])

    plt.plot(history.history["val_"+ metrics])

    plt.xlabel("Epochs")

    plt.ylabel(metrics)

    plt.legend([metrics, "val_" + metrics])

    plt.show()
plotting(history, "accuracy")
plotting(history, "loss")
results = model.predict(test_tweets)

for name, value in zip(model.metrics_names, results):

  print("%s: %.3f" % (name, value))
results
output = []

for value in results:

    if value > 0.5:

        output.append(1)

    else:

        output.append(0)
output
submission = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')
submission
submission['target'] = output

submission.to_csv("output.csv", index=False)

submission.head()