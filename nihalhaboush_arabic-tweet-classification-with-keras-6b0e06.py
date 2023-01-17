# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data_dir = '../input/arabic-sentiment-twitter-corpus/arabic_tweets'
! pip install tf-nightly 
import tensorflow as tf 

tf.__version__
batch_size = 32

raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(

    data_dir,

    batch_size=batch_size,

    validation_split=0.2,

    subset="training",

    seed=1337,

)

raw_val_ds = tf.keras.preprocessing.text_dataset_from_directory(

    data_dir,

    batch_size=batch_size,

    validation_split=0.2,

    subset="validation",

    seed=1337,

)
raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory(

    data_dir, batch_size=batch_size

)



print(

    "Number of batches in raw_train_ds: %d"

    % tf.data.experimental.cardinality(raw_train_ds)

)

print(

    "Number of batches in raw_val_ds: %d" % tf.data.experimental.cardinality(raw_val_ds)

)

print(

    "Number of batches in raw_test_ds: %d"

    % tf.data.experimental.cardinality(raw_test_ds)

)
for text_batch, label_batch in raw_train_ds.take(1):

    for i in range(5):

        print(text_batch.numpy()[i].decode('utf-8').strip())

        print(label_batch.numpy()[i])

        print('--------------------------------')
BUFFER_SIZE = 10000

BATCH_SIZE = 64
train_dataset = train_dataset.shuffle(BUFFER_SIZE)

train_dataset = train_dataset.padded_batch(BATCH_SIZE)



test_dataset = test_dataset.padded_batch(BATCH_SIZE)
import tensorflow_datasets as tfds

import tensorflow as tf
import matplotlib.pyplot as plt



def plot_graphs(history, metric):

  plt.plot(history.history[metric])

  plt.plot(history.history['val_'+metric], '')

  plt.xlabel("Epochs")

  plt.ylabel(metric)

  plt.legend([metric, 'val_'+metric])

  plt.show()
model = tf.keras.Sequential([

    tf.keras.layers.Embedding(encoder.vocab_size, 64),

    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),

    tf.keras.layers.Dense(64, activation='relu'),

    tf.keras.layers.Dense(1)

])
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),

              optimizer=tf.keras.optimizers.Adam(1e-4),

              metrics=['accuracy'])
history = model.fit(train_dataset, epochs=20,

                    validation_data=test_dataset, 

                    validation_steps=50)
test_loss, test_acc = model.evaluate(test_dataset)



print('Test Loss: {}'.format(test_loss))

print('Test Accuracy: {}'.format(test_acc))
def pad_to_size(vec, size):

  zeros = [0] * (size - len(vec))

  vec.extend(zeros)

  return vec
def sample_predict(sample_pred_text, pad):

  encoded_sample_pred_text = encoder.encode(sample_pred_text)



  if pad:

    encoded_sample_pred_text = pad_to_size(encoded_sample_pred_text, 64)

  encoded_sample_pred_text = tf.cast(encoded_sample_pred_text, tf.float32)

  predictions = model.predict(tf.expand_dims(encoded_sample_pred_text, 0))



  return (predictions)
# predict on a sample text without padding.



sample_pred_text = ('The movie was cool. The animation and the graphics '

                    'were out of this world. I would recommend this movie.')

predictions = sample_predict(sample_pred_text, pad=False)

print(predictions)
# predict on a sample text with padding



sample_pred_text = ('The movie was cool. The animation and the graphics '

                    'were out of this world. I would recommend this movie.')

predictions = sample_predict(sample_pred_text, pad=True)

print(predictions)
plot_graphs(history, 'accuracy')