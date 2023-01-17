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
df = pd.read_json('../input/arxiv-papers-2010-2020/arXiv_title_abstract_20200809_2011_2020.json')
df
df.drop(columns='year',inplace=True)
val_df = df.sample(frac=0.1, random_state=1007)
train_df = df.drop(val_df.index)
test_df = train_df.sample(frac=0.1, random_state=1007)
train_df.drop(test_df.index, inplace=True)
train_df
val_df
test_df
! pip install -q tf-nightly
import tensorflow as tf
from tensorflow import keras
tf.__version__
from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
def df_to_dataset(dataframe, target, shuffle=True, batch_size=10):
  dataframe = dataframe.copy()
  labels = dataframe.pop(target)
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds
raw_train_ds = tf.data.Dataset.from_tensor_slices((train_df.abstract.values, train_df.title.values))
raw_val_ds = tf.data.Dataset.from_tensor_slices((val_df.abstract.values, val_df.title.values))
raw_test_ds  = tf.data.Dataset.from_tensor_slices((test_df.abstract.values, test_df.title.values))
raw_train_ds
#dataset = tf.data.Dataset.from_tensor_slices((train_df.abstract.values, train_df.title.values))
#dataset
print(
    "Number of batches in raw_train_ds: %d"
    % tf.data.experimental.cardinality(raw_train_ds)
)
# TOKENIZE

train_text = train_df.abstract.to_numpy()
tok = Tokenizer(oov_token='<unk>')
tok.fit_on_texts(train_text)
tok.word_index['<pad>'] = 0
tok.index_word[0] = '<pad>'
train_seqs = tok.texts_to_sequences(train_text)
train_seqs = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')

train_labels = train_df.title.to_numpy()
train_seqs_labels = tok.texts_to_sequences(train_labels)
train_seqs_labels = tf.keras.preprocessing.sequence.pad_sequences(train_seqs_labels, padding='post')

valid_text = val_df.abstract.to_numpy()
valid_seqs = tok.texts_to_sequences(valid_text)
valid_seqs = tf.keras.preprocessing.sequence.pad_sequences(valid_seqs, padding='post')

valid_labels = val_df.title.to_numpy()
valid_seqs_labels = tok.texts_to_sequences(valid_labels)
valid_seqs_labels = tf.keras.preprocessing.sequence.pad_sequences(valid_seqs_labels, padding='post')

# CONVERT TO TF DATASETS
BUFFER_SIZE = 1024
train_ds = tf.data.Dataset.from_tensor_slices((train_seqs,train_seqs_labels))
valid_ds = tf.data.Dataset.from_tensor_slices((valid_seqs,valid_seqs_labels))

train_ds = train_ds.shuffle(BUFFER_SIZE).batch(batch_size)
valid_ds = valid_ds.batch(batch_size)

# PREFETCH

train_ds = train_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
valid_ds = valid_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
train_ds
x, y = train_df.abstract.to_numpy(), train_df.title.to_numpy()
x, y = tf.convert_to_tensor(x),tf.convert_to_tensor(y)
for text_batch, label_batch in train_ds.take(1):
    for i in range(5):
        print(text_batch.numpy()[i])
        print(label_batch.numpy()[i])
import matplotlib.pyplot as plt

def plot_graphs(history, metric):
  plt.plot(history.history[metric])
  plt.plot(history.history['val_'+metric], '')
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend([metric, 'val_'+metric])
  plt.show()
train_dataset, test_dataset = train_df, val_df
import tensorflow_datasets as tfds
vocab_size = 10000
tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    df, vocab_size)
encoder = tokenizer
print('Vocabulary size: {}'.format(encoder.vocab_size))
sample_string = 'Hello TensorFlow.'

encoded_string = encoder.encode(sample_string)
print('Encoded string is {}'.format(encoded_string))

original_string = encoder.decode(encoded_string)
print('The original string: "{}"'.format(original_string))
assert original_string == sample_string
for index in encoded_string:
  print('{} ----> {}'.format(index, encoder.decode([index])))
BUFFER_SIZE = 10000
BATCH_SIZE = 64
type(train_ds)
train_ds
#dataset_train = tf.data.Dataset.from_tensor_slices(train_df.abstract)
#dataset_val = tf.data.Dataset.from_tensor_slices(val_df.abstract)
train_dataset = train_ds.shuffle(BUFFER_SIZE)
train_dataset = train_ds.padded_batch(BATCH_SIZE)
test_dataset = valid_ds.padded_batch(BATCH_SIZE)
def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label
train_ds
# Vectorize the data.
train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

# Do async prefetching / buffering of the data for best performance on GPU.
train_ds = train_ds.cache().prefetch(buffer_size=10)
val_ds = val_ds.cache().prefetch(buffer_size=10)
test_ds = test_ds.cache().prefetch(buffer_size=10)
train_ds
from tensorflow.keras import layers

# A integer input for vocab indices.
inputs = tf.keras.Input(shape=(None,), dtype="int64")

# Next, we add a layer to map those vocab indices into a space of dimensionality
# 'embedding_dim'.
x = layers.Embedding(max_features, embedding_dim)(inputs)
x = layers.Dropout(0.5)(x)

# Conv1D + global max pooling
x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
x = layers.GlobalMaxPooling1D()(x)

# We add a vanilla hidden layer:
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.5)(x)

# We project onto a single unit output layer, and squash it with a sigmoid:
predictions = layers.Dense(1, activation="sigmoid", name="predictions")(x)

model = tf.keras.Model(inputs, predictions)

# Compile the model with binary crossentropy loss and an adam optimizer.
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
epochs = 3

# Fit the model using the train and test datasets.
model.fit(train_ds, validation_data=val_ds, epochs=epochs)
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(encoder.vocab_size, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])
model.summary()
history = model.fit(train_dataset, epochs=10,
                    validation_data=test_dataset, 
                    validation_steps=30)
# your code here 

# your code here 
# your code here 