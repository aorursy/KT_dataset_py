# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow_datasets as tfds

import tensorflow as tf

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data_dir = '../input/arabic-sentiment-twitter-corpus/arabic_tweets'
!pip install tf-nightly
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
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

import string

import re



def custom_standardization(input_data):

    lowercase = tf.strings.lower(input_data)

    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")

    return tf.strings.regex_replace(

        stripped_html, "[%s]" % re.escape(string.punctuation), ""

    )



max_features = 20000

embedding_dim = 128

sequence_length = 500



vectorize_layer = TextVectorization(

    standardize=custom_standardization,

    max_tokens=max_features,

    output_mode="int",

    output_sequence_length=sequence_length,

)



text_ds = raw_train_ds.map(lambda x, y: x)

vectorize_layer.adapt(text_ds)
def vectorize_text(text, label):

    text = tf.expand_dims(text, -1)

    return vectorize_layer(text), label





# Vectorize the data.

train_ds = raw_train_ds.map(vectorize_text)

val_ds = raw_val_ds.map(vectorize_text)

test_ds = raw_test_ds.map(vectorize_text)



# Do async prefetching / buffering of the data for best performance on GPU.

train_ds = train_ds.cache().prefetch(buffer_size=10)

val_ds = val_ds.cache().prefetch(buffer_size=10)

test_ds = test_ds.cache().prefetch(buffer_size=10)
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