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
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization



# Model constants.

max_features = 5000

embedding_dim = 64

sequence_length = 500



# Now that we have our custom standardization, we can instantiate our text

# vectorization layer. We are using this layer to normalize, split, and map

# strings to integers, so we set our 'output_mode' to 'int'.

# Note that we're using the default split function,

# and the custom standardization defined above.

# We also set an explicit maximum sequence length, since the CNNs later in our

# model won't support ragged sequences.

vectorize_layer = TextVectorization(

    max_tokens=max_features,

    output_sequence_length=sequence_length,

)



# Now that the vocab layer has been created, call `adapt` on a text-only

# dataset to create the vocabulary. You don't have to batch, but for very large

# datasets this means you're not keeping spare copies of the dataset in memory.



# Let's make a text-only dataset (no labels):

text_ds = raw_train_ds.map(lambda x, y: x)

# Let's call `adapt`:

vectorize_layer.adapt(text_ds)
def vectorize_text(text, label):

    text = tf.expand_dims(text, -1)

    return vectorize_layer(text), label





# Vectorize the data.

train_ds = raw_train_ds.map(vectorize_text)

val_ds = raw_val_ds.map(vectorize_text)

test_ds = raw_test_ds.map(vectorize_text)
model = tf.keras.Sequential([

    tf.keras.layers.Embedding(5000, 64),

    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32,return_sequences=True)),

    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),

    tf.keras.layers.Dense(64,activation = 'relu'),

    tf.keras.layers.Dense(1)

])
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=7)