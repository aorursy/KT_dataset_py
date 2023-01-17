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
vectorize_layer
def vectorize_text(text, label):

    text = tf.expand_dims(text, -1)

    return vectorize_layer(text), label



train_ds = raw_train_ds.map(vectorize_text)

val_ds = raw_val_ds.map(vectorize_text)

test_ds = raw_test_ds.map(vectorize_text)



train_ds = train_ds.cache().prefetch(buffer_size=10)

val_ds = val_ds.cache().prefetch(buffer_size=10)

test_ds = test_ds.cache().prefetch(buffer_size=10)
from tensorflow.keras import layers

model = tf.keras.Sequential([

    tf.keras.layers.Embedding(max_features, 64),

    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,  return_sequences=True)),

    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),

    tf.keras.layers.Dense(64, activation='relu'),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(1)

])
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),

              optimizer=tf.keras.optimizers.Adam(1e-4),

              metrics=['accuracy'])
history = model.fit(train_ds, epochs=10,

                    validation_data=val_ds,

                    validation_steps=30)
import matplotlib.pyplot as plt

acc = history.history['accuracy']

val_acc = history.history['val_accuracy']



loss=history.history['loss']

val_loss=history.history['val_loss']



#epochs_range = range(22)



plt.figure(figsize=(15, 15))

plt.subplot(1, 2, 1)

plt.plot(acc, label='Training Accuracy')

plt.plot(val_acc, label='Validation Accuracy')

plt.legend(loc='lower right')

plt.title('Training and Validation Accuracy')



plt.subplot(1, 2, 2)

plt.plot(loss, label='Training Loss')

plt.plot(val_loss, label='Validation Loss')

plt.legend(loc='upper right')

plt.title('Training and Validation Loss')

plt.show()
test_loss, test_acc = model.evaluate(test_ds)



print('Test Loss: {}'.format(test_loss))

print('Test Accuracy: {}'.format(test_acc))