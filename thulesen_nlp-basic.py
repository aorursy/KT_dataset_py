import numpy as np

import pandas as pd

import tensorflow as tf

import tensorflow_hub as hub

from tensorflow.keras.models import Sequential, Model

from tensorflow.keras import Input

from tensorflow.keras.layers import (InputLayer, Dense, Flatten, GlobalMaxPooling1D, GlobalAveragePooling1D, 

                                     SimpleRNN, LSTM, 

                                     Bidirectional, Lambda, Embedding, Dropout, Conv1D, MaxPooling1D)

from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

from tqdm.keras import TqdmCallback # progress bars

import re

import matplotlib.pyplot as plt
# DATA LOADING

df_train = pd.read_csv('../input/nlp-getting-started/train.csv')

df_test = pd.read_csv('../input/nlp-getting-started/test.csv')



# concatenate the dataframes so we can do some data analysis

df = pd.concat([df_train, df_test], keys=['train', 'test'])

df.text = df.text.astype("string")
vocab_size = 30000

sequence_length = 30
regex_nonalpha = re.compile('[^a-zA-Z0-9\s]')

regex_http = re.compile("http\S*")

regex_apostrophe = re.compile("'")

regex_space = re.compile("\s\s+")

def alphanumeric(text):

    text = regex_http.sub("", text)

    text = regex_apostrophe.sub("", text)

    text = regex_nonalpha.sub(" ", text)

    text = regex_space.sub(" ", text)

    return text.lower().strip()
def alphanumeric_tf(text_tensor):

    text_tensor = tf.strings.regex_replace(text_tensor, "http\S*", "")

    text_tensor = tf.strings.regex_replace(text_tensor, "'", "")

    text_tensor = tf.strings.regex_replace(text_tensor, "[^a-zA-Z0-9\s]", " ")

    text_tensor = tf.strings.regex_replace(text_tensor, "\s\s+", " ")

    return tf.strings.strip(tf.strings.lower(text_tensor))
text_vectorizer = TextVectorization(max_tokens=vocab_size,

                                    standardize=alphanumeric_tf,

                                    output_sequence_length=sequence_length,

                                   )

text_vectorizer.adapt(df.text.to_numpy())

vocab = text_vectorizer.get_vocabulary()
df.iloc[:5].text
text_vectorizer(df.iloc[:5].text.to_numpy())
X_train = text_vectorizer(df_train.text.to_numpy())

y = df_train.target.to_numpy()

X_test = text_vectorizer(df_test.text.to_numpy())



ds_train = tf.data.Dataset.from_tensor_slices((X_train, y))

ds_test = tf.data.Dataset.from_tensor_slices((X_test))
# SPLIT TRAIN INTO TRAIN/DEV



ds = ds_train.shuffle(buffer_size=10000).cache().enumerate()

ds_train = ds.filter(lambda i, data: i % 10 <= 7).map(lambda _, data: data).cache()

ds_dev = ds.filter(lambda i, data: i % 10 > 7).map(lambda _, data: data).cache()
# BATCHING

BATCH_SIZE = 1024



ds_train = ds_train.batch(BATCH_SIZE)

ds_dev = ds_dev.batch(BATCH_SIZE)

ds_test = ds_test.batch(BATCH_SIZE)
embedding_dim = 64

#embedding_http = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"

#embedding_layer = hub.KerasLayer(embedding_http, output_shape=[embedding_dim], input_shape=[], 

#                           dtype=tf.string, trainable=True)

embedding_layer = tf.keras.layers.Embedding(vocab_size, 

                                            embedding_dim, 

                                            input_length=sequence_length,

                                            mask_zero=True,

                                           )
# very simple pooling model

model = Sequential([

    embedding_layer,

    GlobalAveragePooling1D(),

    Dense(16, activation='relu'),

    Dense(1),

])

model.summary()

model.compile(optimizer='Adam',

              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),

              metrics=['accuracy'],

             )

history = model.fit(ds_train, epochs=100, validation_data=ds_dev, verbose=0,

                    callbacks=[TqdmCallback(verbose=1)]

                   ).history

plt.plot(history['accuracy'])

plt.plot(history['val_accuracy'])
# RNN model with masking

model = Sequential([

    embedding_layer,

    SimpleRNN(64),

    Dense(64, activation='relu'),

    Dropout(0.5),

    Dense(1),

])

model.summary()

model.compile(optimizer='Adam',

              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),

              metrics=['accuracy'],

             )

history = model.fit(ds_train, epochs=100, validation_data=ds_dev, verbose=0,

                    callbacks=[TqdmCallback(verbose=1)]

                   ).history

plt.plot(history['accuracy'])

plt.plot(history['val_accuracy'])
!wget http://nlp.stanford.edu/data/glove.6B.zip

!unzip -q glove.6B.zip
word_index = dict(zip(vocab, range(len(vocab))))

path_to_glove_file = "./glove.6B.100d.txt"



embeddings_index = {}

with open(path_to_glove_file) as f:

    for line in f:

        word, coefs = line.split(maxsplit=1)

        coefs = np.fromstring(coefs, "f", sep=" ")

        embeddings_index[word] = coefs



print("Found %s word vectors." % len(embeddings_index))
num_tokens = len(vocab) + 2

embedding_dim = 100

hits = 0

misses = 0



# Prepare embedding matrix

embedding_matrix = np.zeros((num_tokens, embedding_dim))

for word, i in word_index.items():

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        # Words not found in embedding index will be all-zeros.

        # This includes the representation for "padding" and "OOV"

        embedding_matrix[i] = embedding_vector

        hits += 1

    else:

        misses += 1

print("Converted %d words (%d misses)" % (hits, misses))
embedding_layer = Embedding(

    num_tokens,

    embedding_dim,

    embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),

    trainable=False,

)
# very simple pooling model

model = Sequential([

    embedding_layer,

    GlobalAveragePooling1D(),

    Dense(16, activation='relu'),

    Dense(1),

])

model.summary()

model.compile(optimizer='Adam',

              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),

              metrics=['accuracy'],

             )

history = model.fit(ds_train, epochs=100, validation_data=ds_dev, verbose=0,

                    callbacks=[TqdmCallback(verbose=1)]

                   ).history

plt.plot(history['accuracy'])

plt.plot(history['val_accuracy'])
# RNN model with masking

model = Sequential([

    embedding_layer,

    SimpleRNN(64),

    Dense(64, activation='relu'),

    Dropout(0.5),

    Dense(1),

])

model.summary()

model.compile(optimizer='Adam',

              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),

              metrics=['accuracy'],

             )

history = model.fit(ds_train, epochs=100, validation_data=ds_dev, verbose=0,

                    callbacks=[TqdmCallback(verbose=1)]

                   ).history

plt.plot(history['accuracy'])

plt.plot(history['val_accuracy'])
# model from https://keras.io/examples/nlp/pretrained_word_embeddings/#build-the-model



model = Sequential([

    embedding_layer,

    Conv1D(128, 2, activation="relu"),

    MaxPooling1D(2),

    #Conv1D(128, 2, activation="relu"),

    #MaxPooling1D(2),

    Conv1D(128, 5, activation="relu"),

    GlobalMaxPooling1D(),

    Dense(128, activation="relu"),

    Dropout(0.5),

    Dense(1)

])

model.summary()



model.compile(optimizer='Adam',

              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),

              metrics=['accuracy'],

             )

history = model.fit(ds_train, epochs=100, validation_data=ds_dev, verbose=0,

                    callbacks=[TqdmCallback(verbose=1)]

                   ).history

plt.plot(history['accuracy'])

plt.plot(history['val_accuracy'])
model.fit(ds_train.concatenate(ds_dev), epochs=100, verbose=0, callbacks=[TqdmCallback(verbose=1)])

predictions = model.predict(ds_test)
predictions = (tf.sigmoid(model.predict(ds_test)).numpy() > 0.5).astype(int).reshape((-1,))
submission = pd.DataFrame({

    'id': df_test.id.to_numpy(),

    'target': predictions,

})
submission.to_csv('submission.csv', index=False)