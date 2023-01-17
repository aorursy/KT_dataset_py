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



import tensorflow as tf

import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences
train_df = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')

train_df['text'] = train_df['text'].str.strip().astype(str)

train_df['selected_text'] = train_df['selected_text'].str.strip().astype(str)
vocab_size = 5000

embedding_dim = 64

max_length = int(train_df['text'].str.len().max())

trunc_type = 'post'

padding_type = 'post'

oov_tok = '<OOV>'

val_split = 0.0

batch_size = 128
tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)

tokenizer.fit_on_texts(train_df['text'])

word_index = tokenizer.word_index

dict(list(word_index.items())[0:10])
train_sequences = tokenizer.texts_to_sequences(train_df['text'])

train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

print(train_padded.shape)
def get_label(train_sequence, label_sequence):

    start_index = 0

    for i in range(len(train_sequence) - len(label_sequence) + 1):

        if train_sequence[i:i+len(label_sequence)] == label_sequence:

            start_index = i

            break

    return [start_index, len(label_sequence)]
label_sequences = tokenizer.texts_to_sequences(train_df['selected_text'])

transform_labels = [get_label(train_sequences[idx], label_sequences[idx]) for idx in range(len(train_sequences))]

start_labels = np.array([label[0] for label in transform_labels])

length_labels = np.array([label[1] for label in transform_labels])

length_labels.shape
sentiment_df = pd.get_dummies(train_df['sentiment'], drop_first=True)

sentiment_df.shape
text_input = tf.keras.Input(shape=(train_padded.shape[1],))

embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)(text_input)

lstm = tf.keras.layers.LSTM(embedding_dim)(embedding)



sentiment_input = tf.keras.Input(shape=(sentiment_df.shape[1],))



combine = tf.keras.layers.concatenate([lstm, sentiment_input], axis=-1)

start_output = tf.keras.layers.Dense(1, activation=None)(combine)

length_output = tf.keras.layers.Dense(1, activation=None)(combine)



model = tf.keras.Model([text_input, sentiment_input], [start_output, length_output])

model.compile(loss=['mean_squared_error', 'mean_squared_error'], optimizer='adam')

model.summary()
if val_split > 0:

    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=3, mode='min')

    best_model = 'best_model.h5'

    mc = tf.keras.callbacks.ModelCheckpoint(best_model,

                                            monitor='val_loss',

                                            mode='min',

                                            save_weights_only=True,

                                            save_best_only=True)
num_epochs = 100

if val_split > 0:

    history = model.fit([train_padded, sentiment_df],

                        [start_labels, length_labels],

                        epochs=num_epochs,

                        batch_size=batch_size,

                        validation_split=val_split,

                        callbacks=[es, mc])

else:

    history = model.fit([train_padded, sentiment_df],

                        [start_labels, length_labels],

                        epochs=num_epochs,

                        batch_size=batch_size,

                        validation_split=val_split)
import matplotlib.pyplot as plt

%matplotlib inline



def plot_graphs(history, string):

  plt.plot(history.history[string])

  plt.plot(history.history['val_'+string])

  plt.xlabel("Epochs")

  plt.ylabel(string)

  plt.legend([string, 'val_'+string])

  plt.show()

  

if val_split > 0:

    plot_graphs(history, "loss")
if val_split > 0:

    model.load_weights(best_model)
test_df = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')

test_df['text'] = test_df['text'].str.strip().astype(str)

test_sequences = tokenizer.texts_to_sequences(test_df['text'])

test_sequences
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

print(test_padded.shape)
test_sentiment_df = pd.get_dummies(test_df['sentiment'], drop_first=True)
predictions = model.predict([test_padded, test_sentiment_df])

start_predictions, length_predictions = predictions

start_predictions = start_predictions.reshape(-1, len(start_predictions))[0]

length_predictions = length_predictions.reshape(-1, len(length_predictions))[0]
from nltk.tokenize import word_tokenize

def extract_text_from_prediction(start_prediction, length_prediction, text):

    words = word_tokenize(text)

    start_index = int(round(start_prediction, 0))

    length = int(round(length_prediction, 0))

    selected_words = words[start_index: start_index + length]

    return '"' + " ".join(selected_words).strip() + '"'

test_df["selected_text"] = [extract_text_from_prediction(start_predictions[i], length_predictions[i], test_df.iloc[i]["text"]) for i in range(len(start_predictions))]
test_df[["textID", "selected_text"]].to_csv("submission.csv", index=False)