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
vocab_size = 10000

embedding_dim = 200

max_length = int(train_df['text'].str.len().max())

trunc_type = 'post'

padding_type = 'post'

oov_tok = '<OOV>'

val_split = 0.0

batch_size = 512
tokenizer = Tokenizer(num_words=vocab_size,

                      filters='!"#$%&()*+,-./:;<=>?',

                      lower=True,

                      split=" ",

                      oov_token=oov_tok)

tokenizer.fit_on_texts(train_df['text'])

word_index = tokenizer.word_index

dict(list(word_index.items())[0:10])
pretrained_embed = True

if pretrained_embed:

    glove_file = '/kaggle/input/glove-global-vectors-for-word-representation/glove.twitter.27B.{}d.txt'.format(embedding_dim)

    emb_dict = {}

    with open(glove_file) as glove:

        for line in glove:

            values = line.split()

            word = values[0]

            emb_dict[word] = np.asarray(values[1:], dtype='float32')

        

    emb_matrix = np.zeros((vocab_size, embedding_dim))

    for word, i in tokenizer.word_index.items():

        if i < vocab_size:

            if word in emb_dict:

                emb_matrix[i] = emb_dict[word]

        else:

            break

    emb_matrix.shape
train_sequences = tokenizer.texts_to_sequences(train_df['text'])

train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

print(train_padded.shape)
def get_label(train_sequence, label_sequence):

    result_sequence = [0]*len(train_sequence)

    for i in range(len(train_sequence) - len(label_sequence) + 1):

        if train_sequence[i:i+len(label_sequence)] == label_sequence:

            result_sequence[i:i+len(label_sequence)] = [1]*len(label_sequence)

            return result_sequence

    return result_sequence
label_sequences = tokenizer.texts_to_sequences(train_df['selected_text'])

transform_labels = [get_label(train_sequences[idx], label_sequences[idx]) for idx in range(len(train_sequences))]

len(transform_labels)
label_padded = pad_sequences(transform_labels, maxlen=max_length, padding=padding_type, truncating=trunc_type)

label_padded.shape
sentiment_df = pd.get_dummies(train_df['sentiment'], drop_first=True)

sentiment_df.shape
text_input = tf.keras.Input(shape=(train_padded.shape[1],))

embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)(text_input)

lstm = tf.keras.layers.LSTM(64)(embedding)



sentiment_input = tf.keras.Input(shape=(sentiment_df.shape[1],))



combine = tf.keras.layers.concatenate([lstm, sentiment_input], axis=-1)

transform = tf.keras.layers.Dense(128)(combine)

output = tf.keras.layers.Dense(max_length, activation='sigmoid')(transform)



model = tf.keras.Model([text_input, sentiment_input], output)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
model.layers[1].set_weights([emb_matrix])

model.layers[1].trainable = False

model.summary()
if val_split > 0:

    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=3, mode='min')

    best_model = 'best_model.h5'

    mc = tf.keras.callbacks.ModelCheckpoint(best_model,

                                            monitor='val_accuracy',

                                            mode='max',

                                            save_weights_only=True,

                                            save_best_only=True)
num_epochs = 50

if val_split > 0:

    history = model.fit([train_padded, sentiment_df],

                        label_padded,

                        epochs=num_epochs,

                        batch_size=batch_size,

                        validation_split=val_split,

                        callbacks=[es, mc])

else:

    history = model.fit([train_padded, sentiment_df],

                        label_padded,

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

    plot_graphs(history, "accuracy")

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

test_sentiment_df
probabilities = model.predict([test_padded, test_sentiment_df])

probabilities
probabilities.shape
predictions = (probabilities > 0.5).astype(int)

predictions
from nltk.tokenize import word_tokenize

def extract_text_from_prediction(prediction, text):

    words = word_tokenize(text)

    selected_words = []

    for i, is_selected in enumerate(prediction):

        if is_selected == 1 and i < len(words):

            selected_words.append(words[i])

    return '"' + " ".join(selected_words).strip() + '"'

test_df["selected_text"] = [extract_text_from_prediction(predictions[i], test_df.iloc[i]["text"]) for i in range(len(predictions))]
test_df[["textID", "selected_text"]].to_csv("submission.csv", index=False)