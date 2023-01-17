import json
# Loading data from the json file to a python array

# Note that this is done only for one dataset

dataset = []

for line in open("/kaggle/input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset.json", "r"):

    dataset.append(json.loads(line))

    json.load
# Testing if the dataset loaded correctly

dataset[0]
article_link=[]

headline=[]

is_sarcastic=[]
for item in dataset: 

    article_link.append(item['article_link'])

    headline.append(item['headline'])

    is_sarcastic.append(item['is_sarcastic'])
# Checking for values in the array

print(article_link[3])

print(headline[3])

print(is_sarcastic[3])
import tensorflow as tf 

from tensorflow import keras as k

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences as pd

import numpy as np
token=Tokenizer(oov_token="<oov>")
token.fit_on_texts(headline)

word_index=token.word_index

len(word_index)
seq = token.texts_to_sequences(headline)

padded = pd(seq, padding='post')

padded[0]
print(padded.shape)
#e data we need

data_len = len(headline)

# Test size is 20% of the dataset

train_size = round((data_len * 80) / 100)

print(train_size)
# Splitting the data into train and test sets



train_headline=headline[0:train_size]

test_headline=headline[train_size:]

train_result=is_sarcastic[0:train_size]

test_result=is_sarcastic[train_size:]
# Using a different token to differenciate it from the above token which was used for illustration purposes

token2 = Tokenizer(oov_token="<OOV>")
token2.fit_on_texts(train_headline)

word_index_2 = token2.word_index



train_seq = token2.texts_to_sequences(train_headline)

train_pad = pd(train_seq)



test_seq = token2.texts_to_sequences(test_headline)

test_pad = pd(test_seq)
vocab_size = len(word_index_2) + 1

vocab_size
vocab_size = len(word_index_2) + 1

model = k.Sequential([

    k.layers.Embedding(vocab_size, 50),

    k.layers.GlobalAveragePooling1D(),

    k.layers.Dense(24, activation='relu'),

    k.layers.Dense(1, activation='sigmoid')

])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



model.summary()
train_pad = np.array(train_pad)

train_result = np.array(train_result)

test_pad = np.array(test_pad)

test_result = np.array(test_result)
type(train_pad)
training = model.fit(train_pad, train_result, epochs=30, validation_data=(test_pad, test_result), verbose=2)

# Testing predictions for dome random sentences

sentences = [

    'Meh, Kind of good',

    'Climate is perfect'

]



sequences = token2.texts_to_sequences(sentences)

latest_padded = pd(sequences)

model.predict(latest_padded)
import matplotlib.pyplot as plt





def plot_graphs(training, string):

  plt.plot(training.history[string])

  plt.plot(training.history['val_'+string])

  plt.xlabel("Epochs")

  plt.ylabel(string)

  plt.legend([string, 'val_'+string])

  plt.show()
plot_graphs(training, 'accuracy')
plot_graphs(training, 'loss')