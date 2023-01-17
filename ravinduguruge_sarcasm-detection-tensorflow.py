import json
# Loading data from the json file to a python array
# Note that this is done only for one dataset
dataset = []
for line in open("/kaggle/input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset.json", "r"):
    dataset.append(json.loads(line))
    json.load
# Testing if the dataset loaded correctly
dataset[0]
# Creating arrays for the keys of the json file
article_link = []
headline = []
is_sarcastic = []
# Append values from the json file to the relavant array
for item in dataset:
    article_link.append(item['article_link'])
    headline.append(item['headline'])
    is_sarcastic.append(item['is_sarcastic'])
# Checking for values in the array
print(article_link[0])
print(headline[0])
print(is_sarcastic[0])
# Importing tensorflow and related libraries
import tensorflow as tf
from tensorflow import keras as k
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences as pd
import numpy as np
token = Tokenizer(oov_token="<OOV>")
token.fit_on_texts(headline)
# Calculating the number of words indexed
word_index = token.word_index
len(word_index)
# An illustration of using pad_sequences
seq = token.texts_to_sequences(headline)
padded = pd(seq, padding='post')
padded[0]
print(padded.shape)
# Since we have to check for sarcasm in headlines, let's consider it to be the data we need
data_len = len(headline)
data_len
# Train size is 80% of the dataset
# Test size is 20% of the dataset
train_size = round((data_len * 80) / 100)
print(train_size)
# Splitting the data into train and test sets
train_headline = headline[0:train_size]
test_headline = headline[train_size:]

train_result = is_sarcastic[0:train_size]
test_result = is_sarcastic[train_size:]
# Using a different token to differenciate it from the above token which was used for illustration purposes
token2 = Tokenizer(oov_token="<OOV>")
token2.fit_on_texts(train_headline)
word_index_2 = token2.word_index

train_seq = token2.texts_to_sequences(train_headline)
train_pad = pd(train_seq)

test_seq = token2.texts_to_sequences(test_headline)
test_pad = pd(test_seq)
vocab_size = len(word_index_2) + 1
model = k.Sequential([
    k.layers.Embedding(vocab_size, 50),
    k.layers.GlobalAveragePooling1D(),
    k.layers.Dense(24, activation='relu'),
    k.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# If this step is ignored, you might encounter and error as follows:
#  Failed to find data adapter that can handle input: <class 'numpy.ndarray'>, (<class 'list'> containing values of types {"<class 'int'>"})
train_pad = np.array(train_pad)
train_result = np.array(train_result)
test_pad = np.array(test_pad)
test_result = np.array(test_result)
# Training the model with 30 epochs
training = model.fit(train_pad, train_result, epochs=30, validation_data=(test_pad, test_result), verbose=2)
# Testing predictions for dome random sentences
sentences = [
    'Meh, Kind of good',
    'Climate is perfect'
]

sequences = token2.texts_to_sequences(sentences)
latest_padded = pd(sequences)
model.predict(latest_padded)