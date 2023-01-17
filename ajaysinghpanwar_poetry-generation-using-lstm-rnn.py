import tensorflow as tf
import string
import requests
import pandas as pd
# Getting data using requests module
response = requests.get('https://raw.githubusercontent.com/laxmimerit/poetry-data/master/adele.txt')
response
# Checking the text we got
response.text
# Let's separate the text we got line by line
data = response.text.splitlines()
data
len(data)
# Join data by space and check it's length
len(" ".join(data))
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
# Tokenizing the text
token = Tokenizer()
token.fit_on_texts(data)
# token.word_counts
# Check help on tokenizer class
# help(token)
# Let's check indexes of the tokenized word
token.word_index
# Encoding the text to making it suitable to feed to our LSTM model
encoded_text = token.texts_to_sequences(data)
# Checking encoded text
encoded_text
x = ['i love you']
token.texts_to_sequences(x)
# Checking the length of the vocabulary
vocab_size = len(token.word_counts) + 1

print(vocab_size)
datalist = []
for d in encoded_text:
  if len(d)>1:
    for i in range(2, len(d)):
      datalist.append(d[:i])
      print(d[:i])
max_length = 20
sequences = pad_sequences(datalist, maxlen=max_length, padding='pre')

print(sequences)
X = sequences[:, :-1]
y = sequences[:, -1]
# Encoding y to categorical form
y = to_categorical(y, num_classes = vocab_size)

print(y)
# Let's check the sequence length
seq_length = X.shape[1]

print(seq_length)
# Creating the model
model = Sequential()
model.add(Embedding(vocab_size, 50, input_length = seq_length))
model.add(LSTM(100, return_sequences = True))
model.add(LSTM(100))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(vocab_size, activation = 'softmax'))
model.summary()
# Compiling the model
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
# Fitting the model
model.fit(X, y, batch_size = 32, epochs = 100)
poetry_length = 10

def generate_poetry(seed_text, n_lines):

  for i in range(n_lines):
    text = []
    for _ in range(poetry_length):
      encoded = token.texts_to_sequences([seed_text])
      encoded = pad_sequences(encoded, maxlen = seq_length, padding = 'pre')

      y_pred = np.argmax(model.predict(encoded), axis = -1)

      predicted_word = ""
      for word, index in token.word_index.items():
        if index == y_pred:
          predicted_word = word
          break

      seed_text = seed_text + ' ' + predicted_word
      text.append(predicted_word)

    seed_text = text[-1]
    text = ' '.join(text)
    print(text)



seed_text = 'i love you'
generate_poetry(seed_text, 5)
seed_text = 'god'
generate_poetry(seed_text, 5)