import tensorflow as tf
device_name = tf.test.gpu_device_name()
import string
import requests
import pandas as pd

from PIL import Image
import re
# For handling string
import io
import seaborn as sns
from io import BytesIO
import random
import matplotlib.pyplot as plt 
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
file = open('/kaggle/input/milagreok/MilagresLimpo.txt','r')
file.seek(0)
data = file.read().splitlines()
len(data)
type(data)
dados = " ".join(data)
len(dados)
type(dados)
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
word_cloud = WordCloud(width = 1000,
                       height = 800,
                       colormap = 'Blues', 
                       margin = 0,
                       max_words = 200,  
                       min_word_length = 4,
                       max_font_size = 120, min_font_size = 15,  
                       background_color = "white").generate(dados)

plt.figure(figsize = (10, 15))
plt.imshow(word_cloud, interpolation = "gaussian")
plt.axis("off")
plt.show()
from nltk.corpus import stopwords
#stopwords.words('portuguese')
def get_top_ngram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(n, n)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) 
                  for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:10]
from sklearn.feature_extraction.text import CountVectorizer
top_bigrams = get_top_ngram(data,2)[:10]
x,y = map(list,zip(*top_bigrams))
sns.barplot(x = y,y = x)
# Visualising the most frequent trigrams occurring in the conversation
from sklearn.feature_extraction.text import CountVectorizer
top_trigrams = get_top_ngram(data
                             ,3)[:10]
x,y = map(list,zip(*top_trigrams))
sns.barplot(x = y,y = x)
token = Tokenizer()
token.fit_on_texts(data)
token.word_counts
token.word_index
encoded_text = token.texts_to_sequences(data)
encoded_text
x = ['o sentido da vida']
token.texts_to_sequences(x)
vocab_size = len(token.word_counts) + 1
datalist = []
for d in encoded_text:
  if len(d)>1:
    for i in range(2, len(d)):
      datalist.append(d[:i])
      print(d[:i])
max_length = 20
sequences = pad_sequences(datalist, maxlen=max_length, padding='pre')
sequences
X = sequences[:, :-1]
y = sequences[:, -1]
vocab_size
y = to_categorical(y, num_classes=vocab_size)
y
seq_length = X.shape[1]
seq_length
import tensorflow as tf
device_name = tf.test.gpu_device_name()


model = Sequential()

model.add(Embedding(vocab_size, 50, input_length=seq_length))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100))
model.add(Dense(100, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, batch_size=32, epochs=100)
poetry_length = 10


def generate_poetry(seed_text, n_lines):

  for i in range(n_lines):
    text = []
    for _ in range(poetry_length):
      encoded = token.texts_to_sequences([seed_text])
      encoded = pad_sequences(encoded, maxlen=seq_length, padding='pre')

      y_pred = np.argmax(model.predict(encoded), axis=-1)

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
    

seed_text = 'saúde'
generate_poetry(seed_text, 5)