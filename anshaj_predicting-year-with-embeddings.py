import tensorflow as tf

import numpy as np 

import pandas as pd

import tensorflow as tf

import os

import plotly.express as px

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split

from nltk.corpus import stopwords



device='cuda:0'

print('Tensorflow Version:', tf.__version__)
data = pd.read_csv('/kaggle/input/news-about-major-cryptocurrencies-20132018-40k/crypto_news_parsed_2013-2018_40k.csv')

print(data.info())

data.head()
for col in data.columns:

    print(f'The unique values in {col}:', data[col].nunique())
year_dist = data['year'].value_counts()

px.bar(x=year_dist.index, y = year_dist, title = 'Distribution of years in the dataset', 

       labels = {'x' : 'year', 'y' : 'rows in dataset'})
X = data['text'].astype('str')

y = data['year']

X.shape, y.shape
stopwords = set(stopwords.words('english'))

X = X.apply(lambda x: ' '.join([x for x in x.split() if x not in stopwords]))



# Replace the years with new encoded numbers

year_dict = {2013 : 0, 2014: 1, 2015: 2, 2016 : 3, 2017 : 4, 2018 : 5}

y = y.replace(year_dict)

y.value_counts()
# Creating the train and the test set

training_sentences, testing_sentences, training_labels, testing_labels = train_test_split(X, y, test_size=0.2, stratify = y) # Stratify with y to have enough of each class in the training set

training_sentences = training_sentences.tolist()

testing_sentences = testing_sentences.tolist()
vocab_size = 300000

embedding_dim = 32

max_length = 500

trunc_type='post'

oov_tok = "<OOV>"
training_sentences[0], testing_sentences[0]
# Create a tokenizer and prepare the train and test set

tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)

tokenizer.fit_on_texts(training_sentences)

word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(training_sentences)

padded = pad_sequences(sequences,maxlen=max_length, truncating=trunc_type)



testing_sequences = tokenizer.texts_to_sequences(testing_sentences)

testing_padded = pad_sequences(testing_sequences,maxlen=max_length)
model = tf.keras.Sequential([

    tf.keras.layers.Embedding(vocab_size, 32),

    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128)),

    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.Dense(6, activation='softmax')

])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
num_epochs = 10

history = model.fit(padded, training_labels, epochs=num_epochs, validation_data=(testing_padded, testing_labels))
import matplotlib.pyplot as plt





def plot_graphs(history, string):

  plt.plot(history.history[string])

  plt.plot(history.history['val_'+string])

  plt.xlabel("Epochs")

  plt.ylabel(string)

  plt.legend([string, 'val_'+string])

  plt.show()
plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')