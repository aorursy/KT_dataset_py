import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem.snowball import SnowballStemmer
stop_words = set(stopwords.words('english'))

import tensorflow
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers.embeddings import Embedding

import re

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from sklearn.metrics import accuracy_score
def clean_text(sent):
    sent = sent.lower().split()
    #print (sent), (type(sent))
    words_ = [word for word in sent if word not in stop_words]
    sent = " ".join(words_)
    sent = re.split(r"\/|\:|\(|\)|\?|\,|\.| |\n|!|;", sent)
    #print (sent), (type(sent))
    sent = list(filter(None, sent))
    #print (sent), (type(sent))
    stemmed_words = [stemmer.stem(word) for word in sent]
    sent = " ".join(stemmed_words)
    #print (sent), (type(sent))
    return sent
stemmer = PorterStemmer()

train_data = pd.read_csv('../input/train-film-rating/train_film_rating.csv')
train_data['text'] = train_data['text'].apply(lambda x: clean_text(x))

train_data.sample(10)
vocabulary_size = 20000
tokenizer = Tokenizer(num_words= vocabulary_size)
tokenizer.fit_on_texts(train_data['text'])

sequences = tokenizer.texts_to_sequences(train_data['text'])
data = pad_sequences(sequences, maxlen=50)
labels = train_data['rating_01']
model = Sequential()
model.add(Embedding(20000, 100, input_length=50))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(data, np.array(labels), validation_split=0.4, epochs=3)

predict = model.predict(data)
predict = predict * 100 // 50

print (accuracy_score(predict, labels))
test_data = pd.read_csv('../input/test-film-rating/test_film_rating.csv')
test_data['text'] = test_data['text'].apply(lambda x: clean_text(x))

test_data.sample(10)
#tokenizer.fit_on_texts(test_data['text'])
sequences = tokenizer.texts_to_sequences(test_data['text'])
data_test = pad_sequences(sequences, maxlen=50)
labels_test = test_data['rating_01']
predict_test = model.predict(data_test)
predict_test = predict_test * 100 // 50

print (accuracy_score(predict_test, labels_test))
unsup_data = pd.read_csv('../input/unsup-film-rating/unsup_film_rating.csv')
unsup_data['text'] = unsup_data['text'].apply(lambda x: clean_text(x))

unsup_data.sample(10)
sequences = tokenizer.texts_to_sequences(unsup_data['text'])
data_unsup = pad_sequences(sequences, maxlen=50)
predict_test = model.predict(data_unsup)
predict_test = predict_test * 100 // 50

print (predict_test)