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

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import csv
#file: list of words found in the reviews, one per line
data_path = '/kaggle/input/fake-news/test.csv'
with open(data_path) as f:
  x_test = pd.read_csv(data_path)
data_path = '/kaggle/input/fake-news/train.csv'
with open(data_path) as f:
  train = pd.read_csv(data_path)

data_path = '/kaggle/input/fake-news/submit.csv'
with open(data_path) as f:
  y_test = pd.read_csv(data_path)
x_test = x_test.drop('id', 1)
y_test = y_test.drop('id', 1)
x_test = x_test.drop('author', 1)
train = train.drop('author', 1)
train = train.drop('id', 1)
x_test = x_test.fillna('')
train = train.fillna('')
y_test = y_test.fillna('')
test = x_test
test['label'] = y_test.values
totaldata = pd.concat([train, test])
import nltk
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
stop_words = set(stopwords.words('english'))
def clean(text):
    # Lowering letters
    text = text.lower()
    
    # Removing html tags
    text = re.sub(r'<[^>]*>', '', text)
    
    # Removing twitter usernames
    text = re.sub(r'@[A-Za-z0-9]+','',text)
    
    # Removing urls
    text = re.sub('https?://[A-Za-z0-9]','',text)
    
    # Removing numbers
    text = re.sub('[^a-zA-Z]',' ',text)
    
    word_tokens = word_tokenize(text)
    
    filtered_sentence = []
    for word_token in word_tokens:
        if word_token not in stop_words:
            filtered_sentence.append(word_token)
    
    # Joining words
    text = (' '.join(filtered_sentence))
    return text

totaldata['title'] = train['title'].apply(clean)
totaldata['text'] = train['text'].apply(clean)
y = totaldata['label'].values
totaldata = totaldata.drop('label', 1)
totaldata['title_and_text'] = totaldata['title'] + " " + totaldata['text']

totaldata = totaldata.drop('title', 1)
totaldata = totaldata.drop('text', 1)
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(totaldata['title_and_text'].values)
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, test_size=0.2, random_state=11)
x_test, x_val, y_test, y_val = train_test_split(x, y, shuffle=True, test_size=0.5, random_state=11)
clf = MultinomialNB()
clf.fit(x_train, y_train)
print(clf.score(x_train, y_train))
print(clf.score(x_test, y_test))

from tensorflow.keras import regularizers, initializers, optimizers, callbacks
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
MAX_NB_WORDS = 100000 # max number of words for tokenizer
MAX_SEQUENCE_LENGTH = 1000 # max length of each sentences, including padding
VALIDATION_SPLIT = 0.2 # 20% of data for validation (not used in training)
EMBEDDING_DIM = 100 # embedding dimensions for word vectors
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(totaldata['title_and_text'])
sequences = tokenizer.texts_to_sequences(totaldata['title_and_text'])
word_index = tokenizer.word_index
print('Vocabulary size:', len(word_index))
embeddings_index = {}
GLOVE_DIR = "/kaggle/input/glove-embeddings/glove.6B.100d.txt"
f = open(GLOVE_DIR, encoding='utf8')
for line in f:
 values = line.split()
 word = values[0]
 embeddings_index[word] = np.asarray(values[1:], dtype='float32')
f.close()
embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
 embedding_vector = embeddings_index.get(word)
 if embedding_vector is not None:
     embedding_matrix[i] = embedding_vector

model = Sequential()
model.add(Input(shape=(MAX_SEQUENCE_LENGTH,)))
model.add(Embedding(len(word_index)+1,
 EMBEDDING_DIM,
 weights = [embedding_matrix],
 input_length = MAX_SEQUENCE_LENGTH,
 trainable=False,
 name = 'embeddings'))
model.add(LSTM(60, return_sequences=True))
model.add(GlobalMaxPool1D())
model.add(Dropout(0.1))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))
data = pad_sequences(sequences, padding = 'post', maxlen = MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', y.shape)

x_train, x_test, y_train, y_test = train_test_split(data, y, shuffle=True, test_size=0.2, random_state=11)
x_test, x_val, y_test, y_val = train_test_split(data, y, shuffle=True, test_size=0.5, random_state=11)
model.compile(optimizer='adam', loss='binary_crossentropy',
 metrics = ['accuracy'])
model.summary()
import tensorflow
history = model.fit(x_train, y_train, epochs = 10, batch_size=128, validation_data=(x_val, y_val))
model2 = Sequential()
model2.add(MaxPooling2D((2, 2), input_shape=(MAX_SEQUENCE_LENGTH,)))
model2.add(Conv2D(32, (3, 3), activation='relu'))
model2.add(MaxPooling2D((2, 2)))
model2.add(Conv2D(64, (3, 3), activation='relu'))
model2.add(MaxPooling2D((2, 2)))
model2.add(Conv2D(64, (3, 3), activation='relu'))
model2.add(Flatten())
model2.add(Dense(64, activation='relu'))
model2.add(Dense(1, activation='sigmoid'))
model2.compile(optimizer='adam', loss='binary_crossentropy',
 metrics = ['accuracy'])
#model2.summary()
model2.summary()
history = model2.fit(x_train, y_train, epochs = 10, batch_size=128, validation_data=(x_val, y_val))