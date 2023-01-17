# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn import preprocessing
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        read_data = pd.read_csv(os.path.join(dirname, filename))

#exploring data
print(read_data.head())

#independent training values
x_values = read_data["review"]
print(x_values[1:10])

#dependent values
label_encoder = preprocessing.LabelEncoder()
y_values = label_encoder.fit_transform(read_data['sentiment'])
print(y_values[0:10])

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
stopwords = set(stopwords.words('english'))
print(stopwords)

splitted_list = x_values[0].split(" ")
print(splitted_list)
from bs4 import BeautifulSoup
for i,word_data in enumerate(x_values):
    x_values[i] = BeautifulSoup(x_values[i], 'lxml').get_text()
    splitted_values = x_values[i].split(" ")
    x_values[i] = []
    for soft_words_split in splitted_values:
        if soft_words_split not in stopwords:
            x_values[i].append(soft_words_split + " ")
            
print(x_values[0])
#tokenize
#give input to embedding layer

from keras.preprocessing.text import one_hot
from keras.preprocessing.text import text_to_word_sequence

encoded_words = x_values;
vocab_size = len(x_values[0])
print(vocab_size)
# integer encode the document
listToStr = ' '.join([str(elem) for elem in x_values[0]])
result = one_hot(listToStr, round(vocab_size*1.3))
print(result)

for i,values in enumerate(x_values):
    listToStr = ' '.join([str(elem) for elem in x_values[i]])
    vocab_size = len(x_values[i])
    encoded_words[i] = one_hot(listToStr, round(vocab_size*1.3))


print(len(encoded_words[0]))

max_strings_length = 0
for values in encoded_words:
    if max_strings_length < len(values): max_strings_length = len(values)

X_data = sequence.pad_sequences(encoded_words, maxlen=max_strings_length)

x_train, x_test, y_train, y_test = train_test_split(X_data, y_values, test_size=0.33)

from keras.layers import LSTM
from keras.layers import Dropout

model = Sequential()
model.add(Embedding(5000,32,input_length = max_strings_length))
model.add(LSTM(200, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(100, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=25)

accuracy_score = model.evaluate(x_test,y_test, verbose=0)
print(accuracy_score)
import random

for i in range(9):
    RandomNumber = random.randint(0,len(x_test))
    sentiment = model.predict_classes(x_test[RandomNumber]) 
    print("sentence: %s and sentiment: %d",x_test[RandomNumber],sentiment)