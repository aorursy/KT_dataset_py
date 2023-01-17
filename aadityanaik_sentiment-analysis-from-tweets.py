# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import RegexpTokenizer

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

dataframe = pd.read_csv("../input/training.1600000.processed.noemoticon.csv", encoding = "ISO-8859-1", header=None).iloc[:, [0, 4, 5]].sample(frac=1).reset_index(drop=True)
from keras.preprocessing import sequence
import re

def preprocess_tweet(tweet):
    #Preprocess the text in a single tweet
    #arguments: tweet = a single tweet in form of string 
    #convert the tweet to lower case
    tweet.lower()
    #convert all urls to sting "URL"
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
    #convert all @username to "AT_USER"
    tweet = re.sub('@[^\s]+','AT_USER', tweet)
    #correct all multiple white spaces to a single white space
    tweet = re.sub('[\s]+', ' ', tweet)
    #convert "#topic" to just "topic"
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    tweet = re.sub(r'\W*\b\w{1,3}\b', '', tweet)
    return tweet

users = np.array(dataframe.iloc[:, 1].values)
tweets = np.array(dataframe.iloc[:, 2].apply(preprocess_tweet).values)
sentiment = np.array(dataframe.iloc[:, 0].values)

print(tweets)
from keras.preprocessing.text import Tokenizer

vocab_size = 400000
tk = Tokenizer(num_words=vocab_size)
#tw = tweets
tk.fit_on_texts(tweets)
t = tk.texts_to_sequences(tweets)
X = np.array(sequence.pad_sequences(t, maxlen=20, padding='post'))
y = sentiment

print(X.shape, y.shape)
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

y[y == 4] = 1

model = Sequential()

model.add(Embedding(vocab_size, 32, input_length=20))
model.add(Conv1D(filters=128, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))
model.add(Conv1D(filters=64, kernel_size=6, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))
model.add(Conv1D(filters=32, kernel_size=7, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))
model.add(Conv1D(filters=32, kernel_size=8, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit(X, y, batch_size=128, verbose=1, validation_split=0.2, epochs=10)

model.save('model.h5')

with open('file_to_write', 'w') as f:
    s = "Accuracy " + history.history['acc'] + " train " + history.history['val_acc'] + " validation"
    f.write(s)

plt.plot(history.history['loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
plt.savefig('Figure 1')

plt.plot(history.history['acc'])
plt.xlabel('Accuracy')
plt.ylabel('Loss')
plt.show()
plt.savefig('Figure 1')

