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

dataframe = pd.read_csv("../input/training.1600000.processed.noemoticon.csv", encoding = "ISO-8859-1")
# Here we should check that the pandas dataframe contains relevant and meaningful information
print(dataframe.shape) # The dataframe contains 16 000 000 lines and 6 colums
# First let's check the column names of the dataframe
print(list(dataframe.columns.values))
# Let's print some of the lines of this dataframe (just 5 by default)
print(dataframe.head())
# Here with 20 lines
print(dataframe.head(n=20))
# Here with the last 20 lines to see if there is any change in the data
print(dataframe.tail(n= 20))

# Here we can immediatly see 2 things: 
# 1/ Tweets contains really diverse information with things such as urls, ponctuation, and tags
# These tags are not relevant for sentiment classification and should probably be removed from the training, so that the model does not try to overinterpret the tweets content.
# Hence, we should first take the time to prepare the data and perform data preprocessing.
# 2/ Some colums are not useful at all to predict positive or negative sentiment and should be removed from the training model.
# Here for example, we don't want to keep the tweet date or the author's name
# In fact we should drop all the columns except the sentiment classifier (column 0) and the tweet itself (column 6) => Speed up the training phase and increase accuracy
# The author column is also not so interesting for the learning process since it shouldn't be an indicator for positive or negative sentiment
# We decide to also drop this column
data = dataframe.iloc[:, [0, 5]].sample(frac=1).reset_index(drop=True)
# The dataframe now contains 2 colums which are respectively the sentiment and the tweet itself
# Let's now explore a bit more the data to evaluate the global accuracy of the tweets classification which was performed automatically
print(data.shape)
print(data)
# We have 2 classes for sentiment classification: either positive (0) or negative (4)
# We should replace these 2 labels by 0 for a negative sentiment and 1 for a positive sentiment for better clarity

data.iloc[:, 0] = data.iloc[:, 0].replace(4, 1)
print(data.shape)
print(data)
# In progress: data visualization (10 tweets with classes 0 and 10 tweets whith class 1)
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

# On pourrait supprimer les mots qui n'ont pas de signification particulière, qui ne permettent pas d'établir la positivité ou négativité d'un tweet
tweets = np.array(data.iloc[:, 1].apply(preprocess_tweet).values)
sentiment = np.array(data.iloc[:, 0].values)

print(tweets)
from keras.preprocessing.text import Tokenizer

vocab_size = 400000
tk = Tokenizer(num_words=vocab_size)
#tw = tweets
tk.fit_on_texts(tweets)
t = tk.texts_to_sequences(tweets)
X = np.array(sequence.pad_sequences(t, maxlen=20, padding='post'))
y = sentiment

print(X)
print(X.shape, y.shape)
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

y[y == 4] = 1

model = Sequential()

model.add(Embedding(vocab_size, 32, input_length=20))

# Add a LSTM layer to see if we can get better results than just with convolution and dense layers.
# from keras import regularizers
# embed_dim = 128
# lstm_out = 256
# model.add(LSTM(lstm_out, dropout_U=0.2, dropout_W=0.2, kernel_regularizer=regularizers.l2(0.0004),activity_regularizer=regularizers.l1(0.0002)))
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

history = model.fit(X, y, batch_size=128, verbose=1, validation_split=0.2, epochs=5)


model.save('model.h5')

with open('file_to_write', 'w') as f:
    s = "Accuracy " + str(history.history['acc'][0]) + " train " +  str(history.history['val_acc'][0]) + " validation"
    f.write(s)

plt.plot(history.history['loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
plt.savefig('Figure 1')

plt.plot(history.history['acc'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()
plt.savefig('Figure 1')

