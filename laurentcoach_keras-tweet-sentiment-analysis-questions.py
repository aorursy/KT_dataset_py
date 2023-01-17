import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from numpy import array

import numpy as np

from keras.preprocessing.text import one_hot

from keras.preprocessing.sequence import pad_sequences

from keras.preprocessing.text import text_to_word_sequence

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Flatten

from keras.layers.embeddings import Embedding

import pandas as pd



from bs4 import BeautifulSoup

import re

import string

import nltk

from nltk.tokenize import WordPunctTokenizer



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/"))



# Any results you write to the current directory are saved as output.
#Load TRAIN & TEST Dataset

train = pd.read_csv("../input/train-dataset2/train_E6oV3lV.csv")

test = pd.read_csv("../input/test-dataset/test_tweets_anuFYb8.csv")
pat1 = r'@[A-Za-z0-9]+'

pat2 = r'https?://[A-Za-z0-9./]+'

combined_pat = r'|'.join((pat1, pat2))

def tweet_cleaner(text):

    tok = WordPunctTokenizer()



    soup = BeautifulSoup(text, 'lxml')

    souped = soup.get_text()

    stripped = re.sub(combined_pat, '', souped)

    try:

        clean = stripped.decode("utf-8-sig").replace(u"\ufffd", "?")

    except:

        clean = stripped

    letters_only = re.sub("[^a-zA-Z]", " ", clean)

    lower_case = letters_only.lower()

    # During the letters_only process two lines above, it has created unnecessay white spaces,

    # I will tokenize and join together to remove unneccessary white spaces

    words = tok.tokenize(lower_case)

    return (" ".join(words)).strip()
training = train.tweet

dftrain = []

for t in training:

    dftrain.append(tweet_cleaner(t))
label = array(train.label)
vocab_size_train = 50

encoded_train = [one_hot(d, vocab_size_train) for d in dftrain]
max_length = 150 # Max number of word in a sentence

padded_train = pad_sequences(encoded_train, maxlen=max_length, padding='post')
model = Sequential()

model.add(Embedding(vocab_size_train, 8, input_length=max_length))

model.add(Flatten())

model.add(Dense(1, activation='sigmoid'))

# compile the model

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

# summarize the model

print(model.summary())
model.fit(padded_train, label, epochs=10, verbose=0)

# evaluate the model

loss, accuracy = model.evaluate(padded_train, label, verbose=0)

print('Accuracy: %f' % (accuracy*100))
# Clean Test dataset

testing = test.tweet

dftest = []

for t in testing:

    dftest.append(tweet_cleaner(t))
#Word Embedding

vocab_size_train = 50

encoded_test = [one_hot(d, vocab_size_train) for d in dftest]

max_length = 150 # Max number word in a sentence

padded_test = pad_sequences(encoded_test, maxlen=max_length, padding='post')
#Predict Test data

predictions = model.predict(padded_test, verbose=0)
predictions