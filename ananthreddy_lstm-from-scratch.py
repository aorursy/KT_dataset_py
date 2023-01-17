# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import warnings

from random import shuffle

import os



warnings.filterwarnings('ignore')
train_data = pd.read_csv('../input/train.csv')

print (train_data.shape)

train_data.head()
test_data = pd.read_csv('../input/test.csv')

print (test_data.shape)

test_data.head()
train_data = train_data.drop(['id', 'qid1', 'qid2'], 1)

test_data = test_data.drop(['test_id'], 1)
train_data = train_data.fillna('')

test_data = test_data.fillna('')
import pickle

import nltk

import re

from nltk.corpus import stopwords

from nltk import word_tokenize

from string import punctuation

from nltk.stem import SnowballStemmer



stop_words = set(stopwords.words('english'))
import re



def text_to_wordlist(text, remove_stop_words=True, stem_words=False):

    # Clean the text, with the option to remove stop_words and to stem words.



    # Clean the text

    text = re.sub(r"[^A-Za-z0-9]", " ", text)

    text = re.sub(r"what's", "", text)

    text = re.sub(r"What's", "", text)

    text = re.sub(r"\'s", " ", text)

    text = re.sub(r"\'ve", " have ", text)

    text = re.sub(r"can't", "cannot ", text)

    text = re.sub(r"n't", " not ", text)

    text = re.sub(r"I'm", "I am", text)

    text = re.sub(r" m ", " am ", text)

    text = re.sub(r"\'re", " are ", text)

    text = re.sub(r"\'d", " would ", text)

    text = re.sub(r"\'ll", " will ", text)

    text = re.sub(r"60k", " 60000 ", text)

    text = re.sub(r" e g ", " eg ", text)

    text = re.sub(r" b g ", " bg ", text)

    text = re.sub(r"\0s", "0", text)

    text = re.sub(r" 9 11 ", "911", text)

    text = re.sub(r"e-mail", "email", text)

    text = re.sub(r"\s{2,}", " ", text)

    text = re.sub(r"quikly", "quickly", text)

    text = re.sub(r" usa ", " America ", text)

    text = re.sub(r" USA ", " America ", text)

    text = re.sub(r" u s ", " America ", text)

    text = re.sub(r" uk ", " England ", text)

    text = re.sub(r" UK ", " England ", text)

    text = re.sub(r"india", "India", text)

    text = re.sub(r"switzerland", "Switzerland", text)

    text = re.sub(r"china", "China", text)

    text = re.sub(r"chinese", "Chinese", text) 

    text = re.sub(r"imrovement", "improvement", text)

    text = re.sub(r"intially", "initially", text)

    text = re.sub(r"quora", "Quora", text)

    text = re.sub(r" dms ", "direct messages ", text)  

    text = re.sub(r"demonitization", "demonetization", text) 

    text = re.sub(r"actived", "active", text)

    text = re.sub(r"kms", " kilometers ", text)

    text = re.sub(r"KMs", " kilometers ", text)

    text = re.sub(r" cs ", " computer science ", text) 

    text = re.sub(r" upvotes ", " up votes ", text)

    text = re.sub(r" iPhone ", " phone ", text)

    text = re.sub(r"\0rs ", " rs ", text) 

    text = re.sub(r"calender", "calendar", text)

    text = re.sub(r"ios", "operating system", text)

    text = re.sub(r"gps", "GPS", text)

    text = re.sub(r"gst", "GST", text)

    text = re.sub(r"programing", "programming", text)

    text = re.sub(r"bestfriend", "best friend", text)

    text = re.sub(r"dna", "DNA", text)

    text = re.sub(r"III", "3", text) 

    text = re.sub(r"the US", "America", text)

    text = re.sub(r"Astrology", "astrology", text)

    text = re.sub(r"Method", "method", text)

    text = re.sub(r"Find", "find", text) 

    text = re.sub(r"banglore", "Banglore", text)

    text = re.sub(r" J K ", " JK ", text)

    

    # Remove punctuation from text

    text = ''.join([c for c in text if c not in punctuation])

    

    # Optionally, remove stop words

    if remove_stop_words:

        text = text.split()

        text = [w for w in text if not w in stop_words]

        text = " ".join(text)

    

    # Optionally, shorten words to their stems

    if stem_words:

        text = text.split()

        stemmer = SnowballStemmer('english')

        stemmed_words = [stemmer.stem(word) for word in text]

        text = " ".join(stemmed_words)

    

    # Return a list of words

    return(text)
train_data['question1_modified'] = train_data.apply(lambda x: text_to_wordlist(x['question1']), axis = 1)

train_data['question2_modified'] = train_data.apply(lambda x: text_to_wordlist(x['question2']), axis = 1)
test_data['question1_modified'] = test_data.apply(lambda x: text_to_wordlist(x['question1']), axis = 1)

test_data['question2_modified'] = test_data.apply(lambda x: text_to_wordlist(x['question2']), axis = 1)
import pickle



pickle.dump(train_data['question1_modified'], open('pickle_train_question1_modified', 'wb'))

pickle.dump(train_data['question2_modified'], open('pickle_train_question2_modified', 'wb'))



pickle.dump(test_data['question1_modified'], open('pickle_test_question1_modified', 'wb'))

pickle.dump(test_data['question2_modified'], open('pickle_test_question2_modified', 'wb'))
from keras.preprocessing.text import Tokenizer



train_text = np.hstack([train_data.question1_modified, train_data.question2_modified])

tokenizer = Tokenizer()

tokenizer.fit_on_texts(train_text)
train_data['tokenizer_1'] = tokenizer.texts_to_sequences(train_data.question1_modified)

train_data['tokenizer_2'] = tokenizer.texts_to_sequences(train_data.question2_modified)
test_text = np.hstack([test_data.question1_modified, test_data.question2_modified])

tokenizer = Tokenizer()

tokenizer.fit_on_texts(test_text)
test_data['tokenizer_1'] = tokenizer.texts_to_sequences(test_data.question1_modified)

test_data['tokenizer_2'] = tokenizer.texts_to_sequences(test_data.question2_modified)
train_data['tokenizer'] = train_data['tokenizer_1'] + train_data['tokenizer_2']
test_data['tokenizer'] = test_data['tokenizer_1'] + test_data['tokenizer_2']
print (train_data['tokenizer_1'][0])

print (train_data['tokenizer_2'][0])

print (train_data['tokenizer'][0])
print (test_data['tokenizer_1'][0])

print (test_data['tokenizer_2'][0])

print (test_data['tokenizer'][0])
max_length = 500

max_token = np.max([np.max(train_data.tokenizer.max()),np.max(test_data.tokenizer.max())])

print (max_length, max_token)
y_train = train_data[['is_duplicate']]

X_train = train_data[['tokenizer']]
X_test = test_data[['tokenizer']]
from keras.preprocessing import sequence



X_train = sequence.pad_sequences(X_train.tokenizer, maxlen = max_length)
X_test = sequence.pad_sequences(X_test.tokenizer, maxlen = max_length)
from keras.models import Sequential, Model

from keras.layers import Input, Embedding, Dense, Dropout, LSTM
model_1 = Sequential()

model_1.add(Embedding(max_token, 32))

model_1.add(Dropout(0.3))



model_1.add(LSTM(32))



model_1.add(Dropout(0.3))

model_1.add(Dense(1, activation = 'sigmoid'))

model_1.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model_1.summary()
model_1.fit(X_train, y_train, epochs = 5, batch_size=128)
prediction = model.predict(X_test)