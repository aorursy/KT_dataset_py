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
import re

from nltk.tokenize import word_tokenize

from collections import defaultdict, Counter

import random

import string



import bs4 as bs

import urllib.request

import re
raw_html = urllib.request.urlopen('https://en.wikipedia.org/wiki/World_War_II')

raw_html = raw_html.read()



article_html = bs.BeautifulSoup(raw_html, 'lxml')

article_paragraphs = article_html.find_all('p')

article_text = ''



for para in article_paragraphs:

    article_text += para.text



article_text = article_text.lower()
class MarkovChain:

    def __init__(self):

        self.lookup_dict = defaultdict(list)



    def _preprocess(self, string):

        cleaned = re.sub(r'\W+', ' ', string).lower()

        tokenized = word_tokenize(cleaned)

        return tokenized



    def add_document(self, string):

        preprocessed_list = self._preprocess(string)

        pairs = self.__generate_tuple_keys(preprocessed_list)

        for pair in pairs:

            self.lookup_dict[pair[0]].append(pair[1])



    def __generate_tuple_keys(self, data):

        if len(data) < 1:

            return

        for i in range(len(data) - 1):

            yield [ data[i], data[i + 1] ]



    def generate_text(self, string):

        if len(self.lookup_dict) > 0:

            print("Next word suggestions:", Counter(self.lookup_dict[string]).most_common()[:3])

        return
#training_1 = "How are you? How many people attend? How did it go?"

training_1 = article_text

my_next_word = MarkovChain()

my_next_word.add_document(training_1)
#my_next_word.generate_text(input().lower())

my_next_word.generate_text("world")
class MarkovChain:

    def __init__(self):

        self.lookup_dict = defaultdict(list)



    def _preprocess(self, string):

        cleaned = re.sub(r'\W+', ' ', string).lower()

        tokenized = word_tokenize(cleaned)

        return tokenized

    

    def add_document(self, string):

        preprocessed_list = self._preprocess(string)

        pairs = self.__generate_tuple_keys(preprocessed_list)

        for pair in pairs:

            self.lookup_dict[pair[0]].append(pair[1])

        pairs2 = self.__generate_2tuple_keys(preprocessed_list)

        for pair in pairs2:

            self.lookup_dict[tuple([pair[0], pair[1]])].append(pair[2])

        pairs3 = self.__generate_3tuple_keys(preprocessed_list)

        for pair in pairs3:

            self.lookup_dict[tuple([pair[0], pair[1], pair[2]])].append(pair[3])



    def __generate_tuple_keys(self, data):

        if len(data) < 1:

            return 

        for i in range(len(data) - 1):

            yield [ data[i], data[i + 1] ]



    #to add two words tuple as key and the next word as value

    def __generate_2tuple_keys(self, data):

        if len(data) < 2:

            return

        for i in range(len(data) - 2):

            yield [ data[i], data[i + 1], data[i+2] ]

  

    #to add three words tuple as key and the next word as value 

    def __generate_3tuple_keys(self, data):

        if len(data) < 3:

            return

        for i in range(len(data) - 3):

            yield [ data[i], data[i + 1], data[i+2], data[i+3] ]



    def oneword(self, string):

        return Counter(self.lookup_dict[string]).most_common()[:3]



    def twowords(self, string):

        suggest = Counter(self.lookup_dict[tuple(string)]).most_common()[:3]

        if len(suggest)==0:

            return self.oneword(string[-1])

        return suggest

    

    def threewords(self, string):

        suggest = Counter(self.lookup_dict[tuple(string)]).most_common()[:3]

        if len(suggest)==0:

            return self.twowords(string[-2:])

        return suggest

    

    def morewords(self, string):

        return self.threewords(string[-3:])



    def generate_text(self, string):

        if len(self.lookup_dict) > 0:

            tokens = string.split(" ")

            if len(tokens)==1:

                print("Next word suggestions:", self.oneword(string))

            elif len(tokens)==2:

                print("Next word suggestions:", self.twowords(string.split(" ")))

            elif len(tokens)==3:

                print("Next word suggestions:", self.threewords(string.split(" ")))

            elif len(tokens)>3:

                print("Next word suggestions:", self.morewords(string.split(" ")))

        return
#training_1 = "How are you? How many people attend? How did it go?"

training_1 = article_text

my_next_word = MarkovChain()

my_next_word.add_document(training_1)
#my_next_word.generate_text(input().lower())

my_next_word.generate_text("world war")
#pip install python-docx

#pip install doc3
#pip install docx
from keras.preprocessing.text import Tokenizer

import nltk

from nltk.tokenize import word_tokenize

import numpy as np

import re

from keras.utils import to_categorical

#from doc3 import training_doc3



#training_doc3 = 'Hello, this is a sample test file for our next word prediction model. How are you doing? How many people have attended the session? When will you attend the class? What time is the class scheduled at? What is the agenda of the class?'

training_doc3 = article_text



cleaned = re.sub(r'\W+', ' ', training_doc3).lower()

tokens = word_tokenize(cleaned)

train_len = 4

text_sequences = []

for i in range(train_len,len(tokens)):

    seq = tokens[i-train_len:i]

    text_sequences.append(seq)



sequences = {}

count = 1

for i in range(len(tokens)):

    if tokens[i] not in sequences:

        sequences[tokens[i]] = count

        count += 1



tokenizer = Tokenizer()

tokenizer.fit_on_texts(text_sequences)

sequences = tokenizer.texts_to_sequences(text_sequences)



#vocabulary size increased by 1 for the cause of padding

vocabulary_size = len(tokenizer.word_counts)+1

n_sequences = np.empty([len(sequences),train_len], dtype='int32')



for i in range(len(sequences)):

    n_sequences[i] = sequences[i]



train_inputs = n_sequences[:,:-1]

train_targets = n_sequences[:,-1]

train_targets = to_categorical(train_targets, num_classes=vocabulary_size)

seq_len = train_inputs.shape[1]
from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Embedding

model = Sequential()

model.add(Embedding(vocabulary_size, seq_len, input_length=seq_len))

model.add(LSTM(50,return_sequences=True))

model.add(LSTM(50))

model.add(Dense(50,activation='relu'))

model.add(Dense(vocabulary_size, activation='softmax'))

# compiling the network

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_inputs,train_targets,epochs=100,verbose=0)
from keras.preprocessing.sequence import pad_sequences

#input_text = input().strip().lower()

input_text = "hiroshima and".strip().lower()

encoded_text = tokenizer.texts_to_sequences([input_text])[0]

pad_encoded = pad_sequences([encoded_text], maxlen=seq_len, truncating='pre')

print(encoded_text, pad_encoded)

for i in (model.predict(pad_encoded)[0]).argsort()[-3:][::-1]:

  pred_word = tokenizer.index_word[i]

  print("Next word suggestion:",pred_word)