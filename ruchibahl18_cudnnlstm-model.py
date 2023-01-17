# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import Sequential

from keras.layers import CuDNNLSTM, Dropout, Dense,Conv1D, MaxPooling1D

from keras.layers.embeddings import Embedding

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

import string

import nltk

import re

from nltk.corpus import stopwords

from nltk.stem.snowball import SnowballStemmer

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
test_data = pd.read_csv("../input/test.tsv",delimiter='\t')

train_data = pd.read_csv("../input/train.tsv",delimiter="\t")
def clean_text(text):

    

    ## Remove puncuation

    text = text.translate(string.punctuation)

    

    ## Convert words to lower case and split them

    text = text.lower().split()

    

    ## Remove stop words

    stops = set(stopwords.words("english"))

    text = [w for w in text if not w in stops and len(w) >= 3]

    

    text = " ".join(text)

    ## Clean the text

    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)

    text = re.sub(r"what's", "what is ", text)

    text = re.sub(r"\'s", " ", text)

    text = re.sub(r"\'ve", " have ", text)

    text = re.sub(r"n't", " not ", text)

    text = re.sub(r"i'm", "i am ", text)

    text = re.sub(r"\'re", " are ", text)

    text = re.sub(r"\'d", " would ", text)

    text = re.sub(r"\'ll", " will ", text)

    text = re.sub(r",", " ", text)

    text = re.sub(r"\.", " ", text)

    text = re.sub(r"!", " ! ", text)

    text = re.sub(r"\/", " ", text)

    text = re.sub(r"\^", " ^ ", text)

    text = re.sub(r"\+", " + ", text)

    text = re.sub(r"\-", " - ", text)

    text = re.sub(r"\=", " = ", text)

    text = re.sub(r"'", " ", text)

    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)

    text = re.sub(r":", " : ", text)

    text = re.sub(r" e g ", " eg ", text)

    text = re.sub(r" b g ", " bg ", text)

    text = re.sub(r" u s ", " american ", text)

    text = re.sub(r"\0s", "0", text)

    text = re.sub(r" 9 11 ", "911", text)

    text = re.sub(r"e - mail", "email", text)

    text = re.sub(r"j k", "jk", text)

    text = re.sub(r"\s{2,}", " ", text)

    ## Stemming

    text = text.split()

    stemmer = SnowballStemmer('english')

    stemmed_words = [stemmer.stem(word) for word in text]

    text = " ".join(stemmed_words)

    return text
X = [clean_text(w) for w in train_data["Phrase"]]
train_data.head()
max_len = max([len(s.split()) for s in train_data['Phrase']])
vocab_size = 20000

tokenizer = Tokenizer(num_words = vocab_size)

tokenizer.fit_on_texts(X)
sequences = tokenizer.texts_to_sequences(X)

X = pad_sequences(sequences, maxlen=50)
from keras.utils import np_utils
Y = np_utils.to_categorical(train_data["Sentiment"])
#embedding_dim = len(tokenizer.word_index)+1
model = Sequential()

model.add(Embedding(vocab_size, 128, input_length=50))

model.add(Dropout(0.2))

model.add(Conv1D(64, 5, activation='tanh'))

model.add(MaxPooling1D(pool_size=4))

model.add(CuDNNLSTM(128))

model.add(Dense(Y.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=10, validation_split=0.2)
X_test = [clean_text(w) for w in test_data["Phrase"]]

vocab_size = 20000

tokenizer = Tokenizer(num_words = vocab_size)

tokenizer.fit_on_texts(X_test)

sequences = tokenizer.texts_to_sequences(X_test)

X_test = pad_sequences(sequences, maxlen=50)
Y_test = model.predict(X_test)
Y_test = [np.argmax(val) for val in Y_test]
submission = pd.DataFrame({'PhraseId' : test_data["PhraseId"],

                                'Sentiment' : Y_test})
submission.to_csv("submission.csv", index=False)