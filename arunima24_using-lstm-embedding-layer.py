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
#Importing the Libraries



import os

import sys

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from keras.models import Model

from keras.layers import Dense, Embedding, Input

from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.optimizers import Adam

from sklearn.metrics import roc_auc_score



import keras.backend as K
#I've added glove vectors in the input. https://www.kaggle.com/rtatman/glove-global-vectors-for-word-representation



#Loading the word vectors to convert the words from the tweets to vector format



print('Loading word vectors...')

word2vec = {}

with open(os.path.join('../input/glove6b50dtxt/glove.6B.50d.txt'), encoding = "utf-8") as f:

  # is just a space-separated text file in the format:

  # word vec[0] vec[1] vec[2] ...

  for line in f:

    values = line.split() #split at space

    word = values[0]

    vec = np.asarray(values[1:], dtype='float32') #numpy.asarray()function is used when we want to convert input to an array.

    word2vec[word] = vec

print('Found %s word vectors.' % len(word2vec))

#Reading the tweets to dataframe



print('Loading in tweets...')



train = pd.read_csv("../input/nlp-getting-started/train.csv")

train.head()
#Removing (dropping) the columns 'keyword' and 'location'as we're concerned with the tweets' text



train = train.drop(["keyword","location"],axis=1)

train.head()
#Storing the values of tweets and target in respective variables 



tweets = train["text"].values

target = train["target"].values

#Tokenizing the words



tokenizer = Tokenizer(num_words=20000) #vectorize a text corpus, by turning each text into either a sequence of integers (each integer being the index of a token in a dictionary) or into a vector where the coefficient for each token could be binary, based on word count, based on tf-idf.

tokenizer.fit_on_texts(tweets) #Updates internal vocabulary based on a list of texts.

sequences = tokenizer.texts_to_sequences(tweets) #Converts a text to a sequence of words (or tokens).

#Creating an array for indexing each word 



word2idx = tokenizer.word_index #indexing each word from vector list

print('Found %s unique tokens.' % len(word2idx))



data = pad_sequences(sequences,100) #padding each tweet vector with 0s to a uniform length of 100

print('Shape of data tensor:', data.shape)

print('Filling pre-trained embeddings...')

num_words = min(20000, len(word2idx) + 1)

embedding_matrix = np.zeros((num_words, 50)) #fill array embedding_matrix with 0s with size num_words, embedding_matrix i.e. 20000,50

#Creating an embedding matrix to create the embedding layer for LSTM



embedding1=[]

for word, i in word2idx.items():

    if i < 20000:

        embedding1 = word2vec.get(word)

        if embedding1 is not None:

            embedding_matrix[i] = embedding1

#Embedding layer



embedding_layer = Embedding( #Turns positive integers (indexes) into dense vectors of fixed size.

  num_words,

  50,

  weights=[embedding_matrix],

  input_length=100,

  trainable=False

)

#Creating the model



print('Building model...')



# create an LSTM network with a single LSTM

input_ = Input(shape=(100,))

x = embedding_layer(input_)

x = Bidirectional(LSTM(15, return_sequences=True))(x)

x = GlobalMaxPool1D()(x)

output = Dense(1, activation="sigmoid")(x)



model = Model(input_, output)

model.compile(

  loss='binary_crossentropy',

  optimizer=Adam(lr=0.01),

  metrics=['accuracy'],

)

model.summary()
#training the model



print('Training model...')

r = model.fit(

  data,

  target,

  batch_size=128,

  epochs=100,

  validation_split=0.2

)



print("Done with the Training")

#Repeating the steps for test dataset to predict the values obtained from the model training



print("Loading in the test dataset\n")



test = pd.read_csv("../input/nlp-getting-started/test.csv")

test.head()

test = test.drop(["keyword","location"],axis=1)

tweets_test = test["text"].values



tokenizer = Tokenizer(num_words=20000) #vectorize a text corpus, by turning each text into either a sequence of integers (each integer being the index of a token in a dictionary) or into a vector where the coefficient for each token could be binary, based on word count, based on tf-idf.

tokenizer.fit_on_texts(tweets_test) #Updates internal vocabulary based on a list of texts.

sequences = tokenizer.texts_to_sequences(tweets_test) #Converts a text to a sequence of words (or tokens).

word2idx = tokenizer.word_index #indexing each word from vector list

print('Found %s unique tokens.' % len(word2idx))



data = pad_sequences(sequences,100)

print('Shape of data tensor:', data.shape)





print("Predictions:\n\n")

test['target'] = model.predict(data) #predicting the data

test.head()

#Storing the contents of the test dataset into a csv file



import csv

test = test.drop(["text"],axis=1)

test.to_csv("sample_submission.csv",index=False)
#Reading the csv file



sub = pd.read_csv("sample_submission.csv")

sub.head()