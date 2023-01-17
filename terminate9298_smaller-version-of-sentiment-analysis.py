# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import regex as re

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

%matplotlib inline



import warnings

warnings.filterwarnings("ignore")

# Scikit-learn

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from sklearn.manifold import TSNE

from sklearn.feature_extraction.text import TfidfVectorizer



# Keras

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Conv1D, MaxPooling1D, LSTM

from keras import utils

from keras.callbacks import ReduceLROnPlateau, EarlyStopping



# nltk

import nltk

from nltk.corpus import stopwords

from  nltk.stem import SnowballStemmer



# Word2vec

import gensim



# Utility

import re

import os

from collections import Counter

import logging

import time

import pickle

import itertools

import gc

import json

from keras_preprocessing.text import tokenizer_from_json

from keras.models import model_from_json



pd.set_option('max_colwidth', 500)

pd.set_option('max_columns', 500)

pd.set_option('max_rows', 100)
tweet = pd.read_csv('/kaggle/input/first-gop-debate-twitter-sentiment/Sentiment.csv')[['sentiment' , 'text']]
def tweet_sentiment(text):

    if str(text) == 'Negative':

        return 0

    if str(text) == 'Neutral':

        return 2

    if str(text) =='Positive':

        return 1

tweet['sentiment'] = tweet.sentiment.apply(lambda x: tweet_sentiment(x))

tweet = tweet[tweet.sentiment != 2]
# tweet['text'] = tweet['text'].apply(lambda x: re.sub('@[a-zA-Z0-9]+:','',x))

# tweet['text'] = tweet['text'].apply(lambda x: re.sub('@[a-zA-Z0-9]+','',x))

# tweet['text'] = tweet['text'].apply(lambda x: re.sub('RT','',x))

# tweet['text'] = tweet['text'].apply(lambda x: re.sub('[^a-zA-Z0-9\s]','',x))

# tweet['text'] = tweet['text'].apply(lambda x: x.lower())

tweet['count'] = tweet['text'].apply(lambda x: len(x.split()))
tweet = tweet.rename(columns = {'text': 'reviews' })
tweet.head()
alexa = pd.read_csv('/kaggle/input/amazon-alexa-reviews/amazon_alexa.tsv' , delimiter = '\t')[['verified_reviews','feedback']]

alexa = alexa.rename(columns={'verified_reviews':'reviews' , 'feedback':'sentiment'})

alexa['count'] = alexa['reviews'].apply(lambda x: len(x.split()))
alexa.head()
data = pd.concat([alexa , tweet], axis= 0 , sort = True)
data = data[['reviews' , 'sentiment']]
stop_words = stopwords.words("english")

stemmer = SnowballStemmer("english")

TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

def preprocess(text, stem=False):

    # Remove link,user and special characters

    text = re.sub(TEXT_CLEANING_RE, ' ', str(text).lower()).strip()

    tokens = []

    for token in text.split():

        if token not in stop_words:

            if stem:

                tokens.append(stemmer.stem(token))

            else:

                tokens.append(token)

    return " ".join(tokens)

data.reviews = data.reviews.apply(lambda x: preprocess(x))
documents = [_text.split() for _text in data.reviews] 
W2V_SIZE = 30

W2V_WINDOW = 7

W2V_EPOCH = 16

W2V_MIN_COUNT = 5

w2v_model = gensim.models.word2vec.Word2Vec(size=W2V_SIZE, 

                                            window=W2V_WINDOW, 

                                            min_count=W2V_MIN_COUNT, 

                                            workers=8)





w2v_model.build_vocab(documents)
words = w2v_model.wv.vocab.keys()

vocab_size = len(words)

print("Vocab size", vocab_size)
w2v_model.train(documents, total_examples=len(documents), epochs=W2V_EPOCH)
w2v_model.most_similar("awesome")
tokenizer = Tokenizer()

tokenizer.fit_on_texts(data.reviews)

vocab_size = len(tokenizer.word_index)+1

print('Vocab Size is ',vocab_size)
SEQUENCE_LENGTH = 30

EPOCHS = 8

BATCH_SIZE = 1024

x_data = pad_sequences(tokenizer.texts_to_sequences(data.reviews) , maxlen = SEQUENCE_LENGTH)
y_data = data.sentiment

print(x_data.shape)

print(y_data.shape)

y_data = y_data.values.reshape(-1,1)
embedding_matrix = np.zeros((vocab_size , W2V_SIZE))

for word , i in tokenizer.word_index.items():

    if word in w2v_model.wv:

        embedding_matrix[i] = w2v_model.wv[word]

print(embedding_matrix.shape)

embedding_layer = Embedding( vocab_size , W2V_SIZE , weights = [embedding_matrix] , input_length = SEQUENCE_LENGTH, trainable = False)

model = Sequential()

model.add(embedding_layer)

model.add(Dropout(0.5))

model.add(LSTM(100 , dropout = 0.2 , recurrent_dropout = 0.2 ))

model.add(Dense(1 , activation = 'sigmoid'))

model.summary()
model.compile(loss = 'binary_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'] )

callbacks = [ ReduceLROnPlateau(monitor='val_loss', patience=5, cooldown=0),

              EarlyStopping(monitor='val_acc', min_delta=1e-4, patience=5)]
history = model.fit(x_data , y_data , batch_size = BATCH_SIZE , epochs = EPOCHS , validation_split = 0.1  , verbose = 1 , callbacks = callbacks)
def predict(text):

    start_at = time.time()

    # Tokenize text

    x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=SEQUENCE_LENGTH)

    # Predict

    score = model.predict([x_test])[0]



    return {"score": float(score),

       "elapsed_time": time.time()-start_at}  



print(predict('i am Happy'))

print(predict('i not feeling so great .Little Rest can help but you decide what should i do next '))

print(predict('i am sitting in library for 6 hours . i learned alot but i am tired'))

print(predict('i am tired'))

print(predict('good is not good'))

print(predict('bad is not good'))

print(predict('good is not bad'))

print(predict('how i can end up here'))

print(predict('i am sad'))

print(predict('i donot think this is working i tried everything but i have to think of some thing else'))
model.save_weights('model_weights.h5')

with open('model_architecture.json', 'w') as f:

    f.write(model.to_json())

    

model.save('entire_model.h5')

tokenizer_json = tokenizer.to_json()

with open('tokenizer.json', 'w', encoding='utf-8') as f:

    f.write(json.dumps(tokenizer_json, ensure_ascii=False))


