import numpy as np 

import pandas as pd 

import os

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train_data = pd.read_csv("../input/word2vec-nlp-tutorial/labeledTrainData.tsv", delimiter='\t')

train_data
test_data =  pd.read_csv("../input/word2vec-nlp-tutorial/testData.tsv", delimiter='\t')
import re

import pandas

import numpy

import json

from bs4 import BeautifulSoup

from nltk.corpus import stopwords

from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from tensorflow.python.keras.preprocessing.text import Tokenizer
def preprocess(review,remove_stopwords = False):

    # html 제거

    review_text = BeautifulSoup(review,"html5lib").get_text()

    

    # 특수문자 제거

    review_text = re.sub("[^a-zA-Z]"," ",review_text)

    

    # 소문자로 통일 후 리스트화

    words = review_text.lower().split()

    

    if remove_stopwords:

        # 불용어 제거

        stop_words = set(stopwords.words('english'))

        words = [w for w in words if not w in stop_words]

 

    clean_review = ' '.join(words)

    

    return clean_review
clean_train_reviews = []

clean_test_reviews = []



for review in train_data['review']:

    clean_train_reviews.append(preprocess(review,remove_stopwords = True))

    

for review in test_data['review']:

    clean_test_reviews.append(preprocess(review, remove_stopwords = True))
MAX_SEQUENCE_LENGTH = 174



tokenizer = Tokenizer()



tokenizer.fit_on_texts(clean_train_reviews + clean_train_reviews)



text_sequences = tokenizer.texts_to_sequences(clean_train_reviews)

train_inputs = pad_sequences(text_sequences,maxlen=MAX_SEQUENCE_LENGTH, padding='post')



text_sequences = tokenizer.texts_to_sequences(clean_test_reviews)

test_inputs = pad_sequences(text_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
from __future__ import absolute_import, division, print_function, unicode_literals



import tensorflow as tf

from tensorflow import keras



import numpy as np



print(tf.__version__)
len(tokenizer.word_index)
vocab_size = len(tokenizer.word_index) + 1



model = keras.Sequential()

model.add(keras.layers.Embedding(vocab_size, 16))

model.add(keras.layers.GlobalAveragePooling1D())

model.add(keras.layers.Dense(16, activation=tf.nn.relu))

model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))



model.summary()
model.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['acc'])
x_val = train_inputs[:5000]

partial_x_train = train_inputs[5000:]



y_val = np.array(train_data['sentiment'][:5000])

partial_y_train = np.array(train_data['sentiment'][5000:])
type(partial_y_train)
history = model.fit(partial_x_train,

                    partial_y_train,

                    epochs=30,

                    batch_size=512,

                    verbose=1)
pred_val = np.rint(model.predict(x_val)).astype('int32').squeeze()



for i in range(50):

    print(y_val[i], pred_val[i])
pred = np.rint(model.predict(test_inputs)).astype('int32').squeeze()



print(pred.shape)
commit_df = pd.DataFrame({'id': test_data['id'], 'sentiment':pred})
commit_df.to_csv('commit7.csv', index = False)
commit_test = pd.read_csv("./commit7.csv")

commit_test
sample = pd.read_csv("../input/word2vec-nlp-tutorial/sampleSubmission.csv")

sample