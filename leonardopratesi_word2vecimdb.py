# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from bs4 import BeautifulSoup

# Stopwords can be useful to undersand the semantics of the sentence.

# Therefore stopwords are not removed while creating the word2vec model.

# But they will be removed  while averaging feature vectors.

from nltk.corpus import stopwords

#By being tsv format the data is TAB DELIMITED so i put delimiter = \t

train = pd.read_csv("/kaggle/input/imdb-dataset/labeledTrainData.tsv", header=0,\

                    delimiter="\t")



test = pd.read_csv("/kaggle/input/imdb-dataset/testData.tsv",header=0,\

                    delimiter="\t")



train
# one example of review

test['review'][0]
#the dataset is currently made of a review being a piece of text,

#it needs to be converted to an array of words

#lets create a list of list of words



tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')



# strip removes trailing space

sentences = []

for review in train['review']:

    sentences += tokenizer.tokenize(review.strip())

    
sentences[0]
# let's split the words

post_sent = []

for sent in sentences:

    if len(sent) > 0:

        post_sent.append(sent.lower().split())
post_sent[0]
#i need to remove stopwords

post_sent_stop = []

for sent in post_sent:

    stops = set(stopwords.words("english"))     

    words = [w for w in sent if not w in stops]

    post_sent_stop.append(words) 

    

post_sent_stop[0]
from gensim.models import word2vec

print("Training model....")

#min_count = words with occurences less than this will be ignored

#windows = distance between target and words around

model = word2vec.Word2Vec(post_sent_stop,

                          workers=4,

                          size=300,

                          min_count=40,

                          window=10,

                          sample= 1e-3)

print("finish")



# To make the model memory efficient

model.init_sims(replace=True)



# saving the model for later use. Word2Vec.load()

model_name = "word2vec"

model.save(model_name)

#let's check the results

model.wv.most_similar('man')
model.wv.most_similar("bad")
model.wv.doesnt_match("france england germany berlin".split())

#total number of words

model.wv.vectors.shape

from keras.models import Sequential

from keras.layers import Dense, LSTM, GRU



model = Sequential()

model.add(Dense(300, activation='relu'))

model.add(Dense(100, activation='relu'))

model.add(Dense(1))



model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(post_sent_stop, train['sentiment'], batch_size= 100)