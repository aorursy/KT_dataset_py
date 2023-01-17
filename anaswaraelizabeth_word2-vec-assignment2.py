# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

print(os.listdir("../input/embeddings"))



# Any results you write to the current directory are saved as output.
from gensim.models import KeyedVectors

path='../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'

word2vec=KeyedVectors.load_word2vec_format(path,binary=True)



url='https://raw.githubusercontent.com/AnaswaraElizabethAnt/Datasets/master/yelp_labelled.csv'

yelp = pd.read_csv(url, sep='\t',header=None)

yelp.head()
yelp.columns = ['reviews','sentiment']

yelp.head()
import nltk

stopwords = nltk.corpus.stopwords.words('english')

docs_vectors = pd.DataFrame()

for doc in yelp['reviews'].str.lower().str.replace('[^a-z ]',''):

    words = nltk.word_tokenize(doc)

    words_clean = [word for word in words if word not in stopwords]

    temp = pd.DataFrame()

    for word in words_clean:

        try:

            word_vec = pd.Series(word2vec[word])

            temp = temp.append(word_vec, ignore_index = True)

        except:

            pass

    # coming out of first doc

    temp_avg = temp.mean()

    docs_vectors = docs_vectors.append(temp_avg, ignore_index = True)

docs_vectors.shape
docs_vectors['Sentiment'] = yelp['sentiment']
from sklearn.model_selection import train_test_split

from sklearn.ensemble import AdaBoostClassifier

train_x,test_x,train_y,test_y = train_test_split(docs_vectors.drop('Sentiment',axis=1),

                                                 docs_vectors['Sentiment'],

                                                 test_size=0.2,random_state=100)

train_x.shape ,test_x.shape,train_y.shape,test_y.shape
model = AdaBoostClassifier(n_estimators=800, random_state=100)

model.fit(train_x,train_y)

test_pred = model.predict(test_x)

from sklearn.metrics import accuracy_score

accuracy_score(test_y, test_pred)