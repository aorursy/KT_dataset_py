# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import nltk 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import gensim

print(os.listdir("../input/embeddings/GoogleNews-vectors-negative300/"))

# Any results you write to the current directory are saved as output.





# Any results you write to the current directory are saved as output.
url = "../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin"

embeddings = gensim.models.KeyedVectors.load_word2vec_format(url, binary=True)
url1 = 'https://raw.githubusercontent.com/skathirmani/datasets/master/yelp_labelled.csv'

url1
yelp = pd.read_csv(url1,sep = '\t',names=['review','sentiment'])

yelp = yelp.drop(yelp.index[0])

yelp.head()
docs_vectors = pd.DataFrame()

stopwords = nltk.corpus.stopwords.words('english')

for doc in yelp['review'].str.lower().str.replace('[^a-z ]', ''):

    temp = pd.DataFrame()

    for word in doc.split(' '): #for loop for goin through all words in the document

        if word not in stopwords:

            try:

                word_vec = embeddings[word]

                temp = temp.append(pd.Series(word_vec), ignore_index=True)

            except:

                pass

    doc_vector = temp.mean() #column average

    docs_vectors = docs_vectors.append(doc_vector, ignore_index=True)

docs_vectors.shape
pd.isnull(docs_vectors).sum().sum()

docs_vectors['sentiment'] = yelp['sentiment']

docs_vectors = docs_vectors.dropna()
from sklearn.model_selection import train_test_split

from sklearn.ensemble import AdaBoostClassifier

train_x,test_x,train_y,test_y = train_test_split(docs_vectors.drop('sentiment',axis=1),

                                         docs_vectors['sentiment'],

                                         test_size=0.2,

                                         random_state=100)

train_x.shape,test_x.shape,train_y.shape,test_y.shape
model=AdaBoostClassifier(n_estimators=800,random_state=1)

model.fit(train_x,train_y)

test_pred = model.predict(test_x)

from sklearn.metrics import accuracy_score

accuracy_score(test_y,test_pred)