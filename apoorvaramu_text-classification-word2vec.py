# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/embeddings/GoogleNews-vectors-negative300"))

import gensim

# Any results you write to the current directory are saved as output.
import nltk
url="../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin"

embeddings = gensim.models.KeyedVectors.load_word2vec_format(url, binary=True)
# embeddings.most_similar('hyundai', topn=10)

# embeddings.doesnt_match(['rahul','gandhi','sonia','modi','sachin'])

# embeddings.most_similar(positive=['king','woman'],negative=['man'])
url='https://bit.ly/2U7mLmC'

yelp = pd.read_csv(url,sep="\t",delimiter="\t")

yelp.reset_index(level=0, inplace=True)

yelp.columns=["review","sentiment"]

yelp.head()
yelp.loc[1,'review']
embeddings['Not'].shape
docs_vectors = pd.DataFrame()

import nltk

stopwords=nltk.corpus.stopwords.words('english')

for doc in yelp['review'].str.lower().str.replace('[^a-z ]', ''):

    temp = pd.DataFrame()

    for word in doc.split(' '):

        if word not in stopwords:

            try:

                word_vec = embeddings[word]

                temp = temp.append(pd.Series(word_vec), ignore_index=True)

            except:

                pass

    doc_vector = temp.mean()

    docs_vectors = docs_vectors.append(doc_vector, ignore_index=True)

docs_vectors.shape

                                        

    
docs_vectors
pd.isnull(docs_vectors).sum().sum()
docs_vectors['sentiment'] = yelp['sentiment']

docs_vectors = docs_vectors.dropna()
from sklearn.model_selection import train_test_split

from sklearn.ensemble import AdaBoostClassifier

train_x,test_x, train_y,test_y = train_test_split(docs_vectors.drop('sentiment',axis=1),

                                                  docs_vectors['sentiment'],

                                                  test_size = 0.2,

                                                  random_state = 100)

train_x.shape, test_x.shape, train_y.shape, test_y.shape

                                                  

model = AdaBoostClassifier(n_estimators=500, random_state=1)

model.fit(train_x,train_y)

test_pred = model.predict(test_x)

from sklearn.metrics import accuracy_score

accuracy_score(test_y, test_pred)
 