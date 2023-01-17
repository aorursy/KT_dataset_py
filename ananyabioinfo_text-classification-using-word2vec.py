# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import nltk

import gensim

print(os.listdir("../input/embeddings/GoogleNews-vectors-negative300/"))



# Any results you write to the current directory are saved as output.
url = "../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin"

embeddings = gensim.models.KeyedVectors.load_word2vec_format(url, binary=True)
#embeddings.most_similar('camera', topn = 5)

#embeddings.doesnt_match(['apple','banana','flower'])

#embeddings.most_similar(positive = ['king','woman'], negative = ['man'])
url = 'https://bit.ly/2CdYYuf'

yelp = pd.read_csv(url, sep='\t', header = None)

yelp.head()
yelp.rename(columns={0:'Reviews', 1:'Sentiment'}, inplace=True)

yelp.head()
yelp.loc[0, 'Reviews']


docs_vectors = pd.DataFrame() # creating empty final dataframe

stopwords = nltk.corpus.stopwords.words('english') # removing stop words

for doc in yelp['Reviews'].str.lower().str.replace('[^a-z ]', ''): # looping through each document and cleaning it

    temp = pd.DataFrame()  # creating a temporary dataframe(store value for 1st doc & for 2nd doc remove the details of 1st & proced through 2nd and so on..)

    for word in doc.split(' '): # looping through each word of a single document and spliting through space

        if word not in stopwords: # if word is not present in stopwords then (try)

            try:

                word_vec = embeddings[word] # if word is present in embeddings(goole provides weights associate with words(300)) then proceed

                temp = temp.append(pd.Series(word_vec), ignore_index = True) # if word is present then append it to temporary dataframe

            except:

                pass

    doc_vector = temp.mean() # take the average of each column(w0, w1, w2,........w300)

    docs_vectors = docs_vectors.append(doc_vector, ignore_index = True) # append each document value to the final dataframe

docs_vectors.shape
pd.isnull(docs_vectors).sum().sum()
docs_vectors['Sentiment'] = yelp['Sentiment']

docs_vectors = docs_vectors.dropna()
from sklearn.model_selection import train_test_split

from sklearn.ensemble import AdaBoostClassifier



train_x, test_x, train_y, test_y = train_test_split(docs_vectors.drop('Sentiment', axis = 1),

                                                   docs_vectors['Sentiment'],

                                                   test_size = 0.2,

                                                   random_state = 1)

train_x.shape, train_y.shape, test_x.shape, test_y.shape
model = AdaBoostClassifier(n_estimators=800, random_state = 1)

model.fit(train_x, train_y)

test_pred = model.predict(test_x)

from sklearn.metrics import accuracy_score

accuracy_score(test_y, test_pred)