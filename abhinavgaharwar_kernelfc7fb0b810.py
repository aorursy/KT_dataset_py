# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import gensim

print(os.listdir("../input/embeddings"))



# Any results you write to the current directory are saved as output.
import gensim
from gensim.models import KeyedVectors

path= '../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'

word2vec=KeyedVectors.load_word2vec_format(path,binary=True)
embeddings=gensim.models.KeyedVectors.load_word2vec_format(path,binary=True)
len(word2vec['amazon'])
embeddings['amazon']
from sklearn.metrics.pairwise import cosine_similarity

cosine_similarity([embeddings['camera'],embeddings['quality']])
embeddings.most_similar('hyundai',topn=10)  ## similar word to hyundai
embeddings.doesnt_match(['rahul','sonia','gandhi','sachin'])  ## getting the odd man out
embeddings.most_similar(positive=['king','women'],negative=['man'],topn=1)
url="https://raw.githubusercontent.com/skathirmani/datasets/master/imdb_sentiment.csv"

imdb=pd.read_csv(url)
### document term matrix is created usings weight assign to words

## as normal document term matrix has very high dimension

# reducing the dimension by assigning weight
imdb.head(2)
### it wil fail when word in docement does not match word in google
import nltk

docs_vectors=pd.DataFrame()

stopwords=nltk.corpus.stopwords.words('english')  ### do not do stemming

for doc in imdb['review'].str.lower().str.replace('[^a-z ]',' '):

    words=nltk.word_tokenize(doc)

    words_clean=[word for word in words if word not in stopwords]

    temp=pd.DataFrame()

    for word in words_clean:     ### looping through allthe words in a document

        try:

            word_vec=pd.Series(embeddings[word])

            temp=temp.append(word_vec,ignore_index=True)

        except:

            pass

    temp_avg=temp.mean()        ### calculating the mean(column sum)

    docs_vectors=docs_vectors.append(temp_avg,ignore_index=True)

docs_vectors.shape   
docs_vectors ## vector representation of each word
pd.isnull(docs_vectors).sum().sum()  ##nearly 2 rows is completely missing
docs_vectors['sentiment']=imdb['sentiment']

docs_vectors=docs_vectors.dropna()
docs_vectors.shape
### cant use multinomial naive baise as it contain negative values
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score,classification_report

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,GradientBoostingClassifier
train,test=train_test_split(docs_vectors,test_size=0.2,random_state=100)

train_x=train.drop('sentiment',axis=1)

train_y=train['sentiment']

test_x=test.drop('sentiment',axis=1)

test_y=test['sentiment']

ab_model = AdaBoostClassifier(n_estimators=300,random_state=100)

ab_model.fit(train_x,train_y)

ab_pred =ab_model.predict(test_x)

accuracy_score(test_y,ab_pred)
ab_pred[:5]
gb_model = GradientBoostingClassifier(n_estimators=300,random_state=100)

gb_model.fit(train_x,train_y)

gb_pred =gb_model.predict(test_x)

accuracy_score(test_y,gb_pred)