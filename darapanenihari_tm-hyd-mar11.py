# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

data=pd.read_csv('../input/train.csv')



# Any results you write to the current directory are saved as output.
docs=data['question_text']
from wordcloud import WordCloud

import matplotlib.pyplot as plt

wc=WordCloud(background_color='white').generate(''.join(docs))

plt.imshow(wc)
is_sincere = data['target']==0

not_sincere = data['target']==1
docs0 = data[is_sincere]

docs1 = data[not_sincere]
print(docs0.head())

docs1.head()
dc0 = docs0['question_text']

wc = WordCloud(background_color='white').generate(' '.join(dc0))

plt.imshow(wc)
dc1 = docs1['question_text']

wc1 = WordCloud(background_color='white').generate(' '.join(dc1))

plt.imshow(wc1)
import nltk

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split
stopwords = nltk.corpus.stopwords.words('english')

len(stopwords)
docs = docs.str.lower()

docs.head()
docs.str.replace('[^a-z ]','')
stemmer = nltk.stem.PorterStemmer()



def clean_sentence(doc):

    words = doc.split(' ')

    words_clean = [stemmer.stem(word) for word in words if word not in stopwords]

    doc_clean = ' '.join(words_clean)

    return doc_clean





docs_clean = docs.apply(clean_sentence)
train_x, test_x, train_y, test_y = train_test_split(docs_clean, 

                                                    data['target'],

                                                    test_size = 0.2,

                                                    random_state=100)
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(min_df=50).fit(train_x)

train_x = tfidf.transform(train_x)

test_x = tfidf.transform(test_x)

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

from sklearn.metrics import accuracy_score, f1_score
model_mnb = MultinomialNB().fit(train_x,train_y)

test_pred = model_mnb.predict(test_x)

print(accuracy_score(test_y,test_pred))

print ('F1 score:', f1_score(test_y, test_pred))
model_bnb = BernoulliNB().fit(train_x,train_y)

test_pred = model_bnb.predict(test_x)

print(accuracy_score(test_y,test_pred))

print ('F1 score:', f1_score(test_y, test_pred))
train_x, test_x, train_y, test_y = train_test_split(docs_clean, 

                                                    data['target'],

                                                    test_size = 0.2,

                                                    random_state=100)
vectorizer = CountVectorizer(min_df=50).fit(train_x)

train_x = vectorizer.transform(train_x)

test_x = vectorizer.transform(test_x)
model_bnb = BernoulliNB().fit(train_x,train_y)

test_pred = model_bnb.predict(test_x)

print(accuracy_score(test_y,test_pred))

print ('F1 score:', f1_score(test_y, test_pred))
model_mnb = MultinomialNB().fit(train_x,train_y)

test_pred = model_mnb.predict(test_x)

print(accuracy_score(test_y,test_pred))

print ('F1 score:', f1_score(test_y, test_pred))
test_df = pd.read_csv('../input/test.csv')

test_df.shape
test_docs = test_df['question_text']
test_docs = test_docs.str.lower()

test_docs.head()
test_docs_clean = test_docs.str.replace('[^a-z ]','')

test_docs_clean.head()
test_docs_clean.head()
test_df_clean = vectorizer.transform(test_docs_clean)
test_df_pred = model_mnb.predict(test_df_clean)
test_df_pred
test_df.head()
mysub = pd.DataFrame(test_df['qid'])
test_df_pred = pd.DataFrame(test_df_pred)
final_sub = pd.concat([mysub,test_df_pred],axis=1)
final_sub.columns=['qid','prediction']
final_sub
from pandas import DataFrame
final_sub.to_csv("submission.csv",index=False)