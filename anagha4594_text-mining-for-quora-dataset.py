# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



train=pd.read_csv('../input/train.csv')

train.head()
# we get to see that the data is highly imbalanced 93% 0-class and 6% 1-class

train.target.value_counts()/train.shape[0]*100
# Sincere Word Cloud

from wordcloud import WordCloud



plt.figure(figsize=[10,8])

text=' '.join(train.question_text[train['target']==0])

wc_sincere=WordCloud(background_color='white').generate(text)

plt.imshow(wc_sincere)

# InSincere Word Cloud

plt.figure(figsize=[10,8])

text=' '.join(train.question_text[train['target']==1])

wc_insincere=WordCloud(background_color='white').generate(text)

plt.imshow(wc_insincere)

import nltk

docs=train['question_text'].str.lower().str.replace('[^a-z ]','')

stopwords=nltk.corpus.stopwords.words('english')

stopwords.extend([])# to add custom stopword list to our original list ofn stopwords

stemmer=nltk.stem.PorterStemmer()

# Function to clean the each doc with stopwords and lemmitiation

def clean_doc(doc):

    words=doc.split(" ")

    words_clean=[stemmer.stem(w) for w in words if w not in stopwords]

    return(' '.join(words_clean))# re-joining back the the words to docs

docs_clean=docs.apply(clean_doc)

docs_clean.head()
from sklearn.model_selection import  train_test_split

train1,test1=train_test_split(docs_clean,test_size=0.3,random_state=100)
# document-term matrix creation

from sklearn.feature_extraction.text import CountVectorizer

vectorizer=CountVectorizer(min_df=50)# drops the least appearing terms

vectorizer.fit(train1)# to identify the Features

dtm_train=vectorizer.transform(train1) # converts to matrix

dtm_test=vectorizer.transform(test1) # converts to matrix



# taking Labels

train_y=train.iloc[train1.index]['target']

test_y=train.iloc[test1.index]['target']
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score,classification_report

dtcl=DecisionTreeClassifier(random_state=100,max_depth=10)

dtcl.fit(dtm_train,train_y)

pred=dtcl.predict(dtm_test)

print(classification_report(test_y,pred))
from sklearn.naive_bayes import GaussianNB,MultinomialNB

nbcl=MultinomialNB()

nbcl.fit(dtm_train,train_y)

pred_1=nbcl.predict(dtm_test)

print(classification_report(test_y,pred_1))
# document-term matrix creation

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

vectorizer=TfidfVectorizer(min_df=50)# drops the least appearing terms

vectorizer.fit(train1)# to identify the Features

dtm_train=vectorizer.transform(train1) # converts to matrix

dtm_test=vectorizer.transform(test1) # converts to matrix



from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score,classification_report

dtcl=DecisionTreeClassifier(random_state=100,max_depth=10)

dtcl.fit(dtm_train,train_y)

pred=dtcl.predict(dtm_test)

print(classification_report(test_y,pred))
from sklearn.naive_bayes import GaussianNB,MultinomialNB

nbcl=MultinomialNB()

nbcl.fit(dtm_train,train_y)

pred_1=nbcl.predict(dtm_test)

print(classification_report(test_y,pred_1))