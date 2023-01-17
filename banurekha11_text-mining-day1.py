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
data = pd.read_csv('../input/train.csv')
data.head()
#The data is imbalanced

data['target'].value_counts()/len(data)
import nltk

stopwords = nltk.corpus.stopwords.words('english')
from wordcloud import WordCloud

import matplotlib.pyplot as plt



plt.figure(figsize=(20,10))

wc = WordCloud(background_color='yellow',stopwords=stopwords).generate(''.join(data[data['target']==0]['question_text']))

plt.imshow(wc)
plt.figure(figsize=(20,10))

wc = WordCloud(background_color='yellow',stopwords=stopwords).generate(''.join(data[data['target']==1]['question_text']))

plt.imshow(wc)
data['question_text'].head()
docs_clean = data['question_text'].str.lower().str.replace('[^a-z ]','')

stemmer = nltk.stem.PorterStemmer()

def clean_sentence(doc):

    words = doc.split(' ')

    words_clean = [stemmer.stem(word) for word in words if word not in stopwords]

    doc_clean = ' '.join(words_clean)

    return doc_clean

docs_clean = docs_clean.apply(clean_sentence)

docs_clean.head()
len(docs_clean)
data_clean = data.copy()

data_clean['question_text']=docs_clean

data_clean.tail()
from sklearn.model_selection import train_test_split

X=data_clean['question_text']

y=data_clean['target']

X_train,X_test, y_train, y_test =train_test_split(X,y,test_size=0.3,random_state=100)
from sklearn.feature_extraction.text import CountVectorizer



vectorizer = CountVectorizer(min_df=50)

vectorizer.fit(X_train)

dtm_train = vectorizer.transform(X_train)

dtm_test = vectorizer.transform(X_test)



features = vectorizer.get_feature_names()
from sklearn.naive_bayes import GaussianNB,MultinomialNB

from sklearn.metrics import f1_score

MNB_clfr = MultinomialNB()

MNB_clfr.fit(dtm_train,y_train)

predictions = MNB_clfr.predict(dtm_test)



f1_score(y_test,predictions)