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
import pandas as pd

import nltk

from sklearn import metrics

from sklearn import preprocessing

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import MultinomialNB

from sklearn import svm

from sklearn import metrics
train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv', header=0)

test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv', header=0)



train_data = train['text']
count_vectorizer = CountVectorizer()
raw1=train_data.str.replace(r'\b[\w\-.]+?@\w+?\.\w{2,4}\b', 'emailaddr')

raw1=raw1.str.replace(r'(http[s]?\S+)|(\w+\.[A-Za-z]{2,4}\S*)', 'httpaddr')

raw1=raw1.str.replace(r'£|\$', 'moneysymb')  

raw1=raw1.str.replace(r'\b(\+\d{1,2}\s)?\d?[\-(.]?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b', 'phonenumbr')  

raw1=raw1.str.replace(r'\d+(\.\d+)?', 'numbr')

raw1=raw1.str.replace(r'[^\w\d\s]', ' ')

raw1=raw1.str.replace(r'\s+', ' ')

raw1=raw1.str.replace(r'^\s+|\s+?$', '')

raw1=raw1.str.lower()



#Remove stop words

stop_words=nltk.corpus.stopwords.words('english')

raw1=raw1.apply(lambda x: ' '.join(term for term in x.split() if term not in set(stop_words)))



#Stemming

p=nltk.PorterStemmer()

raw1=raw1.apply(lambda x: ' '.join(p.stem(term) for term in x.split()))







train_vectors = count_vectorizer.fit_transform(train['text'])

test_vectors = count_vectorizer.transform(test['text'])
NB = MultinomialNB()

NB.fit(train_vectors, train['target'])
sample_submission = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')

sample_submission["target"] = NB.predict(test_vectors)
sample_submission.to_csv('submit.csv')