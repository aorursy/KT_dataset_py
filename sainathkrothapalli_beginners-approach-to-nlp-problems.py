# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train=pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

train.head()
train.shape
train.info()
train['target'].value_counts()
import seaborn as sns

sns.countplot(train['target'])
(train['target'].value_counts()/train.shape[0])*100
train.isnull().sum()
train['keyword'].fillna('no_keyword',inplace=True)

train['location'].fillna('no_location',inplace=True)
train.isnull().sum().sum()
train['location'].value_counts()[:10]
train['location'].value_counts()[:20].plot(kind='bar')
train['keyword'].value_counts()[:10]
train['keyword'].value_counts()[:20].plot(kind='bar')
#List of stopwords in english

import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords

print(stopwords.words('english'))
import re

from nltk.stem.porter import PorterStemmer
corpus = []

for i in range(0,train.shape[0]):

  review = re.sub('[^a-zA-Z]', ' ',train['text'][i])

  review = review.lower()

  review = review.split()

  ps = PorterStemmer()

  all_stopwords = stopwords.words('english')

  review = [ps.stem(word) for word in review if not word in set(all_stopwords)]

  review = ' '.join(review)

  corpus.append(review)
corpus[:5]
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features = 1000)

X = cv.fit_transform(corpus).toarray()

X[:10]
from sklearn.model_selection import train_test_split

y=train.iloc[:,-1].values

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
from sklearn.naive_bayes import MultinomialNB

classifier = MultinomialNB()

classifier.fit(X_train, Y_train)
from sklearn.metrics import confusion_matrix, accuracy_score

prediction=classifier.predict(X_test)

cm = confusion_matrix(Y_test,prediction)

print(cm)

accuracy_score(Y_test,prediction)