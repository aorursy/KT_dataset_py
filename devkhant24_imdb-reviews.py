# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import pickle

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

import string

import nltk

from random import sample

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, FeatureHasher

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score, f1_score



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
stemmer = PorterStemmer() 

lemmatizer = WordNetLemmatizer()



gnb = GaussianNB()

bnb = BernoulliNB()

mnb = MultinomialNB()

svm = LinearSVC(max_iter=4000, random_state=0)
data = pd.read_csv('/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv')
data.head()
data['sentiment'].value_counts()
dummies = pd.get_dummies(data['sentiment'], drop_first=True)

data = pd.concat([data,dummies], axis=1)

data = data.drop(['sentiment'],axis=1)
train = data.sample(n= 40000)

train
test = data.sample(n = 3000)

test
def clear_text(df):

    all_reviews = []

    grp = df['review'].values.tolist()

    for sent in grp:

        sent = sent.lower()

        pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

        sent = pattern.sub('', sent)

        sent = re.sub(r'[,.\"!@#$%^&*(){}?/;`~:<>+=-]', '', sent)

        tokens = nltk.word_tokenize(sent)

        table = str.maketrans('', '', string.punctuation)

        stripped = [w.translate(table) for w in tokens]

        words = [word for word in stripped if word.isalpha()]

        stop_words = set(stopwords.words('english'))

        stop_words.discard('not')

        words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

        words = ' '.join(words)

        all_reviews.append(words)

    return all_reviews



train_reviews = clear_text(train)
cv = TfidfVectorizer(min_df = 4)

Xtr = cv.fit_transform(train_reviews).toarray()

Ytr = train['positive']
pickle.dump(cv,open('cv-transform.pkl','wb'))
# xtrain, xtest, ytrain, ytest = train_test_split(Xtr, Ytr, test_size=0.2, random_state=0)
# score = cross_val_score(bnb, xtrain, ytrain, cv=2)

# score.mean()
# 0.85
svm.fit(Xtr, Ytr)
# pred = svm.predict(xtest)

# print(accuracy_score(ytest, pred))
# test_reviews = clear_text(test)



# Xte = cv.transform(test_reviews).toarray()

# Yte = test['positive']
# pre = svm.predict(Xte)

# print(accuracy_score(Yte,pre))


pickle.dump(svm,open('svm_model.pkl','wb'))