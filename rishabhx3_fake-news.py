import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

from string import punctuation



from sklearn.naive_bayes import MultinomialNB

from sklearn import metrics

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_df = pd.read_csv('/kaggle/input/fake-news/train.csv')

test_df = pd.read_csv('/kaggle/input/fake-news/test.csv')
train_df.head(5)
test_df.head(5)
train_df.isna().sum()
test_df.isna().sum()
train_df.fillna("missing",inplace = True)

test_df.fillna("missing",inplace = True)
print("Train Shape: ",train_df.shape)

print("Test Shape: ",test_df.shape)
train_df['text'] = train_df['text'] + " " + train_df['title'] + " " + train_df['author']

del train_df['title']

del train_df['author']

del train_df['id']
test_df['text'] = test_df['text'] + " " + test_df['title'] + " " + test_df['author']

del test_df['title']

del test_df['author']

del test_df['id']
print("Train Shape: ",train_df.shape)

print("Test Shape: ",test_df.shape)
stop = set(stopwords.words('english'))

pnc = list(punctuation)

stop.update(pnc)
stemmer = PorterStemmer()

def stem_text(text):

    final_text = []

    for i in text.split():

        if i.strip().lower() not in stop:

            word = stemmer.stem(i.strip())

            final_text.append(word)

    return " ".join(final_text)
train_df['text'] = (train_df['text'].astype(str)).apply(stem_text)

print('Train done!')

test_df['text'] = (test_df['text'].astype(str)).apply(stem_text)

print('Test done!')
print(train_df.shape)

print(test_df.shape)
X = train_df['text']

y = train_df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)
test_df = test_df['text']
print('X_train shape: ',X_train.shape)

print('y_train shape: ',y_train.shape)

print('X_test shape: ',X_test.shape)

print('y_test shape: ',y_test.shape)

print('test_df shape: ',test_df.shape)
cv = CountVectorizer(min_df=0,max_df=1,ngram_range=(1,3))



cv_train = cv.fit_transform(X_train)

cv_test = cv.transform(X_test)

cv_test_df = cv.transform(test_df)



print('Train data shape: ',cv_train.shape)

print('Validation data shape: ',cv_test.shape)

print('Test data shape: ',cv_test_df.shape)
nb = MultinomialNB()
nb.fit(cv_train, y_train)
pred_nb = nb.predict(cv_test)
score = metrics.accuracy_score(y_test, pred_nb)

print(score)
pred_nbs = nb.predict(cv_test_df)

sub_df = pd.read_csv('../input/fake-news/submit.csv')

sub_df['label'] = pred_nbs

sub_df.to_csv('sample_sub_nb.csv', index = False)