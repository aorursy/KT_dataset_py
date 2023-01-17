# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
train_data = pd.read_csv('../input/train.tsv',delimiter='\t')
test_data = pd.read_csv('../input/test.tsv',delimiter='\t')

train_data.head()

# Any results you write to the current directory are saved as output.
print(len(train_data), len(test_data))
# The ratio of test data to train data is 0.3 to 0.7, which is fairly common 
# First we will try logistic regression

# We will re-structure sentiment into positive and negative, by removing the neutral sentiments (with a score of 3) and 
# saying a review has a positive sentiment if it has a score greater than 3, and a negative sentiment if it has a score
# smaller than 3.

train_data_binary = train_data[train_data['Sentiment'] != 3]

train_data_binary['Positively Rated'] = np.where(train_data_binary['Sentiment'] > 3, 1, 0)

# We will split our training data into a train set and a test set, since our test data 'test.tsv' is unlabeled.
# This way we can measure how well our model is predicting the Sentiment.

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_data_binary['Phrase'], 
                                                    train_data_binary['Positively Rated'],
                                                    test_size=0.3, 
                                                    random_state=0)
len(X_train)
from sklearn.feature_extraction.text import CountVectorizer

# Fit the CountVectorizer to the training data
vect = CountVectorizer().fit(X_train)
print(len(vect.get_feature_names()))
# We obtain over 15000 features this way. We can reduce the number of features by restricting to words that appear a minimum
# number of times.
vect1 = CountVectorizer(min_df=10).fit(X_train)
print(len(vect1.get_feature_names()))

# We apply logistic regression to the data

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

X_train_vectorized = vect.transform(X_train)

model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

predictions = model.predict(vect.transform(X_test))

print('AUC: ', roc_auc_score(y_test, predictions))
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

X_train_vectorized = vect1.transform(X_train)

model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

predictions = model.predict(vect1.transform(X_test))

print('AUC: ', roc_auc_score(y_test, predictions))
# We see that by requiring that the frequencies are at least 10, we reduce the number of features from 15124 to 6674
# without reducing our score significantly.
# We try adding n-grams for n=2,3
vect = CountVectorizer(min_df=10, ngram_range=(1,3)).fit(X_train)

X_train_vectorized = vect.transform(X_train)

len(vect.get_feature_names())
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

X_train_vectorized = vect.transform(X_train)

model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

predictions = model.predict(vect.transform(X_test))

print('AUC: ', roc_auc_score(y_test, predictions))
# We try with tfidf vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

vect = TfidfVectorizer(min_df=5).fit(X_train)
len(vect.get_feature_names())
X_train_vectorized = vect.transform(X_train)

model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

predictions = model.predict(vect.transform(X_test))

print('AUC: ', roc_auc_score(y_test, predictions))
# tfidf doesn't seem to be working better
#  we now try naive bayes multinomial classifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_data['Phrase'], 
                                                    train_data['Sentiment'],
                                                    test_size=0.3, 
                                                    random_state=0)
from sklearn import naive_bayes
from sklearn import metrics

vect = CountVectorizer(min_df=10, ngram_range=(1,3)).fit(X_train)

X_train_vectorized = vect.transform(X_train)
NB= naive_bayes.MultinomialNB()
NB.fit(X_train_vectorized,y_train)
predictions = NB.predict(vect.transform(X_test))
metrics.f1_score(y_test,predictions, average= 'micro')

