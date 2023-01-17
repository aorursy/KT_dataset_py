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
test_csv = pd.read_csv('../input/imdb-movie-reviews-dataset/test_data (1).csv')

train_csv = pd.read_csv('../input/imdb-movie-reviews-dataset/train_data (1).csv')



train_X = train_csv['0']   # '0' corresponds to Texts/Reviews

train_y = train_csv['1']   # '1' corresponds to Label (1 - positive and 0 - negative)

test_X = test_csv['0']

test_y = test_csv['1']
from time import time

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.naive_bayes import BernoulliNB, MultinomialNB

from sklearn import metrics
t = time()

tfidf_vectorizer = TfidfVectorizer()

X_train_tfidf = tfidf_vectorizer.fit_transform(train_X)

duration = time() - t

print("Time taken to extract features from training data : %f seconds" % (duration))

print("n_samples: %d, n_features: %d" % X_train_tfidf.shape)
print(X_train_tfidf.toarray())
t = time()

X_test_tfidf = tfidf_vectorizer.transform(test_X)

duration = time() - t

print("Time taken to extract features from test data : %f seconds" % (duration))

print("n_samples: %d, n_features: %d" % X_test_tfidf.shape)
# build naive bayes classification model

t = time()

naive_bayes_classifier = MultinomialNB()



naive_bayes_classifier.fit(X_train_tfidf, train_y)



training_time = time() - t

print("train time: %0.3fs" % training_time)
# predict the new document from the testing dataset

t = time()

y_pred = naive_bayes_classifier.predict(X_test_tfidf)



test_time = time() - t

print("test time:  %0.3fs" % test_time)



# compute the performance measures

score1 = metrics.accuracy_score(test_y, y_pred)

print("accuracy:   %0.3f" % score1)



print(metrics.classification_report(test_y, y_pred,

                                            target_names=['Positive', 'Negative']))



print("confusion matrix:")

print(metrics.confusion_matrix(test_y, y_pred))



print('------------------------------')



#------------------------------------------------------------------------