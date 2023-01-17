# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_messages = pd.read_csv('../input/classifying-20-newsgroups/train.csv', delimiter=',')
train_messages.head()
test_messages = pd.read_csv('../input/classifying-20-newsgroups-test/test.csv', delimiter=',')
test_messages.head()
from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(train_messages.message)
print(X_train_counts.shape)

from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape
from sklearn.naive_bayes import MultinomialNB
NBModel = MultinomialNB().fit(X_train_tfidf, train_messages.topic)
test_messages = pd.read_csv('../input/classifying-20-newsgroups-test/test.csv', delimiter=',')
#test_messages = all_messages.loc[all_messages['dataset'] == 'test']

#It has to use the CountVector and TF-IDF objects created in training which was fit to the training data.
X_test_counts = count_vect.transform(test_messages.message)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)
NB_predicted = NBModel.predict(X_test_tfidf)
#test_messages.insert(4, 'prediction', predicted)
#np.mean(predicted == messages.topic)
test_messages.insert(2, 'topic', NB_predicted)
predictions = test_messages.drop(columns=['message'])
predictions.head()
predictions.to_csv('nb_predictions.csv', sep=',')
print(os.listdir("../working"))
                   
import xgboost

XGB_classifier = xgboost.XGBClassifier()
#XGB_classifier.fit(X_train_tfidf.tocsc(), train_messages.topic)

#XGB_classifier.predict(X_test_tfidf.tocsc())
    