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
# __Author__: Judy Wang
# __Project__: Kaggle: What's Cooking?
# __Created__: 26 Jul 2018

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB


print(50 * '-')
print('LOAD DATA') ##################
# Create data frames.
train_full = pd.read_json('../input/train.json')
test = pd.read_json('../input/test.json')
submission = pd.read_csv('../input/sample_submission.csv')

# Use portion of full training set as training set to train model;
# use rest of full training set as holdout to adjust model parameters.
# Set to value between 0 and 1, noninclusive.
portion_train = 0.5

# Sizes of data sets ()
size_train_full = len(train_full)
size_train = int(size_train_full * portion_train)
size_holdout = size_train_full - size_train
size_test = len(test)
print('Size of full training set: {} (training {}, holdout {})'.format(size_train_full, size_train, size_holdout))
print('Size of test set: {}'.format(size_test))

# Divide training set into (effective) training and holdout sets.
train = train_full.iloc[:size_train]
holdout = train_full.iloc[size_train:]


print(50 * '-')
print('TRAINING') ##################
# Process training set.
corpus_train = map(lambda x: ', '.join(x), train['ingredients'].values)
targets_train = train['cuisine'].values

# Initialize vectorizer to detect presence of ingredient
# (parameters set to ignore quantity and to include single letter words to account for potential brand names).
vectorizer = CountVectorizer(binary=True, token_pattern='(?u)\\b\\w+\\b')

# Construct document-term matrix for training set.
dtm_train = vectorizer.fit_transform(corpus_train).toarray().tolist()

# Create Bernoull Classifier
# (parameters adjusted using entire training set as holdout).
# Alpha values: minimum 1.0e-10, best 0.00015 and 0.00016
model = BernoulliNB(alpha=0.00015, binarize=None, fit_prior=True, class_prior=None)
model.fit(dtm_train, targets_train)

# Create list of predictions for training set.
predictions_train = model.predict(dtm_train)


print(50 * '-')
print('HOLDOUT') ##################
# Process holdout set
corpus_holdout = map(lambda x: ', '.join(x), holdout['ingredients'].values)
targets_holdout = holdout['cuisine'].values

# Construct document-term matrix for holdout set.
dtm_holdout = vectorizer.transform(corpus_holdout)

# Create list of predictions for holdout set.
predictions_holdout = model.predict(dtm_holdout)


print(50 * '-')
print('TEST') ##################
# Process test set.
corpus_test = map(lambda x: ', '.join(x), test['ingredients'].values)

# Construct document-term matrix for test set.
dtm_test = vectorizer.transform(corpus_test)

# Create list of predictions for test set.
predictions_test = model.predict(dtm_test)


print(50 * '-')
print('RESULTS') ##################

# Percent accuracies
accuracy_train = len([i for i in range(size_train) if predictions_train[i] == targets_train[i]]) / size_train
accuracy_holdout = len([i for i in range(size_holdout) if predictions_holdout[i] == targets_holdout[i]]) / size_holdout
print('Training set percent accuracy: {:.9%}'.format(accuracy_train))
print('Holdout set percent accuracy: {:.9%}'.format(accuracy_holdout))
print('Combined accuracy of training and holdout: {:.9%}'.format(accuracy_train + accuracy_holdout))

# # Create CSV file for submission.
# path = '../input/submission.csv'
# data = {'id': test['id'].values, 'cuisine': predictions_test}
# columns = ['id', 'cuisine']
# pd.DataFrame(data)[columns].to_csv(path, sep=',', index=False)

# Check contents in kernel directory.
# print(os.listdir('../input'))