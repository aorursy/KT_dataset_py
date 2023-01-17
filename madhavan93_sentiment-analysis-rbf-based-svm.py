# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import seaborn as sns

import matplotlib.pyplot as plt

sns.set(color_codes=True)



data_train = pd.read_csv("../input/train.tsv", sep="\t")

data_test = pd.read_csv("../input/test.tsv", sep="\t")

sub_file = pd.read_csv('../input/sampleSubmission.csv',sep=',')



data_train.shape
data_test.shape
sub_file.shape
dist = data_train.groupby(["Sentiment"]).size()

dist = dist / dist.sum()

fig, ax = plt.subplots(figsize=(12,8))

sns.barplot(dist.keys(), dist.values);
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import SVC

from sklearn.pipeline import Pipeline

from sklearn.model_selection import StratifiedKFold



svc = SVC(

    C=1.0,

    tol=1e-05, 

    verbose=0,

    max_iter=7500,

    gamma=2

)



tfidf = CountVectorizer(

    input='content',

    encoding='utf-8',

    decode_error='strict',

    strip_accents=None,

    lowercase=True,

    preprocessor=None,

    tokenizer=None,

    stop_words=None,

    token_pattern=r"(?u)\b\w\w+\b",

    ngram_range=(1, 1),

    analyzer='word',

    max_df=1.0,

    min_df=1,

    max_features=None,

    vocabulary=None,

    binary=False,

    dtype=np.int64

)



pipeline = Pipeline([

    ('tfidf', tfidf),

    ('svc', svc),

])



n_split = 3

skf = StratifiedKFold(n_splits=n_split)



X = data_train.Phrase

y = data_train.Sentiment



total_train_score = 0.0

total_test_score = 0.0



for train, test in skf.split(X, y):

    pipeline.fit(X[train], y[train])

    train_score = pipeline.score(X[train], y[train])

    test_score = pipeline.score(X[test], y[test])

    

    total_train_score += train_score

    total_test_score += test_score

    

    print("Train = {}, Test = {}".format(train_score, test_score))

    

average_train_score = total_train_score / n_split

average_test_score = total_test_score / n_split

print("Average training accuracy: {}, Average testing accuracy: {}".format(average_train_score, average_test_score))
test_predictions = pipeline.predict(data_test.Phrase)

print(test_predictions.shape)



sub_file['Sentiment'] = test_predictions

sub_file.to_csv('Submission.csv',index=False)