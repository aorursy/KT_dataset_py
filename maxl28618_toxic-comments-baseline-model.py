# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot

import seaborn as sns





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

sample = pd.read_csv('../input/sample_submission.csv')
train['toxic'] = train['target'] >= 0.5
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline

text_clf = Pipeline([

    ('vect', CountVectorizer()),

    ('tfidf', TfidfTransformer()),

    ('clf', LogisticRegression()),

])
from sklearn.model_selection import cross_val_score

scores = cross_val_score(text_clf, train['comment_text'], train['toxic'], cv=3)

scores.mean()
text_clf.fit(train['comment_text'], train['toxic'])
predicted = text_clf.predict_proba(test['comment_text'])
submission = pd.DataFrame()

submission['id'] = test['id']

submission['prediction'] = predicted.T[1]
submission.to_csv('submission.csv', index=False)