# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings

warnings.filterwarnings("ignore")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('../input/train-data-2.csv')

test_data = pd.read_csv('../input/test-data-2.csv')

submission = pd.read_csv('../input/sample-submission-2.csv')
train_data.head()
test_data.head()
train_data.isnull().sum()
train_data = train_data.fillna('')

test_data = test_data.fillna('')
from sklearn.feature_extraction.text import TfidfVectorizer



vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(train_data['review_sangat_singkat'])

y = train_data['rating']
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import StratifiedKFold



def get_skfold():

    return StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
score = cross_val_score(LogisticRegression(C=3), X, y, scoring='neg_mean_squared_error', cv=get_skfold())
rmse = [np.sqrt(-1 * x) for x in score]

print(np.mean(rmse))
import itertools

def remove_repeating_characters(text):

    return ''.join(''.join(s)[:1] for _, s in itertools.groupby(text))
train_data['review_sangat_singkat'] = train_data['review_sangat_singkat'].apply(remove_repeating_characters)

test_data['review_sangat_singkat'] = test_data['review_sangat_singkat'].apply(remove_repeating_characters)
X = vectorizer.fit_transform(train_data['review_sangat_singkat'])

y = train_data['rating']
score = cross_val_score(LogisticRegression(C=3), X, y, scoring='neg_mean_squared_error', cv=get_skfold())
rmse = [np.sqrt(-1 * x) for x in score]

print(np.mean(rmse))
clf = LogisticRegression(C=3)

clf.fit(X,y)

result = clf.predict(vectorizer.transform(test_data['review_sangat_singkat']))
submission.head()
submission.rating = result
submission.head()
submission.rating.value_counts()
submission.to_csv("submission.csv", index=False)