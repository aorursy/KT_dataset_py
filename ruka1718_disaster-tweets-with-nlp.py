# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn import svm

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
train.head()
train.describe()
missing_cols = ['keyword', 'location']
plt.subplot(121)
plt.bar(train[missing_cols].isnull().sum().index, train[missing_cols].isnull().sum().values)
plt.title('Training Dataset')

plt.subplot(122)
plt.bar(test[missing_cols].isnull().sum().index, test[missing_cols].isnull().sum().values)
plt.title('Testing Dataset')
train[train['target'] == 0]['text'].values
vectorizer = CountVectorizer()
train_vec = vectorizer.fit_transform(train['text'])
test_vec = vectorizer.transform(test['text'])
nsamp = train_vec.shape[0]
Iperm = np.random.permutation(nsamp)

xtr = train_vec[Iperm[:], :]
ytr = train['target']
ytr = ytr[Iperm[:]]
clf = svm.SVC(C=2.6, verbose=10)
scores_perm = cross_val_score(clf, xtr, ytr, cv=3, scoring="f1")
scores = cross_val_score(clf, train_vec, train['target'], cv=3, scoring="f1")
print('Scores without permutation',scores)
print('Scores with permutation',scores_perm)
clf.fit(xtr, ytr)
sample_submission = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')
pred = clf.predict(test_vec)
sample_submission['target'] = pred
sample_submission.head()
sample_submission.to_csv("submission.csv", index=False)