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
train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

submit = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

train[train.target==0].text.values[1], train[train.target==1].text.values[1]
import matplotlib.pyplot as plt

import seaborn as sns



fig, ax = plt.subplots(figsize = (4, 4))

sns.countplot(train.target)
from sklearn.feature_extraction.text import CountVectorizer



CV = CountVectorizer()

CV.fit_transform(train["text"][:7]).toarray()[0]
train_vectors = CV.fit_transform(train.text)

test_vectors = CV.transform(test.text)
from sklearn import linear_model

clf = linear_model.LogisticRegression()
from sklearn.model_selection import cross_val_score

scores = cross_val_score(clf, train_vectors, train.target, cv = 5, scoring = "f1")

scores, scores.mean()
clf.fit(train_vectors, train.target)
submit.target = clf.predict(test_vectors).astype(np.int32)
submit.to_csv("logistic_regression_pred.csv", index = False)