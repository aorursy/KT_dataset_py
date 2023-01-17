# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#reading training and test data

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')



train_df.head()

train_df.shape
#convert target columns to boolean

train_df.dtypes
train_df = train_df[['comment_text', 'target']]

train_df.shape
#converting the target value to 0 or 1

train_df.head()

train_df.dtypes
test_df.shape

test_df.head()
#TF-IDF 

from sklearn.feature_extraction.text import TfidfVectorizer

Vectorize = TfidfVectorizer(stop_words='english', token_pattern=r'\w{1,}', max_features=35000)

X = Vectorize.fit_transform(train_df["comment_text"])

y = np.where(train_df['target'] >= 0.5, 1, 0)

test_X = Vectorize.transform(test_df["comment_text"])
X.shape

y.shape

test_X.shape
#split dataset

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#fitting dataset to classification model

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(C=32,dual=False,n_jobs=-1,solver='sag')

clf.fit(X_train, y_train)
#prediction on test or validation data

y_pred = clf.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, auc

cm = confusion_matrix(y_test, y_pred)

cm

print(accuracy_score(y_test, y_pred))
#F-measure

print(classification_report(y_test, y_pred))
y_pred = clf.predict(test_X)
output=pd.DataFrame({'id':test_df['id'],'prediction':y_pred})

output.to_csv('submission.csv', index=False)
output.head()