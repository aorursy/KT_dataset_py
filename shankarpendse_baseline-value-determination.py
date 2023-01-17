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
sample_submission = pd.read_csv("../input/cat-in-the-dat-ii/sample_submission.csv")

test = pd.read_csv("../input/cat-in-the-dat-ii/test.csv")

train = pd.read_csv("../input/cat-in-the-dat-ii/train.csv")
train.shape
test.shape
train.isnull().sum()
test.isnull().sum()
train.head()
data = train.append(test,sort = False)
data.shape
data.drop(['id','target'], axis = 1, inplace = True)
for col in list(data.columns):

    print(col, data[col].unique())
categorical_features = [col for c, col in enumerate(data.columns) \

                        if not ( np.issubdtype(data.dtypes[c], np.number )  )  ]
categorical_features
data.info()
train.drop(['id','target'], axis = 1, inplace = True)

test.drop(['id'], axis = 1, inplace = True)
for col in list(train.drop(['target'], axis = 1).columns):

    train[col].fillna(train[col].mode()[0], inplace = True)

    test[col].fillna(test[col].mode()[0], inplace = True)
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

for col in categorical_features:

    train[col] = encoder.fit_transform(train[col])

    test[col] = encoder.fit_transform(test[col])
train.isnull().sum()
test.isnull().sum()
train.head()
train.columns
train.info()
test.info()
from sklearn.model_selection import train_test_split
X_train,X_val,y_train,y_val = train_test_split(train.drop(['id','target'], axis = 1), train['target'], test_size = 0.5, random_state = 42, shuffle = True, stratify = train['target'])
X_train.shape
X_val.shape
y_train.shape
y_val.shape
y_train.value_counts()
y_train.shape
from sklearn.linear_model import LogisticRegression

from sklearn import svm
LRC = LogisticRegression(max_iter = 500)

SVMC = svm.SVC()
LRC.fit(X_train,y_train)
LRC.score(X_train,y_train)
LRC.score(X_val,y_val)
test.columns
predictions = LRC.predict(test.drop(['id'],axis = 1))
predictions
train['target'].value_counts()
train.shape
train['target'].value_counts(normalize = True)
test['target'] = LRC.predict_proba(test.drop(['id','target'],axis = 1))
test.columns
test['target']
submit = test[['id','target']]
submit.to_csv("../working/submit.csv", index=False)