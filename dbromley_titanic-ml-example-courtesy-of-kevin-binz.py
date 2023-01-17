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
### Simple Scenario
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")

X_features = ["Pclass"]
y_features = ["Survived"]

X_train = train_df[X_features].copy()
X_test = test_df[X_features].copy()
y_train = train_df[y_features].copy()
print("{}, {}, {}".format(X_train.shape, y_train.shape, X_test.shape))

clf = LogisticRegression()
scores = cross_val_score(clf, X_train, y_train.values.ravel(), cv=10, scoring='accuracy')
print(scores.mean())  # Score: 67.9%

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import Imputer

train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")

all_cols = ["PassengerId", "Survived", "Name", "Sex", "Age", "SibSp", "Parch",
            "Ticket", "Fare", "Cabin", "Embarked", "Pclass", "Survived"]
X_features = ["Pclass", "Age"]
y_features = ["Survived"]

X_train = train_df[X_features].copy()
X_test = test_df[X_features].copy()
y_train = train_df[y_features].copy()
print("Shapes: {}, {}, {}".format(X_train.shape, y_train.shape, X_test.shape))

print("=== X_train nulls ===\n {}".format(X_train.isnull().sum()))
print("=== X_test nulls ===\n {}".format(X_test.isnull().sum()))

### Imputation via pandas

# X_train["Age"].fillna(X_train["Age"].mean(), inplace=True)
# X_test["Age"].fillna(X_test["Age"].mean(), inplace=True)

### Imputation via sklearn

imp = Imputer(strategy='mean', axis=0)
X_train["Age"] = imp.fit_transform(X_train["Age"].values.reshape(-1, 1))
X_test["Age"] = imp.fit_transform(X_test["Age"].values.reshape(-1, 1))

print("=== X_train nulls ===\n {}".format(X_train.isnull().sum()))
print("=== X_test nulls ===\n {}".format(X_test.isnull().sum()))

clf = LogisticRegression()
scores = cross_val_score(clf, X_train, y_train.values.ravel(), cv=10, scoring='accuracy')
print(scores.mean())  # Score: 70.1%

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelBinarizer

train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")

all_cols = ["PassengerId", "Survived", "Name", "Sex", "Age", "SibSp", "Parch",
            "Ticket", "Fare", "Cabin", "Embarked", "Pclass", "Survived"]

X_features = ["Pclass", "Age", "Sex"]
y_features = ["Survived"]

X_train = train_df[X_features].copy()
X_test = test_df[X_features].copy()
y_train = train_df[y_features].copy()
print("Shapes: {}, {}, {}".format(X_train.shape, y_train.shape, X_test.shape))

''' Imputation '''

print("=== X_train nulls ===\n {}".format(X_train.isnull().sum()))
print("=== X_test nulls ===\n {}".format(X_test.isnull().sum()))

X_train["Age"].fillna(X_train["Age"].mean(), inplace=True)
X_test["Age"].fillna(X_test["Age"].mean(), inplace=True)

print("=== X_train nulls ===\n {}".format(X_train.isnull().sum()))
print("=== X_test nulls ===\n {}".format(X_test.isnull().sum()))

''' Encoding '''

print(X_train.head())

lb = LabelBinarizer()
X_train["Sex"] = lb.fit_transform(X_train["Sex"])
X_test["Sex"] = lb.fit_transform(X_test["Sex"])

print(X_train.head())

''' Training '''

clf = LogisticRegression()
scores = cross_val_score(clf, X_train, y_train.values.ravel(), cv=10, scoring='accuracy')
print(scores.mean())  # Score: 79.7%

