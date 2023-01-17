# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import KFold

from sklearn.ensemble import RandomForestClassifier

from sklearn.decomposition import PCA

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import gc



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

sample = pd.read_csv("/kaggle/input/digit-recognizer/sample_submission.csv")

y = train["label"]

X = train.drop(columns="label")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
def DecisionTree(X_train, y_train, X_test, y_test):

    model = DecisionTreeClassifier()

    model.fit(X_train, y_train)

    return  model.score(X_test, y_test)



def RandomForest(X_train, y_train, X_test, y_test):

    model = RandomForestClassifier(n_estimators = 100, max_depth = 10)

    model.fit(X_train, y_train)

    return model.score(X_test, y_test)

model = RandomForestClassifier(n_estimators = 100, max_depth = 10)

model.fit(X_train, y_train)

res = model.predict(test)
submission = pd.concat([pd.Series(range(1,28001), name = "ImageId"), pd.Series(res, name="Label")], axis = 1)

submission.to_csv("RandomForest.csv",index=False)