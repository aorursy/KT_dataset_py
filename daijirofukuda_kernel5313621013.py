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
import pandas as pd

from pandas import DataFrame, Series

import numpy as np

from datetime import datetime, date

import seaborn as sns

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve, auc, accuracy_score

from sklearn.model_selection import GridSearchCV
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")



train_data["Sex"] = train_data["Sex"].mask(train_data["Sex"] == "male", 0)

train_data["Sex"] = train_data["Sex"].mask(train_data["Sex"] == "female", 1)

train_data.Age = train_data.Age.fillna(value=train_data.Age.median())

train_data = train_data.fillna(value=0)



test_data["Sex"] = test_data["Sex"].mask(test_data["Sex"] == "male", 0)

test_data["Sex"] = test_data["Sex"].mask(test_data["Sex"] == "female", 1)

test_data.Age = test_data.Age.fillna(value=test_data.Age.median())

test_data = test_data.fillna(value=0)



train_X = train_data.drop(["Survived", "PassengerId", "Name", "Ticket", "Cabin", "Embarked"], axis=1)

train_y = train_data.Survived



test_X = test_data.drop(["PassengerId", "Name", "Ticket", "Cabin", "Embarked"], axis=1)

# test_y = test_data.Survived



clf = RandomForestClassifier(random_state=0, min_samples_split=20, n_estimators=50, max_depth=10)

# clf = RandomForestClassifier(random_state=0)



clf = clf.fit(train_X, train_y)

pred = clf.predict(test_X)

# fpr, tpr, thresholds = roc_curve(test_y, pred, pos_label=1)

# auc(fpr, tpr)

pred
submission = DataFrame([test_data.PassengerId, pred], index=["PassengerId", "Survived"]).T

submission
submission.to_csv("Submission.csv", index=False)