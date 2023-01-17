# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
print("Shape of train data:" + str(train.shape))
train.head()
train.describe()
sns.countplot(x='Cover_Type', data=train)
from sklearn.model_selection import train_test_split

Y = train["Cover_Type"]
X = train.drop(["Cover_Type"], axis=1)

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.30, random_state=42)
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf.fit(X_train, Y_train)
from sklearn.metrics import f1_score

Y_pred = clf.predict(X_val)
print("F1-score: " + str(f1_score(Y_val, Y_pred, average="micro")))
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()
clf.fit(X_train, Y_train)

Y_pred = clf.predict(X_val)
print("F1-score: " + str(f1_score(Y_val, Y_pred, average="micro")))
test = pd.read_csv("../input/test.csv")

X_test = test.drop(["id"], axis=1)
Y_test = clf.predict(X_test)

submission = pd.DataFrame({
    "id": test["id"],
    "Cover_Type": Y_test
})
submission.to_csv("submission.csv", index=False)
