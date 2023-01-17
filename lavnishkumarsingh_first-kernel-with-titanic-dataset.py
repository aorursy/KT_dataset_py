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
train = pd.read_csv("/kaggle/input/titanic/train.csv")
train.head()
train.shape
drop = ["Name","Sex","Ticket","Cabin"]

train.drop(drop,axis=1,inplace=True)
train.head()
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

train.dropna(inplace=True)

train["Embarked"] = le.fit_transform(train["Embarked"])
train.shape
train
test = pd.read_csv("/kaggle/input/titanic/test.csv")
test.drop(drop,axis=1,inplace=True)
test.isna().sum()
test["Age"]=test["Age"].fillna(train["Age"].mean())
test.isna().sum()
test["Fare"] = test["Fare"].fillna(test["Fare"].mean())
test.isna().sum()
y = train["Survived"]
train.drop("Survived",axis=1,inplace=True)
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(train,y)
test["Embarked"] = le.transform(test["Embarked"])
test.isna().sum()
pred = clf.predict(test)
sample = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
len(sample)
len(pred)
sample.head()
sample["Survived"] = pred
sample.to_csv("result.csv",index=False)
sample