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
import sklearn as sk

import pandas as pd

import numpy as np
df = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

submission = pd.read_csv("../input/gender_submission.csv")

df.head()
df.isna().sum()
test.head()
def cat_age(col):

    if col < 25:

        return "young"

    elif col < 50:

        return "mid"

    return "senior"

df["Age"] = df["Age"].apply(cat_age)

test["Age"] = test["Age"].apply(cat_age)

df.head()

test.head()
df = df[["Pclass", "Sex", "SibSp", "Survived","Age"]]

df["Pclass"] = df["Pclass"].astype("category")

df["Sex"] = df["Sex"].astype("category")

df["SibSp"] = df["SibSp"].astype("category")

df["Age"] = df["Age"].astype("category")

test = test[["Pclass", "Sex", "SibSp","Age"]]

test["Pclass"] = test["Pclass"].astype("category")

test["Sex"] = test["Sex"].astype("category")

test["SibSp"] = test["SibSp"].astype("category")

test["Age"] = test["Age"].astype("category")

df = pd.get_dummies(df)

test = pd.get_dummies(test)
labels = df.pop("Survived")

df.head()
test.head()
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

for _ in range(10):

    model.fit(df, labels)

pred = model.predict(test)
score = model.score(df, labels)

print(score)
submission["Survived"] = pred
submission.to_csv("submission.csv", index=False)