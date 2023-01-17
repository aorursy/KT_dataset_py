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
# linear algebra

import numpy as np

# explorasi dataset

import pandas as pd

# plotting data

import matplotlib.pyplot as plt
df_train = pd.read_csv("/kaggle/input/titanic/train.csv")

df_test = pd.read_csv("/kaggle/input/titanic/test.csv")
df_train.head().T
df_test.head().T
df_train.describe()
df_train.describe(include=["O"])
df_test.describe()
df_test.describe(include=["O"])
woman = df_train.loc[df_train["Sex"] == "female"]["Survived"]

rate_woman = sum(woman)/len(woman)

print(rate_woman)
man = df_train.loc[df_train["Sex"] == "male"]["Survived"]

rate_man = sum(man)/len(man)

print(rate_man)
from sklearn.ensemble import RandomForestClassifier



y = df_train["Survived"]



features = ["Pclass", "Sex", "SibSp", "Parch"]

X = pd.get_dummies(df_train[features])

X_test = pd.get_dummies(df_test[features])



model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test)



print(model.score(X, y))
predictions = model.predict(pd.get_dummies(df_test[features]))
output = pd.DataFrame({'PassengerId': df_test.PassengerId, 'Survived': predictions})
output.to_csv("Submission.csv", index=False)