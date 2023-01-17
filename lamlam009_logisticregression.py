# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/train.csv")



y = df["Survived"]



columns = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

df = df[columns]



from sklearn.preprocessing import OneHotEncoder

df = pd.get_dummies(df)

Pclass_1 = df["Pclass"] == 1

Pclass_2 = df["Pclass"] == 2

Pclass_3 = df["Pclass"] == 3

df = df.join(Pclass_1, lsuffix="", rsuffix="_1")

df = df.join(Pclass_2, lsuffix="", rsuffix="_2")

df = df.join(Pclass_3, lsuffix="", rsuffix="_3")

del df["Pclass"]



X = df
test = pd.read_csv("../input/test.csv")

test = test[columns]



X_test = pd.get_dummies(test)



Pclass_1 = X_test["Pclass"] == 1

Pclass_2 = X_test["Pclass"] == 2

Pclass_3 = X_test["Pclass"] == 3

X_test = X_test.join(Pclass_1, lsuffix="", rsuffix="_1")

X_test = X_test.join(Pclass_2, lsuffix="", rsuffix="_2")

X_test = X_test.join(Pclass_3, lsuffix="", rsuffix="_3")

del X_test["Pclass"]
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import Imputer
my_pipeline = make_pipeline(Imputer(), LogisticRegression())

my_pipeline.fit(X, y)

my_pipeline.score(X, y)
df = pd.read_csv("../input/gender_submission.csv")

y_test = df.Survived

my_pipeline.score(X_test, y_test)
preds = my_pipeline.predict(X_test)
test = pd.read_csv("../input/test.csv")

my_submission = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': preds})

# you could use any filename. We choose submission here

my_submission.to_csv('submission.csv', index=False)