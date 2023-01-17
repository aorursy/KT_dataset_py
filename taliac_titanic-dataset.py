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

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")
df_train = train
train.info()
df_train.head(3)
df_train.isna().any()
df_train.dropna(inplace=True)
df_train.isna().any()
df_train["Sex"] = np.where(train["Sex"]=="male", 0, 1)
df_train["Cabin"] = df_train.Cabin.str[0]
df_train
df_train.Cabin = [ ord(x) - 64 for x in df_train.Cabin]
df_train.Embarked = np.where(df_train["Embarked"] == "C", 1, df_train["Embarked"])

df_train.Embarked = np.where(df_train["Embarked"] == "Q", 2, df_train["Embarked"])

df_train.Embarked = np.where(df_train["Embarked"] == "S", 3, df_train["Embarked"])
df_train["FirstClass"] = np.where(df_train["Pclass"] == 1, 1, 0)

df_train["SecondClass"] = np.where(df_train["Pclass"] == 2, 1,0)

df_train["ThirdClass"] = np.where(df_train["Pclass"] == 3, 1, 0)
df_train
df_train.corr()
features = df_train[["Sex", "Age", "Pclass"]]

survival = df_train["Survived"]
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

train_features = scaler.fit_transform(features)
model = LogisticRegression()
model.fit(train_features, survival)

model.coef_
print(model.score(train_features, survival))
model2 = LogisticRegression()

features2 = df_train[["Sex", "Pclass"]]

train_features2 = scaler.fit_transform(features2)

model2.fit(train_features2, survival)

print(model2.score(train_features2, survival))
model3 = LogisticRegression()

features3 = df_train[["Sex", "Pclass", "Age", "SibSp"]]

train_features3 = scaler.fit_transform(features3)

model3.fit(train_features3, survival)

print(model3.score(train_features3, survival))
model4 = LogisticRegression()

features4 = df_train[["Sex", "FirstClass", "SecondClass", "Age", "SibSp"]]

train_features4 = scaler.fit_transform(features4)

model4.fit(train_features4, survival)

print(model4.score(train_features4, survival))