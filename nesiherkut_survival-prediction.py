# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# It just helps displaying all outputs in a cell instead
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import pandas as pd
df_train = pd.read_csv("../input/titanic/train.csv")
df_test = pd.read_csv("../input/titanic/test.csv")
print(df_train.head())
print(df_test.head())
df_train.shape
df_test.shape
print('Number of survivors:')
df_train.Survived.sum()
df_train.isna().sum()
# Create a groupby object: by_sex_class
by_sex_class = df_train.groupby(by=['Sex', 'Pclass'])

# Write a function that imputes median
def impute_median(series):
    return series.fillna(series.median())

# Impute age and assign to df_train.Age
df_train.Age = by_sex_class['Age'].transform(impute_median)

print('number of nans in age: ', df_train.Age.isna().sum())
df_train.Embarked[df_train.Embarked.isna()]= df_train.Embarked.value_counts().idxmax()
df_train.drop(columns=['Cabin','Name','Ticket','PassengerId'], inplace=True)
df_train.isna().sum()
df_train.Embarked = df_train['Embarked'].astype('category').cat.codes
df_train.Sex = df_train['Sex'].astype('category').cat.codes
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df_train.drop(columns=['Survived'])
y = df_train.Survived
scaler=StandardScaler().fit(X)
x_scaled=scaler.transform(X)
model = LogisticRegression().fit(x_scaled, y)
model.score(x_scaled,y)
import matplotlib.pyplot as plt
%matplotlib inline

model.coef_
model.intercept_

plt.bar(X.columns, model.coef_[0])

# Create a groupby object: by_sex_class
by_sex_class = df_test.groupby(by=['Sex', 'Pclass'])

# Write a function that imputes median
def impute_median(series):
    return series.fillna(series.median())

# Impute age and assign to df_test.Age
df_test.Age = by_sex_class['Age'].transform(impute_median)

print('number of nans in age: ', df_test.Age.isna().sum())

df_test.drop(columns=['Cabin','Name','Ticket','PassengerId'], inplace=True)

df_test.Embarked = df_test['Embarked'].astype('category').cat.codes
df_test.Sex = df_test['Sex'].astype('category').cat.codes
df_test.dropna(inplace=True)
print(df_test.isna().sum())
x_scaled_test=scaler.transform(df_test)
pred = model.predict(x_scaled_test)
print('Number of survivors:',sum(pred))