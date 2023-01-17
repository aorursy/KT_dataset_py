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

test = pd.read_csv("../input/test.csv")

train = pd.read_csv("../input/train.csv")
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train.sample(5)
plt.figure(figsize=(10,10))

sns.heatmap(train.corr(), annot=True, cmap="Blues")
plt.figure(figsize=(9,6))

sns.barplot(x='Sex',y='Survived',data=train)
plt.figure(figsize=(9,6))

sns.barplot(x='Pclass', y='Survived', data=train)
train.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

test.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
train.isnull().sum()
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")
train_numerico = train.drop(['Sex','Embarked'], axis=1)

test_numerico = test.drop(['Sex','Embarked'], axis=1)
imputer.fit(train_numerico)

train_no_null = imputer.transform(train_numerico)
imputer.fit(test_numerico)

test_no_null = imputer.transform(test_numerico)
train_tr = pd.DataFrame(train_no_null, columns=train_numerico.columns)

test_tr = pd.DataFrame(test_no_null, columns=test_numerico.columns)
train_cat = train[['Sex', 'Embarked']]

test_cat = test[['Sex', 'Embarked']]
train_cat_encoded = pd.get_dummies(train_cat)

test_cat_encoded = pd.get_dummies(test_cat)
train_cat_encoded.head()
new_train = train_tr.join(train_cat_encoded)

new_test = test_tr.join(test_cat_encoded)
from sklearn.model_selection import train_test_split



predictors = new_train.drop(['Survived', 'PassengerId'], axis=1)

target = train["Survived"]

x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.2, random_state = 0)
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

decisiontree = DecisionTreeClassifier()

decisiontree.fit(x_train, y_train)

y_pred = decisiontree.predict(x_val)

acc_decisiontree = round(accuracy_score(y_pred, y_val), 2)

print('Acurácia: ',acc_decisiontree)