# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


%matplotlib inline
%load_ext autoreload
%autoreload 2

import matplotlib.pyplot as plt 
df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")
df_train.head(n=5)
df_train.describe()
df_train.info()
plt.matshow(df_train.corr())
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

def transform(df, columns):
    for c in columns:
        le.fit(df[c].values)
        df["%s_label" % c] = le.transform(df[c])
    return df

def prepare_data(df):
    df_prep = df.drop("Name", axis=1)
    
    print("Number of NaN values: %d" % df.isnull().sum().sum())
    df_prep['Cabin'].fillna('not defined', inplace=True)
    df_prep['Age'].fillna(0.0, inplace=True)
    df_prep['Embarked'].fillna('not defined', inplace=True)
    return transform(df_prep, ['Sex', 'Cabin', 'Ticket', 'Embarked'])


df_train_prep = prepare_data(df_train)
df_train_prep.head()
df_train_prep['Cabin'].value_counts()
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

columns = ['Sex_label', 'Cabin_label', 'Ticket_label', 'Embarked_label', 'Pclass', 'Age']
X = df_train_prep[columns]
y = df_train_prep[['Survived']]

clf = DecisionTreeClassifier(random_state=42)
clf = clf.fit(X, y)
result = pd.DataFrame()
df_test_prep = prepare_data(df_test)
df_test_prep.head()

X = df_test_prep[columns]
y = clf.predict(X)

result['PassengerId'] = df_test_prep['PassengerId']
result['Survived'] = y

result.head()

result.to_csv('output.csv', header=['PassengerId', 'Survived'], index=False)