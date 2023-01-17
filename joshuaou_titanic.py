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
data_raw = pd.read_csv('../input/train.csv')

# data_raw.drop(['PassengerId', 'Name'])

data_raw.head()
print(data_raw.shape)

data_raw.isnull().sum()
data_raw.drop(['PassengerId', 'Fare', 'Ticket', 'Cabin', 'Name'], axis=1, inplace=True)

data_raw['Age'].fillna(data_raw['Age'].mean(), inplace=True)

data_raw.dropna(inplace=True)

data_raw.isnull().sum()
for label in data_raw.columns.values:

    if label == 'Age':

        continue

    print('='*16)

    print(data_raw[label].value_counts())
data = pd.get_dummies(data_raw)

data.head()
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split



# Prepare the data

X_raw = data.iloc[:, 1:]

y_raw = data.iloc[:, 0]



X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.2)



model = LinearRegression()

model.fit(X_train, y_train)

model.score(X_test, y_test)
