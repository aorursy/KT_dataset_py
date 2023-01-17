import numpy as np

import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
data = pd.read_csv("../input/titanic/train.csv")

data.head()
data['Cabin'].value_counts()
data = data.drop(['PassengerId', 'Ticket', 'Name', 'Cabin'], axis = 1)

data.head()
data = pd.get_dummies(data)

data.head()
from sklearn.model_selection import train_test_split
data.isna().sum()
data['Age'].fillna(data['Age'].mean(), inplace = True)
data.isna().sum()
data_x = data.drop(['Survived'], axis=1)

data_y = data['Survived']
train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.3, shuffle = True)
from sklearn.linear_model import LogisticRegression



logistic = LogisticRegression(solver='liblinear')

logistic.fit(train_x, train_y)
logistic.score(test_x, test_y)