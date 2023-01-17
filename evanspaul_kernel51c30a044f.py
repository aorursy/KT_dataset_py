import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import statsmodels.api as sm

import seaborn as sns

sns.set()

from sklearn.linear_model import LogisticRegression 

from pandas import Series; from numpy.random import randn

import os

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv('/kaggle/input/titanic/train.csv')
data

data.describe(include='all')
data= data.drop('Cabin', axis =1)
data.isnull().sum()
# data = data.dropna(axis=0)
data['Age'] = data['Age'].fillna(data['Age'].median())
data.describe(include = 'all')
data= data.drop('PassengerId', axis =1)
data = data.drop('Ticket', axis=1)
data = data.drop('Name', axis =1)
data = pd.get_dummies(data)
x= data[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S']]



y= data[['Survived']]
data.shape
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x)
scaler.transform(x)
logit = LogisticRegression()
logit.fit(x,y)
logit.coef_
y_pred = logit.predict(x)
from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(y,y_pred)

print(confusion_matrix)
cm_df = pd.DataFrame(confusion_matrix)

cm_df.columns = ['predict 0', 'predict 1']

cm_df = cm_df.rename(index = {0: 'actual 0', 1: 'actual 1'})

cm_df