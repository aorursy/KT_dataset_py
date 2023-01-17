import numpy as np

import pandas as pd

import seaborn as sns

from matplotlib import pyplot as plt

%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/titanic/train.csv')

df.head(2)
sns.countplot(x='Sex', data = df)
sns.countplot(x='Sex', hue='Pclass', data = df)
sns.heatmap(df.isnull())
df.drop(['Cabin','Fare','Parch','SibSp','Ticket','Name','PassengerId'],axis=1,inplace=True)

df = df.fillna(0)
sns.heatmap(df.isnull())
gender = pd.get_dummies(df['Sex'], drop_first=True)

embark = pd.get_dummies(df['Embarked'], drop_first=True)

pClass = pd.get_dummies(df['Pclass'], drop_first=True)
df = pd.concat([df,gender,embark,pClass],axis=1)

df.head(2)
df.drop(['Pclass','Sex','Embarked'],axis=1,inplace=True)

df.head(2)
X = df.drop('Survived', axis=1)

y = df['Survived']
from sklearn.model_selection import train_test_split

# split data in train & test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)



from sklearn.linear_model import LogisticRegression

# create and configure model

model = LogisticRegression()

# fit model

model.fit(X, y)
pridiction = model.predict(X_test)
from sklearn.metrics import accuracy_score

#check the accuracy of model

accuracy_score(y_test,pridiction)*100