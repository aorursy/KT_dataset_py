import numpy as np # linear algebra

import pandas as pd # data processing



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import math
titanic_data= pd.read_csv("../input/titanic/train.csv")

titanic_data.head(10)
print("No. of passengers in original data : " +str(len(titanic_data.index)))
sns.countplot(x="Survived", data=titanic_data)
sns.countplot(x="Survived", hue="Sex", data=titanic_data)
sns.countplot(x="Survived", hue="Pclass", data=titanic_data)
titanic_data["Age"].plot.hist()
titanic_data["Fare"].plot.hist(bins=20, figsize=(10,5))
titanic_data.info()
sns.countplot(x="SibSp", data=titanic_data)
sns.countplot(x="Parch", data=titanic_data)
titanic_data.isnull()
titanic_data.isnull().sum()
sns.boxplot(x="Pclass", y="Age", data=titanic_data)
titanic_data.head()
titanic_data.drop("Cabin", axis=1, inplace=True)
titanic_data.head()
titanic_data.dropna(inplace=True)
titanic_data.head()
titanic_data.isnull().sum()
sex=pd.get_dummies(titanic_data["Sex"],drop_first=True)

sex.head()
embark=pd.get_dummies(titanic_data["Embarked"],drop_first=True)

embark.head()
titanic_data.drop(["Sex", "Embarked", "Name", "Ticket"], axis=1, inplace=True)
titanic_data=pd.concat([titanic_data,sex,embark], axis=1)
titanic_data.head()
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(titanic_data.drop('Survived',axis=1), titanic_data['Survived'], test_size=0.3, random_state=1)
from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()
logmodel.fit(X_train, Y_train)
predictions=logmodel.predict(X_test)

X_test.head()
from sklearn.metrics import classification_report

classification_report(Y_test, predictions)
predictions
from sklearn.metrics import confusion_matrix

confusion_matrix(Y_test, predictions)
from sklearn.metrics import accuracy_score
accuracy_score(Y_test, predictions)