import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
titanic_data = pd.read_csv("../input/titanic-dataset-applying-logistic-regression/titanic.csv")
titanic_data.head(20)
sns.countplot(x="Survived",data=titanic_data)
sns.countplot(x="Survived", hue="Sex",data=titanic_data)
sns.countplot(x="Survived", hue="Pclass",data=titanic_data)
titanic_data["Age"].plot.hist()

titanic_data["Fare"].plot.hist()

titanic_data["Fare"].plot.hist(bins=20, figsize=(10,5))
titanic_data.info()
sns.countplot(x="SibSp",data=titanic_data)
titanic_data.isnull().sum()
sns.heatmap(titanic_data.isnull())
sns.heatmap(titanic_data.isnull(),cmap="viridis")
sns.boxplot(x="Pclass", y="Age",data=titanic_data)
titanic_data.head(5)
titanic_data.drop("Cabin",axis=1, inplace=True)

titanic_data.dropna(inplace=True)
sns.heatmap(titanic_data.isnull(), cbar=False)
titanic_data.head(2)
pd.get_dummies(titanic_data['Sex'])



embark=pd.get_dummies(titanic_data["Embarked"])

embark.head(5)

sex=pd.get_dummies(titanic_data['Sex'],drop_first=True) 



embark=pd.get_dummies(titanic_data["Embarked"], drop_first=True) 

embark.head(5)



PcI =pd.get_dummies(titanic_data["Pclass"], drop_first=True) 

PcI.head(5)

titanic_data=pd.concat([titanic_data,sex,embark,PcI],axis=1)

titanic_data.head(5)
titanic_data.drop(['Sex','Embarked','PassengerId','Name','Ticket'], axis=1, inplace=True)

titanic_data.drop(['Pclass'], axis=1, inplace=True)

titanic_data.head()
x=titanic_data.drop("Survived",axis=1)

y=titanic_data["Survived"]
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.3,random_state=1)
logmodel=LogisticRegression()

logmodel.fit(x_train, y_train)
predictions = logmodel.predict(x_test)
from sklearn.metrics import classification_report

classification_report(y_test,predictions)
from sklearn.metrics import accuracy_score 

accuracy_score(y_test,predictions)
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, predictions)