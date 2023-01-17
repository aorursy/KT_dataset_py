import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import math

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression 

from sklearn.model_selection import train_test_split

%matplotlib inline

sns.set()



data = pd.read_csv("../input/titanic-extended/full.csv")
data.head(10) #view the first few column of the dataset
data.shape #view the shape of the dataset (m,n)
sns.countplot(x="Survived", hue = "Sex", data=data) #analysis with visulisation
sns.countplot(x="Survived", hue = "Pclass", data=data)
data["Age"].plot.hist()
data.info()
data.isnull().sum() #lists the number of null values in each feature
data.head(5)
data.drop(["Hometown",

           "Age_wiki",

           "Name_wiki", 

           "Ticket",

           "Cabin",

           "Destination",

           "Body",

           "WikiId", 

           "Name","Boarded",

           "Lifeboat",

           "Name",

           "PassengerId"],axis=1, inplace= True)
data.head(5)

data.dropna(inplace=True) #drop all the data entries with missing values
data.isnull().sum()
sex = pd.get_dummies(data["Sex"], drop_first=True)

embark = pd.get_dummies(data["Embarked"], drop_first=True)

clss = pd.get_dummies(data["Class"], drop_first=True)

data.drop('Pclass', axis=1,inplace=True)
data=pd.concat([data,sex,embark,clss], axis=1)

data.head(5)
data.drop(["Sex","Embarked",'Class'],axis=1,inplace=True) 
data.head(5)
x=data.drop("Survived",axis=1) #features

y=data["Survived"] #target variable


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=42)
#build the model and train the model

model = LogisticRegression(max_iter=1000)

model.fit(x_train, y_train)
#perform predictions

predic = model.predict(x_test)
#analyse the accuracy of the model

report = classification_report(y_test,predic)
print(report)


accuracy_score(y_test, predic)