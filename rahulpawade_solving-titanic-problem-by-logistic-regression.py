import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
d1 = pd.read_csv("../input/titanic/train.csv")

d1.shape
d2 = pd.read_csv("../input/titanic/test.csv")

d2.shape
d11 = d1.drop(columns="Survived",axis=1)
d = pd.concat([d11,d2],axis=0)

d.shape
d.info()
d = d.drop(columns ="Cabin",axis=1)

d.info()
d = d.drop(columns="PassengerId",axis=1)

d = d.drop(columns="Name",axis=1)
d.head()
d_object = d.select_dtypes(include="object")
d_object
d_object["Embarked"].value_counts()
d_object = d_object.fillna(value="S",axis=1)
d_object.shape
from sklearn.preprocessing import LabelEncoder
e = LabelEncoder()
d_object["Ticket"] = e.fit_transform(d_object["Ticket"])

d_object["Sex"] = e.fit_transform(d_object["Sex"])

d_object["Embarked"] = e.fit_transform(d_object["Embarked"])
d_object.info()
d_num = d.select_dtypes(include=np.number)
d_num.info()
d_num["Age"] = d_num.fillna(d_num["Age"].mean(),axis=1)
d_num["Fare"] = d_num.fillna(d_num["Fare"].mean(),axis=1)
d_num.shape
data = pd.concat([d_num,d_object],axis=1)

data["Age"].value_counts()
data.info()

data.shape

data["Fare"].value_counts()
from sklearn.preprocessing import StandardScaler
s = StandardScaler()
data = s.fit_transform(data)
data = pd.DataFrame(data,columns=[['Age', 'Embarked', 'Fare', 'Parch', 'Pclass', 'Sex', 'SibSp', 'Ticket']])
x_train = data.iloc[:891,:]

x_train.shape
y_train = d1["Survived"]

y_train.shape
x_test = data.iloc[891:,:]
x_test.shape
from sklearn.linear_model import LogisticRegression

m1 = LogisticRegression()

m1.fit(x_train,y_train)

y_pred = m1.predict(x_test)
y_pred
file = {"PassengerId":(d2["PassengerId"]),"Survived":y_pred}

file = pd.DataFrame(file)

file.to_csv("submission.csv",index=False)
file.head()