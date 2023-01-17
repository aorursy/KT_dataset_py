import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import math

%matplotlib inline

import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data=pd.read_csv("../input/titanic-solution-for-beginners-guide/train.csv")

data.head(10)
print("# of passengers:"+str(len(data.index)))
sns.countplot(x="Survived",data=data)
sns.countplot(x="Survived",hue="Sex",data=data)
sns.countplot(x="Survived",hue="Pclass",data=data)
data["Age"].plot.hist()
data["Fare"].plot.hist(bins=20,figsize=(10,5))
data.info()
data.isnull()
data.isnull().sum()
sns.heatmap(data.isnull(),yticklabels=False,cmap="viridis")
sns.boxplot(x="Pclass",y="Age",data=data)
data.drop("Cabin",axis=1,inplace=True)

data.head(5)
data.dropna(inplace=True)

sns.heatmap(data.isnull(),yticklabels=False,cmap="viridis")
sex=pd.get_dummies(data["Sex"],drop_first=True)

sex.head(5)
embark=pd.get_dummies(data["Embarked"],drop_first=True)

embark.head(5)
pcl=pd.get_dummies(data["Pclass"],drop_first=True)

pcl.head(5)
data=pd.concat([data,sex,embark,pcl],axis=1)

data.head(5)
data.drop(["Sex","Embarked","PassengerId","Name","Ticket","Pclass"],axis=1,inplace=True)

data.head(5)
x=data.drop("Survived",axis=1)

y=data["Survived"]
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)
from sklearn.linear_model import LogisticRegression

logmodel=LogisticRegression()

logmodel.fit(x_train,y_train)
predictions=logmodel.predict(x_test)

from sklearn.metrics import classification_report

classification_report(y_test,predictions)
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test,predictions)
from sklearn.metrics import accuracy_score

accuracy_score(y_test,predictions)