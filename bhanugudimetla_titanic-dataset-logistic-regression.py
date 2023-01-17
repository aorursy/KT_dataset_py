# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import sklearn

import pandas as pd

import seaborn as sns

titanic_data=pd.read_csv("/kaggle/input/ChanDarren_RaiTaran_Lab2a.csv")

titanic_data.head(5)
print("## no of passengers in titanic is:" +str(len(titanic_data.index)))
sns.countplot(x="Survived",data=titanic_data)
sns.countplot(x="Survived", hue="Sex" ,data=titanic_data)
sns.countplot(x="Survived",hue="Pclass",data=titanic_data)
titanic_data["Age"].plot.hist(bins=10)
titanic_data["Fare"].plot.hist(bins=20)
titanic_data.info()
sns.countplot(x="SibSp",data=titanic_data)
titanic_data.isnull()
titanic_data.isnull().sum()
sns.heatmap(titanic_data.isnull(),yticklabels="false",cmap='viridis')
sns.boxplot(x="Pclass",y="Age",data=titanic_data)
titanic_data.head(5)
titanic_data.drop("Cabin",axis=1,inplace=True)
titanic_data.head(5)
sns.heatmap(titanic_data.isnull(),yticklabels=False)
titanic_data.dropna(inplace=True)
titanic_data.isnull().sum()
sns.heatmap(titanic_data.isnull(),yticklabels=False,cbar=False)
titanic_data.head(2)
pcls=pd.get_dummies(titanic_data["Pclass"],drop_first=True)

pcls.head(5)
sex=pd.get_dummies(titanic_data["Sex"],drop_first=True)

sex.head(5)
embark=pd.get_dummies(titanic_data["Embarked"],drop_first=True)

embark.head(5)
titanic_data.head(5)
titanic_data=pd.concat([titanic_data,embark,sex,pcls],axis=1)
titanic_data.head(5)
titanic_data.drop(["Name"],axis=1,inplace=True)
titanic_data.drop(["PassengerId","Sex","Ticket","Embarked","Pclass"],axis=1,inplace=True)
titanic_data.head(5)
X=titanic_data.drop("Survived",axis=1)

y=titanic_data["Survived"]
from sklearn.model_selection import train_test_split

Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.3,random_state=1)
from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()
logmodel.fit(Xtrain,ytrain)
predictions=logmodel.predict(Xtest)
from sklearn.metrics import classification_report
classification_report(ytest,predictions)
from sklearn.metrics import confusion_matrix
confusion_matrix(ytest,predictions)
from sklearn.metrics import accuracy_score
accuracy_score(ytest,predictions)