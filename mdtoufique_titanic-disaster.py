# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import seaborn as sns
from matplotlib import pyplot as plt
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
titanic=pd.read_csv("../input/train.csv")
titanic.head()
titanic.describe()
sns.heatmap(titanic.isnull())
titanic.info()
sns.set_style('whitegrid')
sns.countplot(x="Age",hue="Sex",data=titanic)
sns.distplot(titanic[titanic["Survived"]==1]['Age'].dropna())
sns.boxplot(x="Pclass",y="Age",data=titanic)
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age
titanic['Age'] = titanic[['Age','Pclass']].apply(impute_age,axis=1)
titanic.drop("Cabin",inplace=True,axis=1)
sns.heatmap(titanic.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sex = pd.get_dummies(titanic['Sex'],drop_first=True)
embark = pd.get_dummies(titanic['Embarked'],drop_first=True)
titanic.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
titanic = pd.concat([titanic,sex,embark],axis=1)
titanic.head()
Y_train=titanic["Survived"]
X_train=titanic.drop("Survived",axis=1)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,Y_train)
titanic_test=pd.read_csv("../input/test.csv")
titanic_test.head()
sns.heatmap(titanic_test.isnull(),yticklabels=False,cbar=False,cmap='viridis')
titanic_test['Age'] = titanic_test[['Age','Pclass']].apply(impute_age,axis=1)
sns.heatmap(titanic_test.isnull(),yticklabels=False,cbar=False,cmap='viridis')
titanic_test.drop(["Cabin"],axis=1,inplace=True)
sns.heatmap(titanic_test.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sex_test = pd.get_dummies(titanic_test['Sex'],drop_first=True)
embark_test = pd.get_dummies(titanic_test['Embarked'],drop_first=True)
titanic_test.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
titanic_test = pd.concat([titanic_test,sex,embark],axis=1)
titanic_test.head()
titanic_test.describe()
predictions = logmodel.predict(titanic_test)
