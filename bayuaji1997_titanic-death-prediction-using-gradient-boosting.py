# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data_train=pd.read_csv("../input/train.csv")

data_test=pd.read_csv("../input/test.csv")
data_train.shape
data_test.shape
data_train.head()
data_train.describe(include="all")
data_test.describe(include="all")
data_train.isnull().sum()
erase=["Ticket","Cabin","Name"]

data_train=data_train.drop(erase,axis=1)
#Check Data Train

data_train.head()
#Cleaning Missing Values

data_train["Age"]=data_train["Age"].fillna(data_train["Age"].mean)

data_train=data_train.fillna({"Embarked": "S"})
#Check Missing Values

data_train.isnull().sum()
#CHange Sex to Int

data_train['Sex']=data_train['Sex'].apply(lambda x:1 if x=='male' else 0)
#Check Data Train

data_train.head()
#Check Data Test

data_test.isnull().sum()
data_test.drop(erase,axis=1)
#Fill missing values

data_test["Age"]=data_test["Age"].fillna(data_test["Age"].mean)

data_test["Fare"]=data_test["Fare"].fillna(data_test["Fare"].mean)

#Change column sex into int

data_test['Sex']=data_test['Sex'].apply(lambda x:1 if x=='male' else 0)
#Check Data Test

data_test.head()
data_test.drop(["Cabin","Name"],axis=1)
data_test.describe()
embarked_mapping = {"S": 1, "C": 2, "Q": 3}

data_train['Embarked'] = data_train['Embarked'].map(embarked_mapping)

data_test['Embarked'] = data_test['Embarked'].map(embarked_mapping)

data_train.drop(["Age"],axis=1)

data_test.drop(["Age"],axis=1)
#HeatMap

import seaborn as sns

sns.heatmap(data_train.corr(),annot=True)
sns.barplot(x="Sex",y="Survived",data=data_train)
data_train.info()
from sklearn.model_selection import train_test_split

features=data_train.drop(["PassengerId","Survived","Age","Fare"],axis=1)

target=data_train["Survived"]

X_train,X_test,y_train,y_test=train_test_split(features,target,test_size=0.22,random_state=0)
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import accuracy_score

model = GradientBoostingClassifier()

model.fit(X_train, y_train)

pred = model.predict(X_test)

acc = round(accuracy_score(pred, y_test) * 100, 2)

print(acc)
ids=data_test["PassengerId"]

predictions=data_test.drop(["PassengerId","Name","Age","Cabin","Ticket","Fare"], axis=1)

predictions.head()

test_pred=model.predict(predictions)

test_pred
output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': test_pred })

output.to_csv('submission.csv', index=False)