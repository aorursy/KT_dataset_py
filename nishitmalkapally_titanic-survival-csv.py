# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("/kaggle/input/titanic/train.csv")

test = pd.read_csv("/kaggle/input/titanic/test.csv")

train.head()
test.head()
sns.barplot(train.Sex,train.Survived)
sns.barplot(train.Pclass,train.Survived)
train["Title"]=train.Name.str.extract(' ([A-Za-z]+)\.',expand=False)

test["Title"]=test.Name.str.extract(' ([A-Za-z]+)\.',expand=False)
train['Title'].value_counts()
test['Title'].value_counts()
train['Title']=train['Title'].replace(['Mr','Dr',"Major","Sir"],1)

train['Title']=train['Title'].replace(['Mrs','Mme',"Lady"],2)

train['Title']=train['Title'].replace(['Miss','Ms','Mlle'],3)

train['Title']=train['Title'].replace(['Master'],4)

train['Title']=train['Title'].replace(['Capt','Col',"Countess","Jonkheer","Rev","Don"],5)
test['Title']=test['Title'].replace(['Mr','Dr',"Major","Sir"],1)

test['Title']=test['Title'].replace(['Mrs','Mme',"Lady"],2)

test['Title']=test['Title'].replace(['Miss','Ms','Mlle'],3)

test['Title']=test['Title'].replace(['Master'],4)

test['Title']=test['Title'].replace(['Capt','Col',"Countess","Jonkheer","Rev","Don","Dona"],5)
plt.figure(figsize=(15,6))

sns.barplot(train.Title,train.Survived)
train.Age.fillna(int(train.Age.median()),inplace=True)
test.Age.fillna(int(test.Age.median()),inplace=True)
train.loc[train.Age <= 16 ,"Age"]=0

train.loc[(train.Age >16)&(train.Age<=32),"Age"]=1

train.loc[(train.Age >32)&(train.Age<=48),"Age"]=2

train.loc[(train.Age >48)&(train.Age<=64),"Age"]=3

train.loc[train.Age >64,"Age"]=4
test.loc[train.Age <= 16 ,"Age"]=0

test.loc[(train.Age >16)&(train.Age<=32),"Age"]=1

test.loc[(train.Age >32)&(train.Age<=48),"Age"]=2

test.loc[(train.Age >48)&(train.Age<=64),"Age"]=3

test.loc[train.Age >64,"Age"]=4
sns.barplot(train.Age,train.Survived)
train['Family_size']=train['SibSp']+train['Parch']+1

test['Family_size']=test.SibSp+test.Parch+1
sns.barplot(train.Family_size,train.Survived)
train['Alone'] = 0 

train.loc[train['Family_size'] == 1 , "Alone"]=1
sns.barplot(train.Alone,train.Survived)
test['Alone'] = 0 

test.loc[train['Family_size'] == 1 , "Alone"]=1
train.Embarked.fillna(train.Embarked.mode(),inplace=True)

test.Embarked.fillna(test.Embarked.mode(),inplace=True)
train['Fare_cat'] = pd.qcut(train.Fare,4)

train.Fare_cat.value_counts()
train.loc[train.Fare<=7.91,'Fare']=1

train.loc[(train.Fare>7.91) & (train.Fare <= 14.454),'Fare']=2

train.loc[(train.Fare>14.454) & (train.Fare <= 31.0),'Fare']=3

train.loc[train.Fare>31.0,'Fare']=4





test.loc[train.Fare<=7.91,'Fare']=1

test.loc[(train.Fare>7.91) & (train.Fare <= 14.454),'Fare']=2

test.loc[(train.Fare>14.454) & (train.Fare <= 31.0),'Fare']=3

test.loc[train.Fare>31.0,'Fare']=4
corr = train.corr()

plt.figure(figsize=(10,10))

sns.heatmap(corr,annot=True)
features = ['Pclass','Sex','Fare','Embarked','Title','Alone']
from sklearn.ensemble import RandomForestClassifier

y_train = train['Survived']



X_train= pd.get_dummies(train[features])



X_test = pd.get_dummies(test[features])



clf = RandomForestClassifier(n_estimators=100,max_depth=5,random_state=42)



clf.fit(X_train,y_train)



pred = clf.predict(X_test)



op =  pd.DataFrame({"PassengerID":test['PassengerId'],"Survived":pred})



op.to_csv("Titanic_submission1",index=False)



print("Submission sucessfully saved")
