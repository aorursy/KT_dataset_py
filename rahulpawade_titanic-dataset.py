import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')

train.shape,test.shape
df = pd.DataFrame()

df = pd.concat([train,test],axis=0)
df.head()
df.info()
df.describe()
df.drop(columns='Survived',axis=1,inplace=True)
## droping irrelevent features

df.drop(columns=['PassengerId','Ticket','Name'],axis=1,inplace=True)
sns.countplot(x='Survived',hue='Pclass',data=train)
sns.countplot(x='Survived',hue='Sex',data=train)
sns.countplot(x='Survived',hue='Embarked',data=train)
sns.countplot(x='Survived',hue='SibSp',data=train)
sns.countplot(x='Survived',hue='Parch',data=train)
sns.distplot(train['Age'])
sns.distplot(train['Fare'])
sns.heatmap(df.isna(),yticklabels=False)
df['Age'] = df['Age'].fillna(df['Age'].mean())

df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

df['Fare'] = df['Fare'].fillna(df['Fare'].mean())
df.drop(columns='Cabin',axis=1,inplace=True)
sns.heatmap(df.isna(),yticklabels=False)
df.info()
sex = pd.get_dummies(df['Sex'],drop_first=True)

embarked = pd.get_dummies(df['Embarked'],drop_first=True)
df = pd.concat([df,sex,embarked],axis=1)
df.drop(columns=['Sex','Embarked'],axis=1,inplace=True)
df.head()
df.iloc[:891,:].corrwith(train['Survived'])
sns.pairplot(df.iloc[:891,:])
x_train = df.iloc[:891,:]

y_train = train['Survived']

x_test = df.iloc[891:,:]
from xgboost import XGBClassifier

model = XGBClassifier()
from sklearn.feature_selection import SelectFromModel

m = SelectFromModel(model.fit(x_train,y_train), prefit=True)

x_train = m.transform(x_train)

x_test = m.transform(x_test)
x_train.shape,x_test.shape,y_train.shape
model.fit(x_train,y_train)

y_pred = model.predict(x_test)
f = {'PassengerId':test['PassengerId'],'Survived':y_pred}

f = pd.DataFrame(f)

f.to_csv('submit.csv',index=False)
f.head()