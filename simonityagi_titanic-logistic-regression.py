import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import math
df=pd.read_csv("../input/titanic/train.csv",index_col=0)
df.head(10)
len(df.index)
df.info()
sns.countplot(x='Survived',hue='Sex',data=df)
sns.countplot(x='Survived',hue='Pclass',data=df)
df['Age'].plot.hist(bins=10,figsize=(10,5))
sns.countplot(x='SibSp',hue='Survived',data=df)
sns.countplot(hue='Survived',x='Parch',data=df)
age_mean=df['Age'].mean()

df['Age'].replace(np.nan,age_mean,inplace=True)
df.info()
df.drop(['Name','Ticket','Cabin'],axis=1,inplace=True)
sex=pd.get_dummies(df['Sex'],drop_first=True)

pcl=pd.get_dummies(df['Pclass'],drop_first=True)

emb=pd.get_dummies(df['Embarked'],drop_first=True)

Sb=pd.get_dummies(df['SibSp'],drop_first=True)
df=pd.concat([df,emb,pcl,sex,Sb],axis=1)
df.head(5)
df.drop(['Sex','Pclass','Embarked','SibSp'],axis=1,inplace=True)
df.head(5)
x=df.drop('Survived',axis=1,inplace=False)

y=df[['Survived']]
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
from sklearn.linear_model import LogisticRegression

logmodel=LogisticRegression()

logmodel.fit(x_train,y_train)
predictions=logmodel.predict(x_test)
from sklearn.metrics import classification_report

print(classification_report(y_test,predictions))
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test,predictions))
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,predictions))
x_train=df.drop('Survived',axis=1,inplace=False)

y_train=df[['Survived']]
x_train.head(5)
from sklearn.linear_model import LogisticRegression

logmodel=LogisticRegression()

logmodel.fit(x_train,y_train)
x_test=pd.read_csv("../input/titanic/test.csv",index_col=0)
agemean=x_test['Age'].mean()

x_test['Age'].replace(np.nan,agemean,inplace=True)
Fare=x_test['Fare'].mean()

x_test['Fare'].replace(np.nan,Fare,inplace=True)
x_test.info()
x_test.drop(['Name','Ticket','Cabin'],axis=1,inplace=True)
sex=pd.get_dummies(x_test['Sex'],drop_first=True)

pcl=pd.get_dummies(x_test['Pclass'],drop_first=True)

Sb=pd.get_dummies(x_test['SibSp'],drop_first=True)

emb=pd.get_dummies(x_test['Embarked'],drop_first=True)

x_test=pd.concat([x_test,emb,pcl,sex,Sb],axis=1)
x_test.head(4)
x_test.drop(['Sex','Pclass','SibSp','Embarked'],axis=1,inplace=True)
predictions=logmodel.predict(x_test)
predictions=np.array(predictions,dtype='int64')
submission=pd.read_csv("../input/titanic/gender_submission.csv")
submission['Survived']=predictions
submission.head(5)
submission.to_csv('./predictions.csv')