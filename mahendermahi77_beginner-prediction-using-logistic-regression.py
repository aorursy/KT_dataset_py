#importing required libs

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
#reading input

titanic_train=pd.read_csv("../input/train.csv")

titanic_test=pd.read_csv("../input/test.csv")
#having a glance at the data

titanic_train.head()
titanic_test.head()
#checking for any null values

titanic_train.info()

print("----------------------------------")

titanic_test.info()
#filling missing age columns with median 

titanic_train['Age'].fillna(titanic_train['Age'].median(),inplace=True)

titanic_test['Age'].fillna(titanic_test['Age'].median(),inplace=True)
#filling embarked with S it is the most occurring value

titanic_train["Embarked"].fillna('S',inplace=True)
#Fare has one missing value

titanic_test['Fare'].fillna(titanic_test['Fare'].median(),inplace=True)

titanic_train['Fare']=titanic_train['Fare'].astype(int)

titanic_test['Fare']=titanic_test['Fare'].astype(int)
#dropping the useless columns

titanic_train.drop(['PassengerId','Name','Ticket','Cabin'],axis=1,inplace=True)

titanic_test.drop(['Name','Ticket','Cabin'],axis=1,inplace=True)
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5),gridspec_kw={'width_ratios':[1,2]})

sns.countplot(x='Pclass',data=titanic_train,ax=ax1)

sns.countplot(x='Survived',hue='Pclass',data=titanic_train,ax=ax2)
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5),gridspec_kw={'width_ratios':[0.75,1.5]})

sns.countplot(x='Sex',data=titanic_train,ax=ax1)

sns.countplot(x='Survived',data=titanic_train,hue='Sex',palette='cubehelix',ax=ax2)
fig,(ax1,ax2)=plt.subplots(2,1,figsize=(25,15))

sns.countplot(x='Age',data=titanic_train,ax=ax1)

sns.countplot(x='Survived',data=titanic_train,hue='Age',palette='cubehelix',ax=ax2)

plt.xticks(size=15)
#sorting the Age into 3 groups and 1 others

titanic_train['Age_Child']=(titanic_train['Age']<=18).astype(int)

titanic_train['Age_Youth']=((titanic_train['Age']>18)&(titanic_train['Age']<=35)).astype(int)

titanic_train['Age_Middle']=((titanic_train['Age']>35)&(titanic_train['Age']<=50)).astype(int)



titanic_test['Age_Child']=(titanic_test['Age']<=18).astype(int)

titanic_test['Age_Youth']=((titanic_test['Age']>18)&(titanic_test['Age']<=35)).astype(int)

titanic_test['Age_Middle']=((titanic_test['Age']>35)&(titanic_test['Age']<=50)).astype(int)
#creating dummy variable for sex

titanic_train['Sex_M']=(titanic_train['Sex']=='male').astype(int)



titanic_test['Sex_M']=(titanic_test['Sex']=='male').astype(int)
#dummy variable for Parents and Children

titanic_train['Parch_Y']=(titanic_train['Parch']>=1).astype(int)



titanic_test['Parch_Y']=(titanic_test['Parch']>=1).astype(int)
#Siblings

titanic_train['SibSp_Y']=(titanic_train['SibSp']>=1).astype(int)



titanic_test['SibSp_Y']=(titanic_test['SibSp']>=1).astype(int)
#Embarked

titanic_train['Embarked_S']=(titanic_train['Embarked']=='S').astype(int)

titanic_train['Embarked_C']=(titanic_train['Embarked']=='C').astype(int)



titanic_test['Embarked_S']=(titanic_test['Embarked']=='S').astype(int)

titanic_test['Embarked_C']=(titanic_test['Embarked']=='C').astype(int)
#Passenger class

titanic_train['Pclass_1']=(titanic_train['Pclass']==1).astype(int)

titanic_train['Pclass_2']=(titanic_train['Pclass']==2).astype(int)



titanic_test['Pclass_1']=(titanic_test['Pclass']==1).astype(int)

titanic_test['Pclass_2']=(titanic_test['Pclass']==2).astype(int)
from sklearn.linear_model import LogisticRegression

logreg=LogisticRegression()

X_train=titanic_train.drop(['Sex','SibSp','Parch','Pclass','Embarked','Age','Survived'],axis=1)

Y_train=titanic_train['Survived']

X_test=titanic_test.drop(['Sex','SibSp','Parch','Pclass','Embarked','Age','PassengerId'],axis=1)
logreg.fit(X_train,Y_train)

Y_pred=logreg.predict(X_test)

logreg.score(X_train,Y_train)
s=({"PassengerId":titanic_test["PassengerId"],"Survived":Y_pred})

submit=pd.DataFrame(data=s)

submit.to_csv('titanic.csv',index=False)