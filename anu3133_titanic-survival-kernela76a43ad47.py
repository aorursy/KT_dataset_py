



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import seaborn as sns

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data_train=pd.read_csv("../input/train.csv")

data_test=pd.read_csv("../input/test.csv")

data_train.head(100)
data_train.describe().T

data_train.info()
sns.countplot(data=data_train,x='Survived',hue='Pclass')
sns.countplot(data=data_train,x='Survived',hue='Sex')
sns.boxplot(data=data_train,y='Age',x='Survived',hue='Pclass')
sns.countplot(data=data_train,x='Sex',hue='Survived')
m1=round(data_train[data_train['Pclass']==1]['Age'].dropna().mean())

m2=round(data_train[data_train['Pclass']==2]['Age'].dropna().mean())

m3=round(data_train[data_train['Pclass']==3]['Age'].dropna().mean())

def fi(col):

    Age=col[0]

    Pclass=col[1]

    if pd.isnull(Age):

        if Pclass==1:

            return m1

        elif Pclass==2:

            return m2

        else:

            return m3

    else:

        return Age

data_train['Age']=data_train[['Age','Pclass']].apply(fi,axis=1)

print(data_train["Age"])

    
sns.catplot(data=data_train,kind='count',x='Survived',hue='Sex',col='Pclass',row='Embarked')
data_train['Embarked']=data_train['Embarked'].fillna(method='ffill')
Sex=pd.get_dummies(data=data_train['Sex'],drop_first=True)

embarked=pd.get_dummies(data=data_train['Embarked'],drop_first=True)

X_train = pd.concat([data_train[['Age', 'SibSp', 'Parch']], Sex, embarked], axis=1)

y_train=data_train['Survived']
data_test['Age'] = data_test[['Age', 'Pclass']].apply(fi, axis=1)

sex = pd.get_dummies(data_test['Sex'],drop_first=True)

embarked = pd.get_dummies(data_test['Embarked'],drop_first=True)

X_test = pd.concat([data_test[['Age', 'SibSp', 'Parch']], sex, embarked], axis=1)
from sklearn.tree import DecisionTreeClassifier

dt=DecisionTreeClassifier()

dt.fit(X_train,y_train)

dt.score(X_train,y_train)
prediction=dt.predict(X_test)

data_test['Survived']=prediction
submit = data_test[['PassengerId', 'Survived']]

submit.to_csv('submission.csv', index=False)