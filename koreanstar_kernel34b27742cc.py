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
train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')
import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline

def bar_chart(feature):

    survived=train[train['Survived']==1][feature].value_counts()

    dead=train[train['Survived']==0][feature].value_counts()

    df=pd.DataFrame([survived,dead])

    df.index=('Survived','Dead')

    df.plot(kind='bar',stacked=True,figsize=(10,5))
train.head()
train.drop(['Ticket','PassengerId'],axis=1,inplace=True)

test.drop(['Ticket','PassengerId'],axis=1,inplace=True)
train.head()
def Sex_num(sex):

    if sex=='male': return 0

    else: return 1



    
train['Sex']=train['Sex'].apply(Sex_num)

test['Sex']=test['Sex'].apply(Sex_num)
def Age_cat(age):

    if age<20:return 0

    elif age<40:return 1

    elif age<60:return 2

    else:return 3
train['Age']=train['Age'].apply(Age_cat)

test['Age']=test['Age'].apply(Age_cat)
train.drop(['Fare','Name','Cabin'],axis=1,inplace=True)

test.drop(['Fare','Name','Cabin'],axis=1,inplace=True)
def Embarked_num(Embarked):

    if Embarked=='Q':return 0

    elif Embarked=='C':return 1

    else:return 2
train['Embarked']=train['Embarked'].apply(Embarked_num)

test['Embarked']=test['Embarked'].apply(Embarked_num)
train.drop(['SibSp','Parch'],axis=1,inplace=True)

test.drop(['SibSp','Parch'],axis=1,inplace=True)
from sklearn.tree import DecisionTreeClassifier

dlt=DecisionTreeClassifier()

y_train_df=train['Survived']

X_train_df=train.drop('Survived',axis=1)

dlt.fit(X_train_df,y_train_df)
pred=dlt.predict(test)
from sklearn.metrics import accuracy_score



get_sub=pd.read_csv('../input/gender_submission.csv')

sub=get_sub.drop('PassengerId',axis=1)



print(accuracy_score(pred,sub))
submission=pd.DataFrame({

    'PassengerId':get_sub['PassengerId'],

    'Survived':pred})

submission.to_csv('submission.csv',index=False)