#import libraries

import pandas as pd

from  sklearn.tree import DecisionTreeClassifier



train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')
train.head()
train.drop(['Name','Embarked','Ticket','Cabin'],inplace=True,axis=1)

test.drop(['Name','Embarked','Ticket','Cabin'],inplace=True,axis=1)

train.head()
#to create dummy data columns from categorial ones

train=pd.get_dummies(train)

test=pd.get_dummies(test)

train.head()
#check null values

train.isnull().sum().sort_values(ascending=True)

test.isnull().sum().sort_values(ascending=True)
#fill the null age with mean age and null fare  with mean fare

train['Age'].fillna(train['Age'].mean(),inplace=True)

test['Age'].fillna(test['Age'].mean(),inplace=True)

test['Fare'].fillna(test['Fare'].mean(),inplace=True)
#create feature and label

feature=train.drop('Survived',axis=1)

label=train['Survived']

clf=DecisionTreeClassifier()

clf.fit(feature,label)
#submit

sub=pd.DataFrame()

sub['PassengerId']=test['PassengerId']

sub['Survived']=clf.predict(test)

sub.to_csv('submission.csv',index=False)