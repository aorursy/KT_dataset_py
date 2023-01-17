#import the required libraries

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
#separate training and testing data



train=pd.read_csv("../input/titanic/train.csv")
test=pd.read_csv('../input/titanic/test.csv')
corr=train.corr()

top_co=corr.index

plt.figure(figsize=(20,20))

g=sns.heatmap(train[top_co].corr(),annot=True,cmap="RdYlGn")
#check the heatmap to analyze if there are null/none/empty values in the data



sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#imputing the data with the median

train['Age']=train['Age'].median()
#Drop columns that do not contribute to the prediction



train.drop(['PassengerId', 'Ticket'], axis=1, inplace=True)

test.drop(['PassengerId', 'Ticket'], axis=1, inplace=True)
train.isna().sum()
train['Embarked'].mode()
test.isna().sum()
test['Age']=test['Age'].median()
train.drop(['Cabin'], axis=1, inplace=True)

test.drop(['Cabin'], axis=1, inplace=True)
# Get Title from Name for train

train_title = [i.split(",")[1].split(".")[0].strip() for i in train["Name"]]

train["Title"] = pd.Series(train_title)

train["Title"].head()
g = sns.countplot(x="Title",data=train)

g = plt.setp(g.get_xticklabels(), rotation=45) 
#Convert to categorical values Title 

train["Title"] = train["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

train["Title"] = train["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})

train["Title"] = train["Title"].astype(int)



# Get Title from Name for train

test_title = [i.split(",")[1].split(".")[0].strip() for i in test["Name"]]

test["Title"] = pd.Series(test_title)

test["Title"].head()
g = sns.countplot(x="Title",data=test)

g = plt.setp(g.get_xticklabels(), rotation=45) 
#Convert to categorical values Title 

test["Title"] = test["Title"].replace(['Col', 'Dr', 'Rev', 'Dona'], 'Rare')

test["Title"] = test["Title"].map({"Master":0, "Miss":1, "Ms" : 1  ,"Mrs":1, "Mr":2, "Rare":3})

test["Title"] = test["Title"].astype(int)
train.isna().sum()
test.isna().sum()
train['Embarked'].fillna('S',inplace=True)
test.Fare = test['Fare'].fillna(train['Fare'].median())
train=train.drop('Name',axis=1)
test=test.drop('Name',axis=1)
train['FamilySize'] = train['Parch'] + train['SibSp'] + 1 
test['FamilySize'] = test['Parch'] + test['SibSp'] + 1 
# drop the variable 'SibSp' as we have already created a similar variable FamilySize

train = train.drop(['SibSp'], axis = 1)

test  = test.drop(['SibSp'], axis = 1)
train.head()
test.head()
train['Sex']=train["Sex"].replace({'male':0,'female':1})

test['Sex']=test["Sex"].replace({'male':0,'female':1})
train['Embarked']=train["Embarked"].replace({'C':0,'S':1,'Q':3})
test['Embarked']=test["Embarked"].replace({'C':0,'S':1,'Q':3})
# seperate the feature set and the target set

Y_train = train["Survived"]

X_train = train.drop(labels = ["Survived"],axis = 1)

X_test = test
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
model = RandomForestClassifier(n_estimators=250, min_samples_leaf=4, n_jobs=-1)

model.fit(X_train, Y_train)

model.score(X_train, Y_train)
X_test=test

y_predi=model.predict(X_test)
gn=pd.read_csv('../input/titanic/gender_submission.csv')
Y_test=gn['Survived'].values
accuracy_score(Y_test,y_predi)
confusion_matrix(Y_test,y_predi)
test1=pd.read_csv('../input/titanic/test.csv')
data_to_submit = pd.DataFrame({

    'PassengerId':test1['PassengerId'],

    'Survived':y_predi

})

data_to_submit.to_csv('finalsub4.csv', index = False)
data_to_submit.head()
data_to_submit.count()