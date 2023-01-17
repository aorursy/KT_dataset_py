import pandas as pd 

import numpy as np 

import sklearn 

from sklearn.linear_model import LogisticRegression 

import matplotlib.pyplot as plt 

%matplotlib inline 

import seaborn as sns 

from sklearn.linear_model import LogisticRegression 

from sklearn.metrics import accuracy_score
train = pd.read_csv('../input/titanic/train.csv')

test= pd.read_csv('../input/titanic/test.csv')

print(train.columns.values)
#Check out the first 5 rows of the training dataset

train.head()
# Info about data frame dimensions, column types, and file size

print(train.info())
#Summary of statistics for training data :count, mean, std,...

train.describe()
#rate of women survived

women = train.loc[train.Sex == 'female']["Survived"]

rate_women = sum(women)/len(women)

print (rate_women)
#rate of men survived

men = train.loc[train.Sex == 'male']["Survived"]

rate_men = sum(men)/len(men)

print (rate_men)
sns.barplot(x='Pclass',y='Survived', data=train, ci=None)
sns.barplot(x="Sex",y="Survived",data=train,ci=None)
sns.countplot(x="Survived",data=train)
sns.countplot(x="Sex", data=train)
sns.distplot(train['Age'].dropna(),kde = False,hist_kws=dict(alpha=1))

plt.title("Age distribution of Titanic Passengers")
train.isnull().sum()
train["Age"]=train["Age"].fillna(train['Age'].median())

train["Embarked"]=train["Embarked"].fillna(train["Embarked"].value_counts().idxmax())
train.isnull().sum()
# dropping name, tickiet, Cabin

train=train.drop(labels=['Name', 'Ticket', 'Cabin'], axis=1)
sex = pd.get_dummies(train['Sex'], drop_first=True)

embarked = pd.get_dummies(train['Embarked'], drop_first=True)

Pclass= pd.get_dummies(train['Pclass'], drop_first=True)



train.drop(['Sex', 'Embarked','Pclass'], axis=1, inplace=True)



train = pd.concat([train, sex, embarked,Pclass], axis=1)
train.rename(columns={'male':'Sex', 'Q':'Queenstown', 'S':'Southampton',2:'Pclass2',3:'Pclass3'}, inplace=True)

train.head()
test.isnull().sum()
test["Age"]=test["Age"].fillna(test['Age'].median())

test["Fare"]=test["Fare"].fillna(test["Fare"].median())
test.isnull().sum()
test=test.drop(labels=['Name', 'Ticket', 'Cabin'], axis=1)
sex = pd.get_dummies(test['Sex'], drop_first=True)

embarked = pd.get_dummies(test['Embarked'], drop_first=True)

Pclass= pd.get_dummies(test['Pclass'], drop_first=True)



test.drop(['Sex', 'Embarked','Pclass'], axis=1, inplace=True)

test = pd.concat([test, sex, embarked,Pclass], axis=1)
test.rename(columns={'male':'Sex', 'Q':'Queenstown', 'S':'Southampton',2:'Pclass2',3:'Pclass3'}, inplace=True)
test.head()
X_train = train.drop(['PassengerId', 'Survived'], axis = 1)

y_train = train['Survived']

X_test = test.drop("PassengerId", axis=1).copy()
#Shpe of our training and tsting

X_train.shape, y_train.shape, X_test.shape
#Define Model

model = LogisticRegression()

model.fit(X_train,y_train)
predictions = model.predict(X_test)

accuracy = model.score(X_train, y_train)
#accurcy on training set

print(" Accuracy is : {0} ".format(accuracy))
Submission = pd.DataFrame({

    "PassengerId": test['PassengerId'],

    "Survived": predictions

})



Submission.to_csv('Submission.csv', index = False)