import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

import pandas as pd 
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
#To view the Train data

train
#To view the test data

test
#To get the no.of columns and rows in train data

train.shape
#To get the no.of columns and rows in test data

test.shape
train.columns
# To get the top 5 column values

train.head()
train.describe()
#To get the missing values

print(pd.isnull(train).sum())
sns.countplot(x='Survived',data=train)
sns.countplot(x='Survived',hue='Sex',data=train)
#grouping the sex and survived to get the count of the survived and not survived by sex

train.groupby(['Survived','Sex'])['Survived'].count()
# To get the percentage of the survived by sex



print("% of women survived: " , train[train.Sex == 'female'].Survived.sum()/train[train.Sex == 'female'].Survived.count())

print("% of men survived:   " , train[train.Sex == 'male'].Survived.sum()/train[train.Sex == 'male'].Survived.count())
#Grouping Pclass and survived

train.groupby(['Survived','Pclass'])['Survived'].count()
#To get the percentage of the survival rate wrt Pclass

print("% of Pclass 1 survived",train[train.Pclass==1].Survived.sum()/train[train.Pclass==1].Survived.count())

print("% of Pclass 2 survived",train[train.Pclass==2].Survived.sum()/train[train.Pclass==2].Survived.count())

print("% of Pclass 3 survived",train[train.Pclass==3].Survived.sum()/train[train.Pclass==3].Survived.count())
#grouping SibSp and Survived

train.groupby(['Survived','SibSp'])['Survived'].count()
print("% of sibsp 0 survived",train[train.SibSp==0].Survived.sum()/train[train.SibSp==0].Survived.count())

print("% of sibsp 1 survived",train[train.SibSp==1].Survived.sum()/train[train.SibSp==1].Survived.count())

print("% of sibsp 2 survived",train[train.SibSp==2].Survived.sum()/train[train.SibSp==2].Survived.count())
train.groupby(['Survived','Parch'])['Survived'].count()

print("% of Parch 0 survived",train[train.Parch==0].Survived.sum()/train[train.Parch==0].Survived.count())

print("% of Parch 1 survived",train[train.Parch==1].Survived.sum()/train[train.Parch==1].Survived.count())

print("% of Parch 2 survived",train[train.Parch==2].Survived.sum()/train[train.Parch==2].Survived.count())
train.groupby(['Survived','Age'])['Survived'].count()
#sorting the ages into logical categories

train["Age"] = train["Age"].fillna(-0.5)

test["Age"] = test["Age"].fillna(-0.5)

bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]

labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

train['AgeGroup'] = pd.cut(train["Age"], bins, labels = labels)

test['AgeGroup'] = pd.cut(test["Age"], bins, labels = labels)



#draw a bar plot of Age vs. survival

sns.barplot(x="AgeGroup", y="Survived", data=train)

plt.show()
train.groupby(['Survived','Embarked'])['Survived'].count()
print("% of Embarked c survived",train[train.Embarked=='C'].Survived.sum()/train[train.Embarked=='C'].Survived.count())

print("% of Embarked Q survived",train[train.Embarked=='Q'].Survived.sum()/train[train.Embarked=='Q'].Survived.count())

print("% of Embarked S survived",train[train.Embarked=='S'].Survived.sum()/train[train.Embarked=='S'].Survived.count())
train.groupby(['Survived','Fare'])['Survived'].count()
train.groupby(['Survived','Cabin'])['Survived'].count()
print(pd.isnull(train).sum())
test.describe()
test.columns
train.columns
train.drop(['PassengerId','Name','Ticket','Fare','Cabin'],axis=1,inplace=True)
id=test['PassengerId']

test.drop(['PassengerId','Name','Ticket','Fare','Cabin'],axis=1,inplace=True)
train = train.fillna({"Embarked": "S"})
print(pd.isnull(train).sum())
print(pd.isnull(test).sum())
train
import statsmodels.formula.api as sm
logit_model = sm.logit('Survived~Pclass+Sex+SibSp+Parch+Embarked+AgeGroup',data = train).fit()
logit_model.summary()
logit_model.params
predictions_test=np.round(logit_model.predict(test))
predictions_test
predict_train=np.round(logit_model.predict(train))
predict_train
from sklearn.metrics import accuracy_score 

Accuracy_Score = accuracy_score(train['Survived'],predict_train)
Accuracy_Score
output = pd.DataFrame({ 'PassengerId' : id, 'Survived': predictions_test})
output.to_csv('submission.csv', index=False)
output