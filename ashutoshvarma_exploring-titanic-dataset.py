import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
train = pd.read_csv('../input/train.csv')
train
train.head()
sns.countplot(x = 'Survived',hue= 'Pclass' , data = train)
sns.boxplot(x='Pclass',y='Age',data=train)
sns.distplot(train['Age'],bins =25)
plt.figure(figsize=(10,10))

sns.heatmap(train.isnull())
train.isnull().sum()
train['Age'].value_counts()
sns.boxplot(x='Pclass',y='Age',data=train)
def impute(cols):

    Age=cols[0]

    Pclass=cols[1]

    if pd.isnull(Age):

        if Pclass==1:

            return 37

    

        elif Pclass==2:

            return 28

        else:

            return 25

    else:

        return Age

train['Age']=train[['Age','Pclass']].apply(impute,axis=1);
train['Age'].isnull().sum()
train['Cabin'].isnull().sum()
train = train.drop('Cabin',axis=1)
train['Embarked'].isnull().sum()
train['Embarked'].value_counts()
train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])
sns.heatmap(train.isnull(),cmap='Accent')
train['Sex'].value_counts()
train['Sex'].replace(to_replace=['male','female'],value=[0,1],inplace=True)
train['Embarked'].value_counts()
train['Embarked'].replace(to_replace=['S','C','Q'],value=[0,1,2],inplace=True)
train['Ticket'].value_counts()
train = train.drop('Ticket',axis=1)
plt.figure(figsize=(10,10))

sns.heatmap(train.corr())
test = pd.read_csv('../input/test.csv')
test.head()
train.head()
sns.heatmap(test.isnull())
sns.boxplot(x='Pclass',y='Age',data=test)
def impute1(cols):

    Age=cols[0]

    Pclass=cols[1]

    if pd.isnull(Age):

        if Pclass==43:

            return 

    

        elif Pclass==2:

            return 26

        else:

            return 23

    else:

        return Age

test['Age']=test[['Age','Pclass']].apply(impute1,axis=1);
test = test.drop('Cabin',axis=1)
test = test.drop('Ticket',axis=1)
test['Embarked'].value_counts()
test['Embarked'] = test['Embarked'].fillna(test['Embarked'].mode()[0])
test['Sex'].replace(to_replace=['male','female'],value=[0,1],inplace=True)
test['Embarked'].replace(to_replace=['S','C','Q'],value=[0,1,2],inplace=True)
test['Fare'].value_counts()
test['Fare'] = test['Fare'].fillna(test['Fare'].mean())
test.isnull().sum()
sns.heatmap(test.isnull())
train.head()
X=train[['PassengerId','Pclass','Age','SibSp','Parch','Sex','Embarked']]
y=train['Survived']
from sklearn.linear_model import LogisticRegression
lg = LogisticRegression()
pred = lg.fit(X,y)
pred.score(X,y)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)
lg.fit(X_test,y_test)
y_pred = lg.predict(X_test)
y_pred
from sklearn.model_selection import GridSearchCV

parameters = [{'C': [1, 10, 100, 1000],'max_iter' :[100, 500 , 1000]}]

grid_search = GridSearchCV(estimator = lg,

                           param_grid = parameters,

                           scoring = 'accuracy',

                           cv = 5,

                           n_jobs = -1)

grid_search = grid_search.fit(X_train, y_train)
grid_search.best_score_
grid_search.best_params_

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))
test

test[0:5]
X2 = test[['PassengerId','Pclass','Age','SibSp','Parch','Sex','Embarked']]
y2_pred = lg.predict(X2)

print(y2_pred)
submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": y2_pred

    })
submission.to_csv("titanic_submission.csv", index=False)
sub = pd.read_csv('titanic_submission.csv')
sub