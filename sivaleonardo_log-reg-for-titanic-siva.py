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
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

data_frame = [train_df, test_df]

print(train_df.columns.values)
print(test_df.columns.values)
train_df.head()
test_df.head()
train_df.describe()
train_df.shape, test_df.shape
train_df['Survived'].value_counts()
train_df['Pclass'].value_counts()
train_df['Sex'].value_counts()
test_df['Sex'].value_counts()
train_df[['Pclass', 'Survived']].groupby(['Pclass'],as_index=False).mean().sort_values(by='Survived', ascending=False)

train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[['Embarked','Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df['AgeRange'] = pd.cut(train_df['Age'], 5)

train_df[['AgeRange', 'Survived']].groupby(['AgeRange'], as_index=False).mean().sort_values(by='Survived',ascending=False)
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
train_df['FareRange'] = pd.qcut(train_df['Fare'], 4)

train_df[['FareRange', 'Survived']].groupby(['FareRange'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)

test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)

data_frame = [train_df, test_df]
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)

test_df = test_df.drop(['Name'], axis=1)

data_frame = [train_df, test_df]

train_df.shape, test_df.shape
for dataset in data_frame:

    dataset.loc[dataset['Age'] <= 16.336, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16.336) & (dataset['Age'] <= 32.252), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32.252) & (dataset['Age'] <= 48.168), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48.168) & (dataset['Age'] <= 64.084), 'Age'] = 3

    dataset.loc[(dataset['Age'] > 64.084) & (dataset['Age'] <= 80.00), 'Age'] = 4

    

train_df.head()

    
train_df = train_df.drop(['AgeRange'], axis=1)

data_frame = [train_df, test_df]

train_df.head()
for dataset in data_frame:

    dataset['Embarked'] = dataset['Embarked'].map({'S':0, 'C':1, 'Q':2}).astype(int)

    

train_df.head()
train_df['Embarked'].value_counts()
mode_value_embarked = train_df.Embarked.dropna().mode()[0]

mode_value_embarked
for dataset in data_frame:

    dataset['Embarked'] = dataset['Embarked'].fillna(mode_value_embarked)
for dataset in data_frame:

    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)



train_df.head()
for dataset in data_frame:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31.0), 'Fare'] = 2

    dataset.loc[(dataset['Fare'] > 31.0), 'Fare'] = 3

    

    dataset['Fare'] = dataset['Fare'].astype(int)

    

train_df = train_df.drop(['FareRange'], axis=1)



data_frame=[train_df, test_df]



train_df.head()
train_df.shape, test_df.shape
test_df.head()
for dataset in data_frame:

    dataset['Sex'] = dataset['Sex'].map({'male':0, 'female':1}).astype(int)



train_df.head()
test_df.head()
from sklearn.linear_model import LogisticRegression

logit = LogisticRegression()

X_train = train_df.drop('Survived', axis=1)

Y_train = train_df['Survived']

X_test = test_df.drop('PassengerId', axis=1)



X_train.shape, Y_train.shape, X_test.shape
logit.fit(X_train, Y_train)

Y_pred = logit.predict(X_test)

acc_log = round(logit.score(X_train, Y_train)*100, 2)

acc_log
print(train_df)
train_df['Age'].value_counts()

train_df['SibSp'].value_counts(), train_df['Parch'].value_counts(), train_df['Fare'].value_counts()
test_df['Age'].value_counts(), test_df['SibSp'].value_counts(), test_df['Parch'].value_counts()
for dataset in data_frame:

    dataset['Age'].fillna(dataset['Age'].median(), inplace=True)

    

print (train_df)
X_train = train_df.drop('Survived', axis=1)

Y_train = train_df['Survived']

X_test = test_df.drop('PassengerId', axis=1)



logit.fit(X_train, Y_train)

Y_pred = logit.predict(X_test)

logit_score = logit.score(X_train, Y_train)*100

print(str(logit_score)+' %')
feature_correlation = pd.DataFrame(train_df.columns.delete(0))

feature_correlation.columns = ['Feature']

feature_correlation["Correlation"] = pd.Series(logit.coef_[0])



feature_correlation.sort_values(by='Correlation', ascending=False)
from sklearn.svm import SVC, LinearSVC



svc=SVC()

svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_test)



svc_score = svc.score(X_train, Y_train)*100

print(str(svc_score)+' %')
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors = 10)

knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)



knn_score = knn.score(X_train, Y_train)*100

print(str(knn_score)+' %')
