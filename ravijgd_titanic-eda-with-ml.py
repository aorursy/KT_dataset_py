# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import math
titanic_train= pd.read_csv('../input/titanic/train.csv')

titanic_test= pd.read_csv('../input/titanic/test.csv')
titanic_train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
sns.countplot(x ='Survived', hue='Sex', data=titanic_train)
titanic_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
sns.countplot(x ='Survived', hue='Pclass', data=titanic_train)
titanic_train['Age'].plot.hist(bins = 20, figsize=(10,5))
titanic_train['Age'].nlargest(10).plot(kind='bar');

plt.title('10 largest Ages')

plt.xlabel('Index')

plt.ylabel('Ages');
titanic_train['Age'].nsmallest(10).plot(kind='bar', color = 'C2')

plt.title('10 smallest Ages')

plt.xlabel('Index')

plt.ylabel('Ages');
(titanic_train['Age'].max(), titanic_train['Age'].min())
(titanic_test['Age'].max(), titanic_test['Age'].min())
g = sns.FacetGrid(titanic_train, col='Survived')

g.map(plt.hist, 'Age', bins=20)
grid = sns.FacetGrid(titanic_train, col='Survived', row='Pclass', height=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend();
grid = sns.FacetGrid(titanic_train, row='Embarked', size=2.2, aspect=1.6)

grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')

grid.add_legend()
titanic_train.isnull().sum()
sns.heatmap(titanic_train.isnull(), yticklabels=False, cmap='viridis')
titanic_train.loc[titanic_train.Age.isnull(), 'Age'] = titanic_train.groupby("Pclass").Age.transform('median')

titanic_test.loc[titanic_test.Age.isnull(), 'Age'] = titanic_test.groupby("Pclass").Age.transform('median')
titanic_train.Embarked.value_counts()
from statistics import mode

titanic_train['Embarked'] = titanic_train['Embarked'].fillna(mode(titanic_train['Embarked']))

titanic_test['Embarked'] = titanic_test['Embarked'].fillna(mode(titanic_test['Embarked']))
titanic_train['Fare']  = titanic_train.groupby("Pclass")['Fare'].transform(lambda x: x.fillna(x.mean()))

titanic_test['Fare']  = titanic_test.groupby("Pclass")['Fare'].transform(lambda x: x.fillna(x.mean()))
titanic_train["AgeBucket"] = titanic_train["Age"] // 10

titanic_test["AgeBucket"] = titanic_test["Age"] // 10
titanic_train['Cabin'].value_counts()
titanic_test['Cabin'].value_counts()
titanic_train['Cabin'] = titanic_train['Cabin'].fillna('U')

titanic_test['Cabin'] = titanic_test['Cabin'].fillna('U')
import re

titanic_train['Cabin'] = titanic_train['Cabin'].map(lambda x:re.compile("([a-zA-Z])").search(x).group())

titanic_test['Cabin'] = titanic_test['Cabin'].map(lambda x:re.compile("([a-zA-Z])").search(x).group())
cabin_category = {'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7, 'T':8, 'U':9}

titanic_train['Cabin'] = titanic_train['Cabin'].map(cabin_category)

titanic_test['Cabin'] = titanic_test['Cabin'].map(cabin_category)
titanic_train['Name'] = titanic_train.Name.str.extract(' ([A-Za-z]+)\.', expand = False)

titanic_test['Name'] = titanic_test.Name.str.extract(' ([A-Za-z]+)\.', expand = False)
titanic_train.rename(columns={'Name' : 'Title'}, inplace=True)

titanic_train['Title'] = titanic_train['Title'].replace(['Rev', 'Dr', 'Col', 'Ms', 'Mlle', 'Major', 'Countess', 

                                       'Capt', 'Dona', 'Jonkheer', 'Lady', 'Sir', 'Mme', 'Don'], 'Other')

                                      

titanic_test.rename(columns={'Name' : 'Title'}, inplace=True)

titanic_test['Title'] = titanic_test['Title'].replace(['Rev', 'Dr', 'Col', 'Ms', 'Mlle', 'Major', 'Countess', 

                                       'Capt', 'Dona', 'Jonkheer', 'Lady', 'Sir', 'Mme', 'Don'], 'Other')
titanic_train['familySize'] = titanic_train['SibSp'] + titanic_train['Parch'] + 1

titanic_test['familySize'] = titanic_test['SibSp'] + titanic_test['Parch'] + 1
titanic_train[["familySize", "Survived"]].groupby(['familySize']).mean()
sex= pd.get_dummies(titanic_train['Sex'])

embark = pd.get_dummies(titanic_train['Embarked'])

title = pd.get_dummies(titanic_train['Title'])

sex_t= pd.get_dummies(titanic_test['Sex'])

embark_t = pd.get_dummies(titanic_test['Embarked'])

title_t = pd.get_dummies(titanic_test['Title'])
train_data= pd.concat([titanic_train,sex,embark,title], axis=1)

test_data= pd.concat([titanic_test,sex_t,embark_t,title_t], axis=1)
train_data.columns
train_data.drop(['Title','Sex','Age','SibSp','Parch','Ticket','Cabin','Embarked'], axis=1,inplace=True)

test_data.drop(['Title','Sex','Age','SibSp','Parch','Ticket','Cabin','Embarked'], axis=1,inplace=True)
train_data
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(train_data.drop(['Survived', 'PassengerId'], axis=1), train_data['Survived'], test_size = 0.2, random_state=42)
from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression(max_iter=10000, C=50)

logreg.fit(X_train, y_train)



#R-Squared Score

print("R-Squared for Train set: {:.3f}".format(logreg.score(X_train, y_train)))

print("R-Squared for test set: {:.3f}" .format(logreg.score(X_test, y_test)))
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()



X_train_scaled = scaler.fit_transform(X_train)



# we must apply the scaling to the test set that we computed for the training set

X_test_scaled = scaler.transform(X_test)
logreg = LogisticRegression(max_iter=10000)

logreg.fit(X_train_scaled, y_train)



#R-Squared Score

print("R-Squared for Train set: {:.3f}".format(logreg.score(X_train_scaled, y_train)))

print("R-Squared for test set: {:.3f}" .format(logreg.score(X_test_scaled, y_test)))
from sklearn.svm import LinearSVC



svmclf = LinearSVC(C=100)

svmclf.fit(X_train, y_train)



print('Accuracy of Linear SVC classifier on training set: {:.2f}'

     .format(svmclf.score(X_train, y_train)))

print('Accuracy of Linear SVC classifier on test set: {:.2f}'

     .format(svmclf.score(X_test, y_test)))
svmclf = LinearSVC()

svmclf.fit(X_train_scaled, y_train)



print('Accuracy of Linear SVC classifier on training set: {:.2f}'

     .format(svmclf.score(X_train_scaled, y_train)))

print('Accuracy of Linear SVC classifier on test set: {:.2f}'

     .format(svmclf.score(X_test_scaled, y_test)))
from sklearn.svm import SVC



svcclf = SVC(gamma=0.1)

svcclf.fit(X_train, y_train)



print('Accuracy of Linear SVC classifier on training set: {:.2f}'

     .format(svcclf.score(X_train, y_train)))

print('Accuracy of Linear SVC classifier on test set: {:.2f}'

     .format(svcclf.score(X_test, y_test)))
svcclf = SVC(gamma=50)

svcclf.fit(X_train_scaled, y_train)



print('Accuracy of Linear SVC classifier on training set: {:.2f}'

     .format(svcclf.score(X_train_scaled, y_train)))

print('Accuracy of Linear SVC classifier on test set: {:.2f}'

     .format(svcclf.score(X_test_scaled, y_test)))
scaler = MinMaxScaler()



train_final = scaler.fit_transform(train_data.drop(['Survived', 'PassengerId'], axis=1))

test_final = scaler.transform(test_data.drop(['PassengerId'], axis = 1))
svcclf = SVC(gamma=50)

svcclf.fit(train_final, train_data['Survived'])
titanic_test['Survived'] = svcclf.predict(test_final)
titanic_test[['PassengerId', 'Survived']].to_csv('MySubmission_2.csv', index = False)