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
import sklearn
import matplotlib.pyplot as plt 
import plotly
import plotly.express as px
import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)

# machine learning algorithms 
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
gender_submission = pd.read_csv('../input/titanic/gender_submission.csv')
train.info()
# check missing values in Age feature in test set
missing_values = 0
for i in train['Embarked'].isnull():
    if i == True:
        missing_values +=1
print(missing_values)
train['Pclass'].value_counts()
train['Age'].value_counts()
train.describe()
# variances for age and fare
print('Age\'s variance: {}'.format(train['Age'].var()))
print('Fare\'s variance: {}'.format(train['Fare'].var()))
# plot Age feature to check the disribution
train[['Age']].iplot(kind='hist',bins=27,title='Age Distribution',xTitle='Age',yTitle='Frequency')
# plot Fare distribution 
train['Fare'].iplot(kind='hist',bins=30,xTitle='Fare',yTitle='Frequency',title='Fare Distribution')
# plot sibsp to check the distribution
train['SibSp'].iplot(kind='hist',xTitle='Sibsp',yTitle='frequency',title='Sibling or Spouses Distribution')
# plot Parch to check the distribution
train['Parch'].iplot(kind='hist',xTitle='Parch',yTitle='Frequency')
# check the number of passengers that has 3 or 4 or 5 parents or children
x =0
for i in train['Parch'].values:
    if i == 5:
        x += 1
print(x)
train['Fare'].describe()
train.columns
# create a copy of train data to be modified 
train_insight = train.copy()
# convert the values in Sex attribute
train_insight = train_insight.replace({'Sex':{'female':1,'male':0}})
train['Embarked'].value_counts()
# convert the values in Embarked attribute
train_insight = train_insight.replace({'Embarked':{'C':0,'Q':2,'S':3}}) 
train_corr = train_insight.corr()
train_corr['Survived'].sort_values(ascending=False)
print('Shapes train and test data ')
print('Before: train data{} --- test data{}'.format(train.shape, test.shape))
# remove Ticket and Cabin attributes from train and test sets
train = train.drop(['Ticket','Cabin','PassengerId'], axis=1)
test = test.drop(['Ticket','Cabin'], axis=1)
print('After: train data{} --- test data{}'.format(train.shape, test.shape))
train['Family Size'] = train['Parch'] + train['SibSp'] + 1
test['Family Size'] = train['Parch'] + train['SibSp'] + 1
train_corr = train.corr()
train_corr['Survived'].sort_values(ascending=False)
# train set
train['Is Alone'] = 0
train.loc[train['Family Size']==1, 'Is Alone'] = 1

# test set
test['Is Alone'] = 0
test.loc[test['Family Size']==1, 'Is Alone'] = 1
train_corr = train.corr()
train_corr['Survived'].sort_values(ascending=False)
train = train.drop(['SibSp','Parch','Family Size'], axis=1)
test = test.drop(['SibSp','Parch','Family Size'],axis=1)
train.head()
train['Title'] = train.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
test['Title'] = test.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
pd.crosstab(train['Title'], train['Survived'])
combine = [train, test] # combine is a list

for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Capt','Col','Countess','Don','Dr','Jonkheer','Lady','Major','Mme','Rev','Sir'],'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle','Miss')
    dataset['Title'] = dataset['Title'].replace('Ms','Miss')
    dataset['Title'] = dataset['Title'].replace('Mme','Mrs')

train[['Title','Survived']].groupby(['Title'], as_index=False).mean()
title_mapping = {'Mr':1, 'Miss':2, 'Mrs':3, 'Master':4, 'Rare':5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
    dataset['Title'] = dataset['Title'].astype(int)
train = train.drop('Name', axis=1)
test = test.drop('Name', axis=1)
train_corr = train.corr()
train_corr['Survived']
train['Fare Interval'] = pd.qcut(train['Fare'], 5)
train[['Fare Interval','Survived']].groupby(['Fare Interval'], as_index=False).mean()
# handle missing values in Fare attribute
test['Fare'].fillna(test['Fare'].dropna().median(), inplace=True)
train['Fare'].fillna(test['Fare'].dropna().median(), inplace=True)
train[['Fare']].info()
combine = [train, test]
for dataset in combine:
    dataset.loc[dataset['Fare'] <= 7.854, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.854) & (dataset['Fare'] <= 10.5), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 10.5) & (dataset['Fare'] <= 21.679), 'Fare'] = 2
    dataset.loc[(dataset['Fare'] > 21.679) & (dataset['Fare'] <= 39.688), 'Fare'] = 3
    dataset.loc[(dataset['Fare'] > 39.688) & (dataset['Fare'] <= 512.329), 'Fare'] = 4
    # convert float values in Fare attribute to int 
    dataset['Fare'] = dataset['Fare'].astype(int)

# remove Fare Interval attribute
train = train.drop('Fare Interval', axis=1)

train['Age Interval'] = pd.cut(train['Age'], 5)
train[['Age Interval', 'Survived']].groupby(['Age Interval'], as_index=False).mean()
# handle missing values in Age attribute
train['Age'].fillna(train['Age'].dropna().mean(), inplace=True)
test['Age'].fillna(train['Age'].dropna().mean(), inplace=True)
combine = [train, test]
for dataset in combine:
    dataset.loc[dataset['Age'] <= 16.336, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16.336) & (dataset['Age'] <= 32.252), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32.252) & (dataset['Age'] <= 48.168), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48.168) & (dataset['Age'] <= 64.084), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 64.084) & (dataset['Age'] <= 80.0), 'Age'] = 4
    dataset['Age'] = dataset['Age'].astype(int)

# remove Age Interval
train = train.drop('Age Interval', axis=1)
train['Embarked'].value_counts()
# fill NAN with 'S' because 'S' is for which the value occurs frequently
train['Embarked'].fillna('S', inplace=True)
combine = [train, test]
# convert Embarked and Sex 
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].replace(['C','S','Q'],[0,1,2])
    dataset['Sex'] = dataset['Sex'].replace(['female','male'],[0,1])
    dataset['Embarked'] = dataset['Embarked'].astype(int)
    dataset['Sex'] = dataset['Sex'].astype(int)
X_train = train.drop('Survived',axis=1)
y_train = train['Survived']
X_test = test.drop('PassengerId', axis=1).copy()
# create a model
logreg = LogisticRegression()
# fit train data to model
logreg.fit(X_train, y_train)
# predict based test data
Y_prediction = logreg.predict(X_test)
# calculate the accuracy
logreg_accuracy = logreg.score(X_train, y_train)
logreg_accuracy
# create a model
gaussian = GaussianNB()
# fit it
gaussian.fit(X_train, y_train)
# predict
Y_prediction = gaussian.predict(X_test)
# calculate the accuracy
gaussian_accuracy = gaussian.score(X_train, y_train)
gaussian_accuracy
# create a model
knn = KNeighborsClassifier()
# fit it
knn.fit(X_train, y_train)
# predict
y_prediction = knn.predict(X_test)
# calculate the accuracy
knn_accuracy = knn.score(X_train, y_train)
knn_accuracy
# create a model
svc = SVC()
# fit it
svc.fit(X_train, y_train)
# predict
y_prediction = svc.predict(X_test)
# calculate the accuracy
svc_accuracy = svc.score(X_train, y_train)
svc_accuracy
# create a model
decision_tree = DecisionTreeClassifier()
# fit it
decision_tree.fit(X_train, y_train)
# predict
y_prediction = decision_tree.predict(X_test)
# accuracy
decision_tree_accuracy = decision_tree.score(X_train, y_train)
decision_tree_accuracy
model = pd.DataFrame({
    'Model':['Logistics Regression','Naive Bayes Classifier','Nearest Neighbors', 'Support Vector Machine','Decision Trees'],
    'Score':[logreg_accuracy, gaussian_accuracy, knn_accuracy, svc_accuracy, decision_tree_accuracy]
})
model.sort_values(by='Score', ascending=False)
test
submission = pd.DataFrame({
    'PassengerId':test['PassengerId'],
    'Survived': y_prediction
})
submission.to_csv('../submission.csv', index=False)
