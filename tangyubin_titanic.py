# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
print(os.listdir('./'))
# Any results you write to the current directory are saved as output.
data_train = pd.read_csv('../input/train.csv')
data_test = pd.read_csv('../input/test.csv')
data_train.head(2)
data_train.info()
print('-'*40)
data_test.info()
data_train.describe()
data_test.describe()
data_train.describe(include=['O'])
data_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()\
    .sort_values(by='Survived', ascending=False)
data_train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean()\
    .sort_values(by='Survived', ascending=False)
data_train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False)\
    .mean().sort_values(by='Survived', ascending=False)
data_train.columns.get_values()
data_train[['SibSp', 'Survived']].groupby('SibSp', as_index=False)\
    .mean().sort_values(by='Survived', ascending=False)
data_train[['Parch', 'Survived']].groupby('Parch', as_index=False).mean().sort_values(by='Survived', ascending=False)
g = sns.FacetGrid(data_train, col='Survived')
g.map(plt.hist, 'Age', bins=20)
g = sns.FacetGrid(data_train, col='Survived')
g.map(plt.hist, 'Fare', bins=20)
g = sns.FacetGrid(data_train, row = 'Pclass', col = 'Survived')
g.map(plt.hist, 'Age', bins = 20, alpha=0.7)
g.add_legend()
g = sns.FacetGrid(data_train, row='Embarked', size=2.2, aspect=1.6)
g.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', plette='deep')
g.add_legend()
data_train.columns.get_values()
g = sns.FacetGrid(data_train, row = 'Pclass', col = 'Survived', height = 2.2, aspect = 1.6)
g.map(sns.barplot, 'Sex', 'Fare', alpha = 0.5, ci = None)
g.add_legend()
data_combine = [data_train, data_test]
for data in data_combine:
    data.drop(['Cabin', 'PassengerId', 'Ticket'], axis = 1, inplace=True)
print(data_train.shape, data_test.shape)
for data in data_combine:
    data['Title'] = data.Name.str.extract('([a-zA-Z]+)\.', expand = False)
for data in data_combine:
    data.drop(['Name'], axis=1, inplace=True)
data_train.head(2)
data_test.head(2)
pd.crosstab(data_train['Title'], data_train['Sex'])
for dataset in data_combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
     'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
data_train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()\
    .sort_values(by = 'Survived', ascending = False)
data_train.head()
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in data_combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
sns.barplot(data = data_train, x = 'Title', y = 'Survived')
for dataset in data_combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
data_train.info()
g = sns.FacetGrid(data_train, 'Pclass', 'Sex', size=2.2, aspect=1.6)
g.map(plt.hist, 'Age', bins=20, alpha=0.5)
g.add_legend()
guess_ages = np.zeros((2, 3))
guess_ages
for i in range(2):
    for j in range(3):
        guess_ages[i][j] = data_train[(data_train['Sex'] == i) & \
                                      (data_train['Pclass'] == j + 1)]['Age'].dropna().median()
        guess_ages[i][j] = (guess_ages[i][j] / 0.5 + 0.5) * 0.5
guess_ages
for i in range(2):
    for j in range(3):
        data_train.loc[(data_train['Sex'] == i) & \
                    (data_train['Pclass'] == j + 1)\
                  &(data_train.Age.isnull()), 'Age'] = guess_ages[i][j]
for i in range(2):
    for j in range(3):
        data_test.loc[(data_test['Sex'] == i) & \
                    (data_test['Pclass'] == j + 1)\
                  &(data_test.Age.isnull()), 'Age'] = guess_ages[i][j]
for data in data_combine:
    data.Age = data.Age.astype(int)
print(data_train.info())
print('-'*40)
print(data_test.info())
data_train['AgeBand'] = pd.cut(data_train['Age'], 5)
data_train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean()
for dataset in data_combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']
data_train.head()
data_train = data_train.drop(['AgeBand'], axis=1)
data_test.head()
data_combine = [data_train, data_test]
for data in data_combine:
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

data_train.head()
data_train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=True).mean().sort_values(by='Survived', ascending=False)
for data in data_combine:
    data['IsAlone'] = 0
    data.loc[data['FamilySize'] == 1, 'IsAlone'] = 1
data_train[['IsAlone', 'Survived']].groupby('IsAlone', as_index = False).mean()
data_train = data_train.drop(['FamilySize', 'Parch', 'SibSp'], axis=1)
data_test = data_test.drop(['FamilySize', 'Parch', 'SibSp'], axis=1)
data_combine = [data_train, data_test]
data_train.head()
data_train.columns.get_values()
freq_port = data_train.Embarked.dropna().mode()[0]
for data in data_combine:
    data['Embarked'] = data['Embarked'].fillna(freq_port)
data_train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean()
for dataset in data_combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

data_train.head()
for data in data_combine:
    data['Fare'].fillna(data['Fare'].dropna().median(), inplace=True)
data_train['FareBand'] = pd.qcut(data_train['Fare'], 4)
data_train[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean()
for dataset in data_combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

data_train = data_train.drop(['FareBand'], axis=1)
data_combine = [data_train, data_test]
    
data_train.head(10)
print(data_train.info())
print('-'*40)
print(data_test.info())
X_train = data_train.drop('Survived', axis=1)
y_train = data_train['Survived']
X_train.shape,y_train.shape
data_test.head()
data_test['Pclass'] = data_test['Pclass'].fillna(data_test['Pclass'].dropna().median())
data_test['Pclass'] = data_test['Pclass'].astype(int)
data_test['Sex'] = data_test['Sex'].fillna(data_test['Sex'].dropna().median()).astype(int)
data_test['Age'] = data_test['Age'].fillna(data_test['Age'].dropna().median()).astype(int)
data_test['Title'] = data_test['Title'].fillna(data_test['Title'].dropna().median()).astype(int)
data_test.info()
X_test = data_test
X_train.shape, y_train.shape, X_test.shape
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
logscore = round(logreg.score(X_train, y_train) * 100, 2)
logscore
coeff_df = pd.DataFrame(data_train.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df['Correlation'] = pd.Series(logreg.coef_[0])
coeff_df.sort_values(by='Correlation', ascending=False)
test_df = pd.read_csv('../input/test.csv')
test_df.head()
#y_pred = y_pred[:-1]
#y_pred.shape
submission = pd.DataFrame({
    'PassengerId':test_df['PassengerId'],
    'Survived':y_pred
})
submission.head()
submission.to_csv('submission.csv', index=False)
data_test.tail()
os.listdir('./')
