# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
def input_files():
    for dirname, _, filenames in os.walk('/kaggle/input'):
        for filename in filenames:
            print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
input_files()
# Importing Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
train_df = pd.read_csv('/kaggle/input/titanic/train.csv')
test_df = pd.read_csv('/kaggle/input/titanic/test.csv')
combine = [train_df, test_df]
train_df.tail()
test_df.tail()
print(combine[0].shape, combine[1].shape)
print(train_df.columns.values)
print(test_df.columns.values)
train_df.describe()
train_df.describe(include=['O'])
train_df[['Pclass', 'Survived']].groupby('Pclass', as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[['Sex','Survived']].groupby('Sex', as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[['Embarked','Survived']].groupby('Embarked', as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[['Parch','Survived']].groupby('Parch', as_index=False).mean().sort_values(by='Survived', ascending=False)
grid = sns.FacetGrid(train_df, col='Survived', height=3.5, aspect=1.5)
grid.map(plt.hist, 'Age', bins=20)
grid = sns.FacetGrid(train_df, row='Pclass', col='Survived', height=3.5, aspect=1.5)
grid.map(plt.hist, 'Age', bins=20)
grid = sns.FacetGrid(train_df, 'Embarked', height=3.5, aspect=1.5)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()
grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', height=3.5, aspect=1.5)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=0.5, ci=None)
print("Before Drop:", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)
train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]
print("After Drop:", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.')
print("Check Shape:", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)
train_df.Title.unique()
test_df.Title.unique()
set(list(train_df.Title.unique()) + list(test_df.Title.unique()))
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'rare')
    dataset['Title'] = dataset['Title'].replace(['Mlle', 'Ms'], 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
train_df[['Title', 'Survived']].groupby('Title', as_index=False).mean().sort_values('Survived', ascending=False)
title_mapping = {'Mr':1, 'Miss':2, 'Mrs':3, 'Master':4, 'rare':5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
train_df.head()
set(list(train_df.Title.unique()) + list(test_df.Title.unique()))
print("Before Drop:", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)
train_df = train_df.drop(['PassengerId', 'Name'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]
print("After Drop:", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map({'male':0, 'female':1}).astype(int)
train_df.head()
train_df.info()
grid = sns.FacetGrid(train_df, col='Pclass', row='Sex', height=3.5, aspect=1.5)
grid.map(plt.hist, 'Age', bins=20)
guess_ages = np.zeros((2,3))
guess_ages
for dataset in combine:
    for i in range(0,2):
        for j in range(0,3):
            guess_df = dataset[(dataset['Sex']==i) & (dataset['Pclass']==j+1)]['Age'].dropna()
            age_guess = guess_df.median()
            guess_ages[i,j] = int(age_guess)
    for i in range(0,2):
        for j in range(0,3):
            dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1), 'Age'] = guess_ages[i,j]
    dataset['Age'] = dataset['Age'].astype(int)
train_df.info()
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
train_df[['AgeBand','Survived']].groupby('AgeBand', as_index=False).mean()
for dataset in combine:
    dataset.loc[dataset['Age']<=16, 'Age'] = 0
    dataset.loc[(dataset['Age']>16) & (dataset['Age']<=32), 'Age'] = 1
    dataset.loc[(dataset['Age']>32) & (dataset['Age']<=48), 'Age'] = 2
    dataset.loc[(dataset['Age']>48) & (dataset['Age']<=64), 'Age'] = 3
    dataset.loc[dataset['Age']>64, 'Age'] = 4
train_df.head()
train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]
train_df.head()
for dataset in combine:
    dataset['FamilySize'] = dataset['Parch'] + dataset['SibSp'] + 1
train_df[['FamilySize', 'Survived']].groupby('FamilySize', as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize']==1, 'IsAlone'] = 1
train_df[['IsAlone', 'Survived']].groupby('IsAlone', as_index=False).mean().sort_values(by='Survived', ascending=False)
print("Before Drop:", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)
train_df = train_df.drop(['SibSp', 'Parch', 'FamilySize'], axis = 1)
test_df = test_df.drop(['SibSp', 'Parch', 'FamilySize'], axis=1)
combine=[train_df, test_df]
print("After Drop:", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)
for dataset in combine:
    dataset['Age*Class'] = dataset['Age'] * dataset['Pclass']
train_df.head()
freq_port = train_df.Embarked.mode()[0]
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
train_df[['Embarked', 'Survived']].groupby('Embarked', as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map({'S':0, 'Q':1, 'C':2}).astype(int)
train_df.head()
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
test_df.head()
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
train_df[['FareBand', 'Survived']].groupby('FareBand', as_index=False).mean()
for dataset in combine:
    dataset.loc[(dataset['Fare']<=7.91),'Fare'] = 0
    dataset.loc[(dataset['Fare']>7.91) & (dataset['Fare']<=14.454),'Fare'] = 1
    dataset.loc[(dataset['Fare']>14.454) & (dataset['Fare']<=31.0),'Fare'] = 2
    dataset.loc[(dataset['Fare']>31.0) & (dataset['Fare']<=512.329),'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
train_df.head()
train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]
train_df.head()
X_train = train_df.drop(['Survived'], axis=1)
y_train = train_df['Survived']
X_test = test_df.drop(['PassengerId'], axis=1).copy()
print("Check Shape:", X_train.shape, y_train.shape, X_test.shape)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
acc_logreg = round(logreg.score(X_train, y_train)*100,2)
print(acc_logreg, " %")
coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Features']
coeff_df['Correlation'] = logreg.coef_[0]
coeff_df.sort_values(by='Correlation', ascending=False)
svc = SVC()
svc.fit(X_train, y_train)
acc_svc = round(svc.score(X_train, y_train)*100,2)
print(acc_svc, " %")
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
acc_knn = round(knn.score(X_train, y_train)*100,2)
print(acc_knn, " %")
gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
acc_gaussian = round(gaussian.score(X_train, y_train)*100, 2)
print(acc_gaussian, " %")
perceptron = Perceptron()
perceptron.fit(X_train, y_train)
acc_perceptron = round(perceptron.score(X_train, y_train)*100, 2)
print(acc_perceptron, " %")
linear_svc = LinearSVC()
linear_svc.fit(X_train, y_train)
acc_linear_svc = round(linear_svc.score(X_train, y_train)*100, 2)
print(acc_linear_svc, " %")
sgd = SGDClassifier()
sgd.fit(X_train, y_train)
acc_sgd = round(sgd.score(X_train, y_train)*100, 2)
print(acc_sgd, " %")
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
acc_decision_tree = round(decision_tree.score(X_train, y_train)*100, 2)
print(acc_decision_tree, " %")
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
acc_random_forest = round(random_forest.score(X_train, y_train)*100, 2)
print(acc_random_forest, " %")
models = pd.DataFrame({
    'Models':[
        'Support Vector Machines',
        'KNN',
        'Logistic Regression',
        'Random Forest',
        'Naive Bayes',
        'Perceptron',
        'Stochastic Gradient Decent',
        'Linear SVC',
        'Decision Tree',
    ],
    'Scores':[
        acc_svc,
        acc_knn,
        acc_logreg,
        acc_random_forest,
        acc_gaussian,
        acc_perceptron,
        acc_sgd,
        acc_linear_svc,
        acc_decision_tree,
    ]
})
models.sort_values(by='Scores', ascending=False)
y_pred = random_forest.predict(X_test)
submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived':y_pred
})
submission
submission.to_csv('submission.csv', index=False)