import math

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.svm import SVC
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')
print('Shape of Train is : ',train.shape)

print('Shape of Test is : ',test.shape)
train.head()
train.isnull().sum()
test.head()
test.isnull().sum()
sns.countplot(x = "Survived", data = train)
sns.countplot(x = "Pclass", data = train)
sns.countplot(x = "Sex", data = train)
plt = train[['Sex', 'Survived']].groupby('Sex').mean().plot(kind = 'bar')

plt.set_ylabel('Survival Probability')
sns.countplot(x = "Age", data = train)
sns.countplot(x = "Embarked", data = train)
plt = train[['Embarked', 'Survived']].groupby('Embarked').mean().plot(kind = 'bar')

plt.set_ylabel('Survival Probability')
train['Name']
test_passengerid = test['PassengerId'].copy()
for df in [train, test] :

    df['Title'] = df['Name'].apply(lambda x : x.split(',')[1].split('.')[0], range(len(df)))
train['Title'].value_counts()
for df in [train, test] :

    df['Title'] = df['Title'].replace([' Dona', ' Don', ' Rev', ' Dr', ' Mme', ' Major', ' Lady', ' Sir', ' Mlle', ' Col', ' Capt', ' the Countess', ' Jonkheer'], 'Others')

    df['Title'] = df['Title'].replace(' Ms', ' Miss')

    df.drop(columns = ['PassengerId', 'Name', 'Ticket','Cabin'], inplace = True)
train['Title'].unique()
sns.countplot(x = 'Title', data = train)
plt = train[['Title', 'Survived']].groupby('Title').mean().plot(kind = 'bar')

plt.set_ylabel('Survival Probability')
sns.countplot(train['SibSp'])
plt = train[['SibSp', 'Survived']].groupby('SibSp').mean().plot(kind = 'bar')

plt.set_ylabel('Survival Probability')
sns.countplot(train['Parch'])
plt = train[['Parch', 'Survived']].groupby('Parch').mean().plot(kind = 'bar')

plt.set_ylabel('Survival Probability')
train.isnull().sum()
test.isnull().sum()
corr = train.corr()

corr
NaN_indexes = train['Age'][train['Age'].isnull()].index
for idx in NaN_indexes:

    pred_age = train['Age'][((train.Parch == train.iloc[idx]['Parch']) & (train.SibSp == train.iloc[idx]['SibSp']) & (train.Pclass == train.iloc[idx]['Pclass']))].median()

    

    if not np.isnan(pred_age):

        train['Age'][idx] = pred_age

    else:

        train['Age'][idx] = train['Age'].median()
train.isnull().sum()
train['Embarked'].value_counts()
train['Embarked'] = train['Embarked'].fillna('S')
NaN_indexes = test['Age'][test['Age'].isnull()].index

for idx in NaN_indexes:

    pred_age = test['Age'][((test.Parch == test.iloc[idx]['Parch']) & (test.SibSp == test.iloc[idx]['SibSp']) & (test.Pclass == test.iloc[idx]['Pclass']))].median()

    

    if not np.isnan(pred_age):

        test['Age'][idx] = pred_age

    else:

        test['Age'][idx] = test['Age'].median()
test.isnull().sum()
test['Fare'] = test['Fare'].fillna(test['Fare'].mean())
train['Title'].unique()
for df in [train, test]:

    df['Sex'] = df['Sex'].map({'male' : 0, 'female' : 1})

    df['Embarked'] = df['Embarked'].map({'Q' : 0, 'S' : 1, 'C' : 2})

    df['Title'] = df['Title'].map({' Mr' : 0, ' Mrs' : 1, ' Miss' : 2, ' Master' : 3, 'Others' : 4})
train = shuffle(train)
X_train = train.drop(columns='Survived')

y_train = train.Survived
K_fold = KFold(n_splits = 10, shuffle = True, random_state = 42)
clf = RandomForestClassifier(n_estimators = 9)

scoring = 'accuracy'

score = cross_val_score(clf, X_train, y_train, cv = K_fold, n_jobs = 1, scoring = scoring)

print(score)
clf = KNeighborsClassifier(n_neighbors = 13)

scoring = 'accuracy'

score = cross_val_score(clf, X_train, y_train, cv = K_fold, n_jobs = 1, scoring = scoring)

print(score)
clf = SVC(C=1, kernel = 'rbf', coef0 = 1)

scoring = 'accuracy'

score = cross_val_score(clf, X_train, y_train, cv = K_fold, n_jobs = 1, scoring = scoring)

print(score)
clf = RandomForestClassifier(n_estimators = 9)

clf.fit(X_train, y_train)
prediction = clf.predict(test)
submission = pd.DataFrame({'PassengerId' : test_passengerid, 'Survived' : prediction})
submission.to_csv('SVMClassifier.csv', index=False)