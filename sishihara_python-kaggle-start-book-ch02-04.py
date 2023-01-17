import numpy as np

import pandas as pd



train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')

gender_submission = pd.read_csv('../input/titanic/gender_submission.csv')



data = pd.concat([train, test], sort=False)



data['Sex'].replace(['male', 'female'], [0, 1], inplace=True)

data['Embarked'].fillna(('S'), inplace=True)

data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

data['Fare'].fillna(np.mean(data['Fare']), inplace=True)
age_avg = data['Age'].mean()

age_std = data['Age'].std()



np.random.randint(age_avg - age_std, age_avg + age_std)
np.random.randint(age_avg - age_std, age_avg + age_std)
data['Age'].fillna(data['Age'].median(), inplace=True)
delete_columns = ['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin']

data.drop(delete_columns, axis=1, inplace=True)



train = data[:len(train)]

test = data[len(train):]



y_train = train['Survived']

X_train = train.drop('Survived', axis=1)

X_test = test.drop('Survived', axis=1)
X_train.head()
y_train.head()
from sklearn.linear_model import LogisticRegression





clf = LogisticRegression(penalty='l2', solver='sag', random_state=0)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')

gender_submission = pd.read_csv('../input/titanic/gender_submission.csv')



data = pd.concat([train, test], sort=False)



data['Sex'].replace(['male', 'female'], [0, 1], inplace=True)

data['Embarked'].fillna(('S'), inplace=True)

data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

data['Fare'].fillna(np.mean(data['Fare']), inplace=True)

data['Age'].fillna(data['Age'].median(), inplace=True)
data.head()
import seaborn as sns





data['FamilySize'] = data['Parch'] + data['SibSp'] + 1

train['FamilySize'] = data['FamilySize'][:len(train)]

test['FamilySize'] = data['FamilySize'][len(train):]

sns.countplot(x='FamilySize', data = train, hue='Survived')
data['IsAlone'] = 0

data.loc[data['FamilySize'] == 1, 'IsAlone'] = 1



train['IsAlone'] = data['IsAlone'][:len(train)]

test['IsAlone'] = data['IsAlone'][len(train):]
delete_columns = ['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin']

data.drop(delete_columns, axis=1, inplace=True)



train = data[:len(train)]

test = data[len(train):]



y_train = train['Survived']

X_train = train.drop('Survived', axis=1)

X_test = test.drop('Survived', axis=1)
X_train.head()
sub = pd.read_csv('../input/titanic/gender_submission.csv')
clf.fit(X_train, y_train)

y_pred_familysize_isalone = clf.predict(X_test)



sub['Survived'] = list(map(int, y_pred_familysize_isalone))

sub.to_csv('submission_familysize_isalone.csv', index=False)



sub.head()
clf.fit(X_train.drop('FamilySize', axis=1), y_train)

y_pred_isalone = clf.predict(X_test.drop('FamilySize', axis=1))



sub['Survived'] = list(map(int, y_pred_isalone))

sub.to_csv('submission_isalone.csv', index=False)



sub.head()
clf.fit(X_train.drop('IsAlone', axis=1), y_train)

y_pred_familysize = clf.predict(X_test.drop('IsAlone', axis=1))



sub['Survived'] = list(map(int, y_pred_familysize))

sub.to_csv('submission_familysize.csv', index=False)



sub.head()
clf.fit(X_train.drop(['FamilySize', 'IsAlone'], axis=1), y_train)

y_pred = clf.predict(X_test.drop(['FamilySize', 'IsAlone'], axis=1))



sub['Survived'] = list(map(int, y_pred))

sub.to_csv('submission.csv', index=False)



sub.head()