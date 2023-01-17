import numpy as np

import pandas as pd
!ls ../input/titanic
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')

gender_submission = pd.read_csv('../input/titanic/gender_submission.csv')
gender_submission.head()
train.head()
test.head()
data = pd.concat([train, test], sort=False)
data.head()
print(len(train), len(test), len(data))
data.isnull().sum()
data['Sex'].replace(['male', 'female'], [0, 1], inplace=True)

data['Embarked'].fillna(('S'), inplace=True)

data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
data['Fare'].fillna(np.mean(data['Fare']), inplace=True)
age_avg = data['Age'].mean()

age_std = data['Age'].std()



data['Age'].fillna(np.random.randint(age_avg - age_std, age_avg + age_std), inplace=True)
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
y_pred[:20]
sub = pd.read_csv('../input/titanic/gender_submission.csv')

sub['Survived'] = list(map(int, y_pred))

sub.to_csv('submission.csv', index=False)
import pandas_profiling



train.profile_report()
import matplotlib.pyplot as plt

import seaborn as sns
plt.hist(train.loc[train["Survived"] == 0, "Age"].dropna(),

        bins=30, alpha=0.5, label="0")

plt.hist(train.loc[train["Survived"] == 1, "Age"].dropna(),

        bins=30, alpha=0.5, label="1")

plt.xlabel("Age")

plt.ylabel("count")

plt.legend(title="Survived")
sns.countplot(x="SibSp", hue="Survived", data=train)

plt.legend(loc="upper right", title = "Survived")
sns.countplot(x="Parch", hue="Survived", data=train)

plt.legend(loc="upper right", title = "Survived")
plt.hist(train.loc[train["Survived"] == 0, "Fare"].dropna(),

        bins=30, alpha=0.5, label="0")

plt.hist(train.loc[train["Survived"] == 1, "Fare"].dropna(),

        bins=30, alpha=0.5, label="1")

plt.xlabel("Fare")

plt.ylabel("count")

plt.legend(title="Survived")
sns.countplot(x="Pclass", hue="Survived", data=train)

plt.legend(loc="upper right", title = "Survived")
sns.countplot(x="Sex", hue="Survived", data=train)

plt.legend(loc="upper right", title = "Survived")
sns.countplot(x="Embarked", hue="Survived", data=train)
import os

import torch
def seed_everything(seed=1234):

    randome.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
data["FamilySize"] = data["Parch"] + data["SibSp"] + 1

train["FamilySize"] = data["FamilySize"][:len(train)]

test["FamilySize"] = data["FamilySize"][len(train):]

sns.countplot(x = "FamilySize", data=train, hue="Survived")
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators = 100, max_depth=2, random_state=0)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

y_pred
sub = pd.read_csv('../input/titanic/gender_submission.csv')

sub['Survived'] = list(map(int, y_pred))

sub.to_csv('submission.csv', index=False)