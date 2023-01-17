import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')

train.head()
train.describe()
sns.heatmap(train.isnull(), yticklabels=False, cbar=False)
sns.countplot(x='Survived',hue='Sex',data=train)
sns.countplot(x='Survived', hue='Pclass', data=train)
sns.distplot(train['Age'].dropna(), kde=False)
sns.countplot(x='Parch',hue='Pclass',data=train)
train.drop('Cabin', axis=1, inplace=True)

test.drop('Cabin', axis=1, inplace=True)
train['Title'] = train['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

test['Title'] = test['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

accepted_title = train['Title'].value_counts()[train['Title'].value_counts()>30]

def check_in_accepted_title(title):

    if title in accepted_title:

        return title

    else:

        return ''

train['Title'] = train['Title'].apply(lambda title: check_in_accepted_title(title))

test['Title'] = test['Title'].apply(lambda title: check_in_accepted_title(title))

train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
train.drop('Name', axis=1, inplace=True)

test.drop('Name', axis=1, inplace=True)

grid = sns.FacetGrid(train, row='Pclass', col='Sex')

grid.map(plt.hist, 'Age')

grid.add_legend()
data_all = [train, test]

sex_all = ['male', 'female']

pclass_all = [1, 2, 3]



for d in data_all:

    for sex in sex_all:

        for pclass in pclass_all:

            specific_age = d[(d['Sex'] == sex) & (d['Pclass'] == pclass)]['Age'].dropna()

            d.loc[(d['Age'].isnull()) & (d['Sex'] == sex) & (d['Pclass'] == pclass), 'Age'] = specific_age.median()
train['Embarked'].fillna(train['Embarked'].dropna().mode()[0], inplace=True)
[train.isnull().sum(), test.isnull().sum()]
test['Fare'].fillna(test['Fare'].dropna().mean(), inplace=True)
sex = pd.get_dummies(train['Sex'], drop_first=True)

embark = pd.get_dummies(train['Embarked'], drop_first=True)

pclass = pd.get_dummies(train['Pclass'], drop_first=True)

title = pd.get_dummies(train['Title'], drop_first=True)

train = pd.concat([train, sex, embark, pclass, title], axis=1)

sex2 = pd.get_dummies(test['Sex'], drop_first=True)

embark2 = pd.get_dummies(test['Embarked'], drop_first=True)

pclass2 = pd.get_dummies(test['Pclass'], drop_first=True)

title2 = pd.get_dummies(test['Title'], drop_first=True)

test = pd.concat([test, sex2, embark2, pclass2, title2], axis=1)

# drop reductant columns and also Ticket...

train.drop(['Sex', 'Embarked', 'Pclass', 'Title', 'Ticket'], axis=1, inplace=True)

test.drop(['Sex', 'Embarked', 'Pclass', 'Title', 'Ticket'], axis=1, inplace=True)
from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression(solver='lbfgs')

logmodel.fit(train.drop('Survived',axis=1), train['Survived'])

predictions = logmodel.predict(test)
# prepare for submission file...

submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": predictions})

submission.to_csv('/kaggle/working/submission.csv', index=False)