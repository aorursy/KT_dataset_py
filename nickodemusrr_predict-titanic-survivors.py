# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# reading datasets using pandas

data_train = pd.read_csv('../input/titanic/train.csv')

data_test = pd.read_csv('../input/titanic/test.csv')

data_train.head()
data_train.info()
sns.set_style('darkgrid')

ax = sns.countplot(data_train['Survived'])

for p in ax.patches:

        ax.annotate('{} ({:.1f}%)'.format(p.get_height(), 100* p.get_height()/len(data_train['Survived'])), 

                    (p.get_x()+0.2, p.get_height()-30))
plt.figure(figsize=(12,6))



plt.subplot(121)

ax = sns.countplot(data = data_train, x='Sex')

for p in ax.patches:

        ax.annotate('{} ({:.1f}%)'.format(p.get_height(), 100* p.get_height()/len(data_train['Sex'])), 

                    (p.get_x()+0.2, p.get_height()-30))



plt.subplot(122)

ax = sns.countplot(data = data_train, x='Survived', hue='Sex')

for p in ax.patches:

        ax.annotate('{} ({:.1f}%)'.format(p.get_height(), 100* p.get_height()/len(data_train['Survived'])), 

                    (p.get_x()+0.01, p.get_height()-30))

        

plt.tight_layout()
plt.figure(figsize=(10,6))

ax = sns.countplot(data = data_train, x='Survived', hue='Pclass')

for p in ax.patches:

        ax.annotate('{} ({:.1f}%)'.format(p.get_height(), 100* p.get_height()/len(data_train['Survived'])), 

                    (p.get_x()+0.03, p.get_height()-30))
plt.figure(figsize=(10,6))

ax = sns.countplot(data = data_train, x='Survived', hue='Embarked')

for p in ax.patches:

        ax.annotate('{} ({:.1f}%)'.format(p.get_height(), 100* p.get_height()/len(data_train['Survived'])), 

                    (p.get_x()+0.03, p.get_height()-25))
plt.figure(figsize=(10,6))

ax = sns.countplot(data = data_train, x='Sex', hue='Pclass')

for p in ax.patches:

        ax.annotate('{} ({:.1f}%)'.format(p.get_height(), 100* p.get_height()/len(data_train['Survived'])), 

                    (p.get_x()+0.03, p.get_height()-30))
plt.figure(figsize=(12,6))

ax = sns.countplot(data = data_train, x='Embarked', hue='Pclass')

for p in ax.patches:

        ax.annotate('{} ({:.1f}%)'.format(p.get_height(), 100* p.get_height()/len(data_train['Survived'])), 

                    (p.get_x()+0.01, p.get_height()+5))
data_train.drop(['PassengerId', 'Survived', 'Pclass'], axis=1).describe()
sns.distplot(data_train['Age'].dropna(), kde=False)
plt.figure(figsize=(10,6))

plt.subplot(121)

sns.countplot(data=data_train, x='SibSp')

plt.subplot(122)

sns.countplot(data=data_train, x='Parch')

plt.tight_layout()
sns.distplot(data_train['Fare'])
data_train.isnull().sum()
figure = plt.figure(figsize=(10,6))

sns.boxplot(data=data_train, x='Pclass', y='Age')
first_mean = round(data_train[data_train['Pclass'] == 1]['Age'].dropna().mean())

second_mean = round(data_train[data_train['Pclass'] == 2]['Age'].dropna().mean())

third_mean = round(data_train[data_train['Pclass'] == 3]['Age'].dropna().mean())



# creating function to fill missing age

def filling(col):

    Age = col[0]

    Pclass = col[1]

    if pd.isnull(Age):

        if Pclass == 1:

            return first_mean

        elif Pclass == 2:

            return second_mean

        else:

            return third_mean

    else:

        return Age



data_train['Age'] = data_train[['Age', 'Pclass']].apply(filling, axis=1)
data_train['Embarked'].mode()
data_train['Embarked'].fillna('S', inplace=True)

data_train.isnull().sum()
sex = pd.get_dummies(data_train['Sex'],drop_first=True)

embarked = pd.get_dummies(data_train['Embarked'],drop_first=True)
X = pd.concat([data_train[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']], sex, embarked], axis=1) #creating our features

y= data_train['Survived'] #chosing our target

print(X.head())

print(y.head())
data_test.head()
data_test.info()
data_test.isnull().sum()
data_test.drop(['PassengerId', 'Pclass'], axis=1).describe()
data_test['Fare'].fillna('35.62', inplace=True)

data_test['Age'] = data_test[['Age', 'Pclass']].apply(filling, axis=1)

sex = pd.get_dummies(data_test['Sex'],drop_first=True)

embarked = pd.get_dummies(data_test['Embarked'],drop_first=True)

X_validation = pd.concat([data_test[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']], sex, embarked], axis=1)

X_validation.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=46)



from sklearn.ensemble import RandomForestClassifier 

model = RandomForestClassifier(n_estimators=10)

model.fit(X_train, y_train)

model.score(X_test, y_test)
from sklearn.metrics import classification_report, confusion_matrix



y_predict = model.predict(X_test)

print(classification_report(y_test, y_predict))

print(confusion_matrix(y_test, y_predict))
predictions = model.predict(X_validation)

data_test['Survived'] = predictions

submit = data_test[['PassengerId', 'Survived']]

submit.to_csv('submission.csv', index=False)