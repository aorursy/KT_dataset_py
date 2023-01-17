# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

train = pd.read_csv('/kaggle/input/titanic/train.csv')

train.head()
sns.heatmap(train.isnull(), cbar=False)
sns.countplot(x='Survived', data=train)
sns.countplot(x='Survived',hue='Sex', data=train)
sns.countplot(x='Survived',hue='Pclass', data=train)
train['Age'].plot.hist(bins=35)
train['Fare'].hist(bins=40, figsize=(10,4))
sns.boxplot(x='Pclass',y='Age', data=train)
train.groupby('Pclass').mean()['Age']
def fill_age_if_null(cols):

    age = cols[0]

    pclass = cols[1]

    

    if(pd.isnull(age)):

        if(pclass == '1'):

            return 38

        elif(pclass == '2'):

            return 30

        else:

            return 25

    else:

        return age
train['Age'] = train[['Age', 'Pclass']].apply(fill_age_if_null, axis=1)

train['Fare'].fillna((train['Fare'].mean()), inplace=True)
sns.heatmap(train.isnull(), cbar=False)
train.drop('Cabin', axis=1, inplace=True)
train.head()
sex = pd.get_dummies(train['Sex'], drop_first=True)

sex.head(1)
embark = pd.get_dummies(train['Embarked'], drop_first=True)

embark.head()

train = pd.concat([train, sex, embark], axis=1)

train.head()
train.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)

train.head()
X = train.drop('Survived', axis=1)

y = train['Survived']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn.linear_model import LogisticRegression
logistic_regression_model = LogisticRegression()
logistic_regression_model.fit(X_train, y_train)
predictions  = logistic_regression_model.predict(X_test)

predictions
from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, predictions)
logistic_regression_model.coef_[0]
X.columns
coefficients = pd.DataFrame(logistic_regression_model.coef_[0], X.columns, columns=['Coefficient'])
coefficients
test = pd.read_csv('/kaggle/input/titanic/test.csv')
test['Age'] = test[['Age', 'Pclass']].apply(fill_age_if_null, axis=1)

sex = pd.get_dummies(test['Sex'], drop_first=True)

embark = pd.get_dummies(test['Embarked'], drop_first=True)

test = pd.concat([test, sex, embark], axis=1)

test.drop(['Sex', 'Embarked', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

test['Fare'].fillna((test['Fare'].mean()), inplace=True)
y = train["Survived"]

features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']

features
X = pd.get_dummies(train[features])

X_test = pd.get_dummies(test[features])

#X_test.dropna(inplace=True)
logistic_regression_model_test = LogisticRegression()
logistic_regression_model_test.fit(X, y)
predictions = logistic_regression_model_test.predict(X_test)
output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")