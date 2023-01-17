import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.info()
test.info()
sns.heatmap(train.isnull(), yticklabels=False, cbar=False)
train[['Age','Cabin']].count()
sns.countplot(x='Survived', data=train, hue='Sex', palette='RdBu_r')
sns.countplot(x='Survived', hue='Pclass', data=train, palette='RdBu_r')
train['Age'].plot(kind='hist', bins=30)
train['SibSp'].plot(kind='hist', x='SibSp')
train['Fare'].hist(bins=40)
sns.boxplot(x='Pclass', y='Age', data=train)
train_x = train.drop(['PassengerId', 'Survived', 'Name', 'Ticket'], axis=1)
train_x.head(1)
age_means = train_x.groupby('Pclass').mean()['Age']
def impute_age(cols):

    age = cols[0]

    pclass = cols[1]

    

    if pd.isnull(age):

        return age_means[pclass]

    else:

        return age
train_x['Age'] = train_x[['Age', 'Pclass']].apply(impute_age, axis=1)
train_x['Age'].isnull().sum()
sex = pd.get_dummies(train_x['Sex'], drop_first=True)

embarked = pd.get_dummies(train_x['Embarked'], drop_first=True)
train_x.drop(['Sex', 'Embarked'], inplace=True, axis=1)

train_x = pd.concat([train_x, sex, embarked], axis=1)
train_x.drop('Cabin', axis=1, inplace=True)
train_x.head()
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(train_x, train['Survived'], test_size=0.4, random_state=42)
from sklearn.linear_model import LogisticRegression



log_model = LogisticRegression()

log_model.fit(X_train, y_train)



predictions = log_model.predict(X_test)
from sklearn.metrics import classification_report



print(classification_report(y_test, predictions))