# importing required libraries



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
# reading the titanic dataset into notebook



train = pd.read_csv('../input/titanicdataset-traincsv/train.csv')
# checking the top 5 rows of the dataset



train.head()
# General information on the dataset



train.info()
# Statistical information on the dataset



train.describe()
sns.countplot(x='Survived', hue='Sex', data=train)
# Probability of survival on both female and male



train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
sns.countplot(x='Pclass', hue='Survived', data=train)
# Probability of survival on the passenger class



train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# Histogram depicting different ages with survival



g = sns.FacetGrid(data=train, col='Survived')

g.map(sns.distplot, 'Age', kde=False)
# Number of siblings or spouses



sns.countplot(x='SibSp', hue='Survived', data=train)
# Probability of survival



train[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='SibSp', ascending=True)
# Parents and children



sns.countplot(x='Parch', hue='Survived', data=train)
train[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Parch', ascending=True)
train['Fare'].describe()
sns.distplot(train['Fare'], bins=50)
g = sns.FacetGrid(data=train, col='Survived')

g.map(sns.distplot, 'Fare', kde=False, bins=10)
sns.countplot(x='Embarked',hue='Survived', data=train)
train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
sns.pairplot(train, hue='Survived', diag_kws={'bw': 0.2})
plt.figure(figsize=(15,7))

sns.boxplot(x='Pclass', y='Age', data=train)
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
train[train['Age'].isnull()]
by_sex_class = train.groupby(['Sex', 'Pclass'])
def impute_median(series):

    return series.fillna(series.median())
train['Age'] = by_sex_class['Age'].transform(impute_median)
train[train['Embarked'].isnull()]
# filling up the null values with the top most common category



train['Embarked'] = train['Embarked'].fillna(train['Embarked'].value_counts().index[0])
train['Embarked'].isnull().any()
train[train['PassengerId'] == 830]
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
train.drop('Cabin', axis=1, inplace=True)
train.head()
train.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
train.head()
Sex = pd.get_dummies(train['Sex'], drop_first=True)
Embarked = pd.get_dummies(train['Embarked'], drop_first=True)
Pclass = pd.get_dummies(train['Pclass'], drop_first=True)
# As we have our dummy variables, we will drop the existing columns and replace them with our dummy variables.



train.drop(['Sex', 'Embarked', 'Pclass'], axis=1, inplace=True)
# concatenating the dummy variables to the dataset



train = pd.concat([train, Sex, Embarked, Pclass], axis=1)
train.head()
X = train.drop(['Survived'], axis=1)

y = train['Survived']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
# we will use a simple logistic regression model



from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(accuracy_score(y_test, predictions) * 100)