import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
from statistics import mode
import re
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')
train.head()
train.isnull().sum()
sns.countplot(train['Survived']);
sns.countplot(x='Survived', hue='Sex', data=train);
sns.heatmap(train.isnull(), yticklabels = False, cmap='plasma');
train.describe()
sns.countplot(train['Pclass']);
train.Name.value_counts().head()
train['Age'].hist(bins=40);
train['SibSp'].value_counts()
sns.countplot(train['SibSp'])

plt.title('Count plot for SibSp');
sns.countplot(train['Parch'])

plt.title('Count plot for Parch');
train.Ticket.value_counts(dropna=False, sort=True).head()
train['Fare'].hist(bins=50)

plt.ylabel('Price')

plt.xlabel('Index')

plt.title('Fare Price distribution');
train.Cabin.value_counts(0)
sns.countplot(train['Embarked'])

plt.title('Count plot for Embarked');
sns.heatmap(train.corr(), annot=True);
sns.countplot(x='Survived', hue='Pclass', data=train)

plt.title('Count plot for Pclass categorized by Survived');
age_group = train.groupby('Pclass')['Age']
age_group.median()
age_group.mean()
train.loc[train.Age.isnull(), 'Age'] = train.groupby("Pclass").Age.transform('median')



train["Age"].isnull().sum()
sns.heatmap(train.isnull(), yticklabels = False, cmap='plasma');
train['Sex'][train['Sex'] == 'male'] = 0

train['Sex'][train['Sex'] == 'female'] = 1
train["Embarked"][train["Embarked"] == "S"] = 0

train["Embarked"][train["Embarked"] == "C"] = 1

train["Embarked"][train["Embarked"] == "Q"] = 2
train.head()
df = pd.read_csv('../input/titanic/train.csv')
test['Survived'] = np.nan

full = pd.concat([df, test])
full.isnull().sum()
full.head()
full['Embarked'] = full['Embarked'].fillna(mode(full['Embarked']))
# Convert 'Sex' variable to integer form!

full["Sex"][full["Sex"] == "male"] = 0

full["Sex"][full["Sex"] == "female"] = 1



# Convert 'Embarked' variable to integer form!

full["Embarked"][full["Embarked"] == "S"] = 0

full["Embarked"][full["Embarked"] == "C"] = 1

full["Embarked"][full["Embarked"] == "Q"] = 2
sns.heatmap(full.corr(), annot=True);
full['Age'] = full.groupby("Pclass")['Age'].transform(lambda x: x.fillna(x.median()))
full.isnull().sum()
full['Fare']  = full.groupby("Pclass")['Fare'].transform(lambda x: x.fillna(x.median()))
full['Cabin'] = full['Cabin'].fillna('U')
full['Cabin'].unique().tolist()[:20]
full['Cabin'] = full['Cabin'].map(lambda x:re.compile("([a-zA-Z])").search(x).group())
full['Cabin'].unique().tolist()
cabin_category = {'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7, 'T':8, 'U':9}

full['Cabin'] = full['Cabin'].map(cabin_category)
full['Cabin'].unique().tolist()
full['Name'].head()
full['Name'] = full.Name.str.extract(' ([A-Za-z]+)\.', expand = False)
full['Name'].unique().tolist()
full['Name'].value_counts(normalize = True) * 100
full.rename(columns={'Name' : 'Title'}, inplace=True)
full['Title'] = full['Title'].replace(['Rev', 'Dr', 'Col', 'Ms', 'Mlle', 'Major', 'Countess', 

                                       'Capt', 'Dona', 'Jonkheer', 'Lady', 'Sir', 'Mme', 'Don'], 'Other')
full['Title'].value_counts(normalize = True) * 100
title_category = {'Mr':1, 'Miss':2, 'Mrs':3, 'Master':4, 'Other':5}

full['Title'] = full['Title'].map(title_category)

full['Title'].unique().tolist()
full['familySize'] = full['SibSp'] + full['Parch'] + 1
# Drop redundant features

full = full.drop(['SibSp', 'Parch', 'Ticket'], axis = 1)
full.head()
# Recover test dataset

test = full[full['Survived'].isna()].drop(['Survived'], axis = 1)
test.head()
# Recover train dataset

train = full[full['Survived'].notna()]
train['Survived'] = train['Survived'].astype(np.int8)
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(train.drop(['Survived', 'PassengerId'], axis=1), train['Survived'], test_size = 0.2, random_state=2)
from sklearn.linear_model import LogisticRegression

LogisticRegression = LogisticRegression(max_iter=10000)

LogisticRegression.fit(X_train, y_train)
predictions = LogisticRegression.predict(X_test)

predictions
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, predictions)
acc = (87+54) / (87+54+13+25) * 100

acc
from sklearn.model_selection import KFold

kf = KFold(n_splits = 5, random_state=2)
from sklearn.model_selection import cross_val_score



cross_val_score(LogisticRegression, X_test, y_test, cv = kf).mean() * 100
from sklearn.ensemble import RandomForestClassifier

RandomForest = RandomForestClassifier(random_state=2)
# Set our parameter grid

param_grid = { 

    'criterion' : ['gini', 'entropy'],

    'n_estimators': [100, 300, 500],

    'max_features': ['auto', 'log2'],

    'max_depth' : [3, 5, 7]    

}
from sklearn.model_selection import GridSearchCV



randomForest_CV = GridSearchCV(estimator = RandomForest, param_grid = param_grid, cv = 5)

randomForest_CV.fit(X_train, y_train)
randomForest_CV.best_params_
randomForestFinalModel = RandomForestClassifier(random_state = 2, criterion = 'gini', max_depth = 7, max_features = 'auto', n_estimators = 300)



randomForestFinalModel.fit(X_train, y_train)
predictions = randomForestFinalModel.predict(X_test)
from sklearn.metrics import accuracy_score



accuracy_score(y_test, predictions) * 100
test['Survived'] = randomForestFinalModel.predict(test.drop(['PassengerId'], axis = 1))
test[['PassengerId', 'Survived']].to_csv('MySubmission.csv', index = False)
test.info()