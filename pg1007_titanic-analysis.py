# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
train=pd.read_csv('/kaggle/input/titanic/train.csv')
test=pd.read_csv('/kaggle/input/titanic/test.csv')
train.describe(include='all')




train
import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(train['Survived'])
train.isnull().sum().sort_values()
train.columns
sns.heatmap(train.isnull(), cmap = 'plasma')





sns.countplot(x = 'Survived', hue = 'Sex', data = train)


sns.heatmap(train.corr(), annot = True)

sns.countplot(x = 'Survived', hue = 'Pclass', data = train)
group=train.groupby('Pclass')['Age']
print(group.median())
train.loc[train.Age.isnull(), 'Age'] = train.groupby("Pclass").Age.transform('median')

sns.heatmap(train.isnull(), cmap = 'plasma')


sns.heatmap(train.corr(), annot = True)
# We will get rid of 'Cabin' for now, TOO many missing values!
train.drop('Cabin', axis = 1, inplace = True)
train.isnull().sum().sort_values()
from statistics import mode
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
train["Embarked"] = train["Embarked"].fillna(mode(train["Embarked"]))

train.drop(['Name','Ticket'], axis = 1,inplace=True)
labelencoder_X = LabelEncoder()
train['Embarked'] = labelencoder_X.fit_transform(train['Embarked'])
train["Embarked"] = pd.get_dummies(train["Embarked"])
train['Embarked']
train.info()
train
train['Sex'] = labelencoder_X.fit_transform(train['Sex'])
train["Sex"] = pd.get_dummies(train["Sex"])
train['Sex']

X_train, X_test, y_train, y_test = train_test_split(train.drop(['Survived'], axis = 1), 
                                                    train['Survived'], test_size = 0.25, 
                                                    random_state = 0)
from sklearn.linear_model import LogisticRegression
regressor= LogisticRegression()
regressor.fit(X_train,y_train)
y_pred=regressor.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


train=pd.read_csv('/kaggle/input/titanic/train.csv')
test=pd.read_csv('/kaggle/input/titanic/test.csv')
test['Survived'] = np.nan

full = pd.concat([train, test])
full.isnull().sum().sort_values()

full.info()
full["Embarked"] = full["Embarked"].fillna(mode(full["Embarked"]))

full['Embarked'] = labelencoder_X.fit_transform(full['Embarked'])
full["Embarked"] = pd.get_dummies(full["Embarked"])
full['Embarked']
full['Sex'] = labelencoder_X.fit_transform(full['Sex'])
full["Sex"] = pd.get_dummies(full["Sex"])
full['Sex']

sns.heatmap(full.corr(),annot=True)
full['Age'] = full.groupby("Pclass")['Age'].transform(lambda x: x.fillna(x.median()))
full['Fare']=full.groupby("Pclass")['Fare'].transform(lambda x: x.fillna(x.median()))
full['Cabin'] = full['Cabin'].fillna('U')
full.isnull().sum().sort_values()
# no missing values
full['Cabin'].unique().tolist()
# Let's import our regular expression matching operations module!
import re

# Extract (first) letter!
full['Cabin'] = full['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
full['Cabin'].unique().tolist()
cabin_category = {'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7, 'T':8, 'U':9}
full['Cabin'] = full['Cabin'].map(cabin_category)
full['Cabin'].unique().tolist()
# Extract the salutation!
full['Title'] = full.Name.str.extract(' ([A-Za-z]+)\.', expand = False)
full['Title'].unique().tolist()
# Look at salutations percentages
full['Title'].value_counts(normalize = True) * 100
#  Bundle rare salutations: 'Other' category
full['Title'] = full['Title'].replace(['Rev', 'Dr', 'Col', 'Ms', 'Mlle', 'Major', 'Countess', 
                                       'Capt', 'Dona', 'Jonkheer', 'Lady', 'Sir', 'Mme', 'Don'], 'Other')
title_category = {'Mr':1, 'Miss':2, 'Mrs':3, 'Master':4, 'Other':5}
full['Title'] = full['Title'].map(title_category)
full['Title'].unique().tolist()
print(full.columns.tolist())
full['familySize'] = full['SibSp'] + full['Parch'] + 1
print(full.head())

# Drop redundant features
full = full.drop(['Name', 'SibSp', 'Parch', 'Ticket'], axis = 1)
full.head()
# Recover test dataset
test = full[full['Survived'].isna()].drop(['Survived'], axis = 1)
test
full

# Recover train dataset
train = full[full['Survived'].notna()]
train
# Cast 'Survived' back to integer
train['Survived'] = train['Survived'].astype(np.int8)
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(train.drop(['Survived', 'PassengerId'], axis = 1), 
                                                    train['Survived'], test_size = 0.25, 
                                                    random_state = 0)
# We'll use a logistic regression model again, but we'll go to something more fancy soon! 
from sklearn.linear_model import LogisticRegression
logisticRegression = LogisticRegression()
logisticRegression.fit(X_train, y_train)
predictions = logisticRegression.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix

# Print the resulting confusion matrix
print(confusion_matrix(y_test, predictions))
classification_report(y_test,predictions)
from sklearn.model_selection import KFold

# Set our robust cross-validation scheme!
kf = KFold(n_splits = 10, random_state = 0)
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
logisticRegression=LogisticRegression()
# Print our CV accuracy estimate:
cross_val_score(logisticRegression, X_test, y_test, cv = kf).mean()
from sklearn.ensemble import RandomForestClassifier

#Initialize randomForest
randomForest = RandomForestClassifier(random_state = 0)
# Set our parameter grid
param_grid = { 
    'criterion' : ['gini', 'entropy'],
    'n_estimators': [100, 300, 500],
    'max_features': ['auto', 'log2'],
    'max_depth' : [3, 5, 7]    
}
# Grid search for best grid
from sklearn.model_selection import GridSearchCV


randomForest_CV = GridSearchCV(estimator = randomForest, param_grid = param_grid, cv = 5)
randomForest_CV.fit(X_train, y_train)
# Print best hyperparameters
randomForest_CV.best_params_
# Define our optimal randomForest algo
randomForestFinalModel = RandomForestClassifier(random_state = 0, 
criterion = 'gini', max_depth = 5, max_features = 'auto', n_estimators = 100)
# Fit the model to the training set
randomForestFinalModel.fit(X_train, y_train)
pred = randomForestFinalModel.predict(X_test)
from sklearn.metrics import accuracy_score

# Calculate the accuracy for our powerful random forest!
print("accuracy is: ", round(accuracy_score(y_test, pred), 2))
# Predict!
test['Survived'] = randomForestFinalModel.predict(test)
# Write test predictions for final submission
test[['PassengerId', 'Survived']].to_csv('rf_submission.csv', index = False)