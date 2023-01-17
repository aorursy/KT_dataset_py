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
#import training set

df_train = pd.read_csv('/kaggle/input/titanic/train.csv')

df_train.head()
#Check for null 

df_train.isnull().sum()
#Feature Cleaning

#Filling null values in AGE

df_train['Age'].fillna(df_train['Age'].mean(), inplace = True)



#Filling null values in EMBARKED

df_train['Embarked'].fillna(df_train['Embarked'].mode()[0], inplace = True)
#Feature Engineering



#Map male:1 and female:0

df_train['Sex'].replace({'male' : 1, 'female' : 0}, inplace = True)



#Group Age into bins

df_train.loc[df_train['Age'] <= 16, 'Age'] = 0

df_train.loc[(df_train['Age'] > 16) & (df_train['Age'] <= 30), 'Age'] = 1

df_train.loc[(df_train['Age'] > 30) & (df_train['Age'] <= 60), 'Age'] = 2

df_train.loc[(df_train['Age'] > 60), 'Age'] = 3



#Group Fare into bins

df_train.loc[df_train['Fare'] <= 7.91, 'Fare'] = 0

df_train.loc[(df_train['Fare'] > 7.91) & (df_train['Fare'] <= 14.45), 'Fare'] = 1

df_train.loc[(df_train['Fare'] > 14.45) & (df_train['Fare'] <= 31), 'Fare'] = 2

df_train.loc[(df_train['Fare'] > 31), 'Fare'] = 3



#Map S:0, C:1 and Q:2

df_train['Embarked'].replace({'S' : 0, 'C' : 1, 'Q' : 2}, inplace = True)



#Create a feature that stores family size

df_train['FamilySize'] = df_train['SibSp'] + df_train['Parch'] + 1



#Create a feature that indicates whether the passenger is alone or has family

df_train['IsAlone'] = 0

df_train['IsAlone'].loc[df_train['FamilySize'] <=1 ] = 1



#Drop unnecessary features

df_train.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin'], axis = 1, inplace = True)
X = df_train.drop('Survived', axis = 1)

y = df_train['Survived']
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV



#List Hyperparameters that we want to tune.

max_depth = list(range(1,15))



#Convert to dictionary

hyperparameters = dict(max_depth = max_depth)



#Create new Decision Tree object

DT = DecisionTreeClassifier()



#Use GridSearch to perform Cross Validation

clf = GridSearchCV(DT, hyperparameters, cv=10)



#Fit the model

best_model = clf.fit(X,y)



print('Best max_depth:', best_model.best_estimator_.get_params()['max_depth'])

#Import test data

df_test = pd.read_csv('/kaggle/input/titanic/test.csv')
#Feature Engineering to make data similar to training data



#Filling null values in AGE

df_test['Age'].fillna(df_test['Age'].mean(), inplace = True)



#Filling null values in EMBARKED

df_test['Fare'].fillna(df_test['Fare'].mean(), inplace = True)



df_test['Sex'].replace({'male' : 1, 'female' : 0}, inplace = True)



df_test.loc[df_test['Age'] <= 16, 'Age'] = 0

df_test.loc[(df_test['Age'] > 16) & (df_test['Age'] <= 30), 'Age'] = 1

df_test.loc[(df_test['Age'] > 30) & (df_test['Age'] <= 60), 'Age'] = 2

df_test.loc[(df_test['Age'] > 60), 'Age'] = 3



df_test.loc[df_test['Fare'] <= 7.91, 'Fare'] = 0

df_test.loc[(df_test['Fare'] > 7.91) & (df_test['Fare'] <= 14.45), 'Fare'] = 1

df_test.loc[(df_test['Fare'] > 14.45) & (df_test['Fare'] <= 31), 'Fare'] = 2

df_test.loc[(df_test['Fare'] > 31), 'Fare'] = 3



df_test['Embarked'].replace({'S' : 0, 'C' : 1, 'Q' : 2}, inplace = True)



df_test['FamilySize'] = df_test['SibSp'] + df_test['Parch'] + 1



df_test['IsAlone'] = 0

df_test['IsAlone'].loc[df_test['FamilySize'] <=1 ] = 1



df_test.drop(['Name', 'SibSp', 'Parch', 'Ticket', 'Cabin'], axis = 1, inplace = True)
df_test.head()
#Making Predictions

X_test = df_test.drop('PassengerId', axis = 1)

predictions = clf.predict(X_test)
output = pd.DataFrame({'PassengerId': df_test['PassengerId'], 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")