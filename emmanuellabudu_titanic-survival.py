import numpy as np 
import pandas as pd 

import sklearn
from sklearn.impute import SimpleImputer
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
train_data = pd.read_csv ('../input/titanic/train.csv')
test_data = pd.read_csv ('../input/titanic/test.csv')
#view the first 10 rows of training data
train_data.head (10)
#view data types
train_data.info()
#view the first 10 rows of test data
test_data.head (10)

#view data types
test_data.info()
train_data_og=train_data
test_data_og=test_data
# Identify all columns with missing data in both training and testing data
print(train_data.columns[train_data.isna().any()].tolist())
print(test_data.columns[test_data.isna().any()].tolist())
#Replace missing data in 'Age' column of training data  with median value
#train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())
train_data.Age.fillna(train_data["Age"].median(), inplace=True)
#Replace missing data in 'Cabin' column of training data with mode value since it is categorical data
#train_data['Cabin'] = train_data['Cabin'].fillna(train_data['Cabin'].mode())
train_data.Cabin.fillna(train_data["Cabin"].mode(), inplace=True)
#Replace missing data in 'Embarked' column of training data with mode value since it is categorical data
#train_data['Embarked'] = train_data['Embarked'].fillna(train_data['Embarked'].mode())
train_data.Embarked.fillna(train_data["Embarked"].mode(), inplace=True)
train_data.head(10)
#Replace missing data in 'Age' column of test data  with median value
#test_data['Age'] = test_data['Age'].fillna(test_data['Age'].median())
test_data.Age.fillna(test_data["Age"].median(), inplace=True)
#Replace missing data in 'Cabin' column of test data with mode value since it is categorical data
#test_data['Cabin'] = test_data['Cabin'].fillna(test_data['Cabin'].mode())
test_data.Cabin.fillna(test_data["Cabin"].mode(), inplace=True)
#Replace missing data in 'Fare' column of training data with median value 
#test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].median())
test_data.Fare.fillna(test_data["Fare"].median(), inplace=True)
test_data.head(10)
print(train_data.columns[train_data.isnull().any()].tolist())
print(test_data.columns[test_data.isnull().any()].tolist())
print (pd.unique(train_data['Sex']))
print (pd.unique(train_data['Cabin']))
print (pd.unique(train_data['Ticket']))
print (pd.unique(train_data['Embarked']))
#Check if there are still anymore NaN values
print(train_data.columns[train_data.isnull().any()].tolist())
print(test_data.columns[test_data.isnull().any()].tolist())
#Drop Name and PassengerId columns along
train_data = train_data.drop(['Name','PassengerId', 'Ticket', 'Cabin'], axis=1)
test_data = test_data.drop(['Name', 'PassengerId', 'Ticket','Cabin'], axis=1)
#Handling the 'Sex' column
train_data['Sex'] = train_data['Sex'].apply(lambda x: 1 if x == 'male' else 0)
test_data['Sex'] = test_data['Sex'].apply(lambda x: 1 if x == 'male' else 0)
train_data
#Handling the 'Embarked' column
train_data['Embarked'] = train_data['Embarked'].apply(lambda x: 0 if x == 'S' else 1 if x=='C' else 0)
test_data['Embarked'] = test_data['Embarked'].apply(lambda x: 0 if x == 'S' else 1 if x=='C' else 0)
train_data
#Select x and y  variables
y= train_data['Survived']
X = train_data[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
#split x and y for training and validating the model
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.30, random_state=40)

X_test = test_data[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]

#fit the ML model on input data and respective outputs
model = DecisionTreeClassifier()
model.fit (X_train, y_train)
#test the model with validation data
y_testing = model.predict(X_val)
#Evaluate the performance of the model
print(accuracy_score(y_val,y_testing))
#Make predictions using test data
y_pred = model.predict(X_test)
#Write results of predictions to csv file
predictions_file = pd.DataFrame({'PassengerId': test_data_og['PassengerId'],'Survived': y_pred})
predictions_file.head(10)
predictions_file.to_csv('titanic_predictions.csv', index=False)
print("Done with predictions!")
