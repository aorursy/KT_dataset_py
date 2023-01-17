# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.neighbors import KNeighborsRegressor



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Read both train and test datasets as a DataFrame

train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')
train.head()
# Generate information and statistics about the data

#train.info()

#train.describe()

#test.info()

#test.describe()
import matplotlib.pyplot as plt

import seaborn as sns

sns.countplot(x='Survived', hue='Sex',data= train)
sns.countplot(x='Survived', hue='Pclass',data= train)
# Change all missing Age values to the average Age

train.loc[train['Age'].isnull(), 'Age'] = train['Age'].mean()

test.loc[test['Age'].isnull(), 'Age'] = test['Age'].mean()

#print(train['Age'])
# Add a new column to define if an adult

train["Adult"] = 0

train["Adult"][train["Age"] >= 18] = 1

test["Adult"] = 0

test["Adult"][test["Age"] >= 18] = 1
sns.countplot(x='Survived', hue='Adult',data= train)
# Drop the Cabin column because of the missing data

train.drop('Cabin',axis=1,inplace=True)

test.drop('Cabin',axis=1,inplace=True)
# Drop the two rows with missing values in Embarked

train.dropna(inplace=True)
# Change missing Fare value to the average Fare

test.loc[test['Fare'].isnull(), 'Fare'] = test['Fare'].mean()
# Check both train and test datasets are clean

#train.isnull().sum()

#test.isnull().sum()
# Combine  number of siblings / spouses aboard the Titanic and number of parents / children aboard the Titanic as family

train['Family'] = train['SibSp'] + train['Parch']

test['Family'] = test['SibSp'] + test['Parch']
# Create dummy values for categorical data Embarked and Sex

embark_train = pd.get_dummies(train['Embarked'],drop_first=True)

embark_test = pd.get_dummies(test['Embarked'],drop_first=True)

sex_train = pd.get_dummies(train['Sex'],drop_first=True)

sex_test = pd.get_dummies(test['Sex'],drop_first=True)
test_passengers = test.PassengerId
# Prepare the train and test data by dropping unwanted column.

y_train = train.Survived

train.drop(['Sex','Embarked','Name','Ticket','PassengerId','Survived'],axis=1,inplace=True)

test.drop(['Sex','Embarked','Name','Ticket','PassengerId'],axis=1,inplace=True)

train = pd.concat([train,sex_train,embark_train],axis=1)

test = pd.concat([test,sex_test,embark_test],axis=1)
train.head()
# KNN classifier

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import mean_squared_error
# See mse function at the bottom. We pick the columns we want to train for.  

feature_remove = ['SibSp','Parch','Q','S','Age','Family','Adult']

X_train = train.drop(feature_remove, axis=1)

X_test = test.drop(feature_remove, axis=1)
X_train.head()
# Scale the data

scaler = StandardScaler().fit(X_train)

rescaled_X_train = scaler.transform(X_train)

rescaled_X_test = scaler.transform(X_test)

#print(scaler)

#print(rescaled_X_test)
# Train all the train data

model = KNeighborsClassifier(n_neighbors=5, algorithm='brute')

model.fit(rescaled_X_train, y_train)
# Make predictions using the KNN model

predictions = model.predict(rescaled_X_test)
# Count how many predicted to survive

np.sum(predictions)
# Output the predictions into a csv

predictions = pd.DataFrame({'Survived': predictions})

test_passengers = pd.DataFrame(test_passengers)

output = pd.concat([test_passengers,predictions],axis=1)

#output = pd.DataFrame({'PassengerId': test_passengers, 'Survived': predictions})

output.to_csv('gender_submission.csv', index=False)
output
# Calculate error when using one column. We test for 100 random halves of the train dataset and test on the other half. 

# We output a dictionary that contains the average mean squared error for each column.

def calculate_mses(feature_list, n):

    feature_dict = {}

    for item in feature_list:

        X_train = pd.DataFrame(train[item])

        scaler = StandardScaler().fit(X_train)

        rescaled_X_train = scaler.transform(X_train)

        mses = list()

        for i in range(0,n):

            X_train_2, X_valid, y_train_2, y_valid = train_test_split(rescaled_X_train, y_train, test_size=0.5, random_state=i)

            # Train the data for the validation

            model_valid = KNeighborsClassifier(n_neighbors=5, algorithm='brute')

            model_valid.fit(X_train_2, y_train_2)

            # Make predictions for validation using the KNN model

            predictions_valid = model_valid.predict(X_valid)

            mse = mean_squared_error(y_valid, predictions_valid)

            mses.append(mse)

        feature_dict[item] = np.mean(mses)

    return(feature_dict)
feature_list = ['Pclass','Age','SibSp','Parch','Fare','Adult','Family','male','Q','S']

calculate_mses(feature_list, n=100)
# When we set n=100; Male, Fare and Pclass return the lowest mean squared error so we use these column in the KNN model.