# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.impute import SimpleImputer # Simple Imputer to fill in missing data
from sklearn.ensemble import RandomForestRegressor # Random forest model

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# The first step is reading in the provided training and the testing datasets
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
# print first few lines of data
train.head()
# identify the classes with null values and get the quantity of cells lacking information
missing_counts = train.isnull().sum()
print(missing_counts[missing_counts>0])
# drop the identified columns from training and testing data
reduced_train = train.drop(['Cabin','Embarked'], axis=1)
reduced_test = test.drop(['Cabin','Embarked'], axis=1)
# print last few lines from reduced training dataset
reduced_train.tail()
# Imputing the Age feature
# compute the mean ages - force them to be integers
avg_train_age = int(reduced_train.Age.mean())
avg_test_age = int(reduced_test.Age.mean())
# perform the imputation
train_imputer = SimpleImputer(strategy='constant',fill_value=avg_train_age)
imputed_train = train_imputer.fit_transform(reduced_train)
test_imputer = SimpleImputer(strategy='constant',fill_value=avg_test_age)
imputed_test = test_imputer.fit_transform(reduced_test)

# convert back to dataframes
train_new = pd.DataFrame(imputed_train,index=imputed_train[:,0],columns=reduced_train.columns)
test_new = pd.DataFrame(imputed_test,index=imputed_test[:,0],columns=reduced_test.columns)

# print last few lines from new imputed training dataset
train_new.tail()
# Dropping the Name feature and the Ticket feature
train_small = train_new.drop(['Name','Ticket'], axis=1)
test_small = test_new.drop(['Name','Ticket'],axis=1)

# printing last few lines
train_small.tail()
# numerically encoding the Sex feature
train_small['Sex'] = train_small['Sex'].map( {'female' : 1, 'male' : 0} ).astype(int)
test_small['Sex'] = test_small['Sex'].map( {'female' : 1, 'male' : 0} ).astype(int)

# printing last few lines
train_small.tail()
# break out data into x and y components
x_features = ['PassengerId','Pclass','Sex','Age','SibSp','Parch','Fare']
train_x = train_small[x_features]
test_x = test_small[x_features]

train_y = train_small['Survived']

# implementing the random forest
forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_x, train_y)

test_pred = forest_model.predict(test_x)
# round and make predictions an array of integers
test_pred = np.round(test_pred).astype(int)

# create submission dataframe then convert to csv
submission = pd.DataFrame({
        "PassengerId" : test_x["PassengerId"],
        "Survived": test_pred
    })
submission.to_csv('submission.csv', index=False)