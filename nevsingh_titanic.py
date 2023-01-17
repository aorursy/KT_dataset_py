# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_file_path = '../input/train.csv'
test_file_path = '../input/test.csv'

# create a pandas dataframe from the training and test data
train_df = pd.read_csv(train_file_path)
test_df = pd.read_csv(test_file_path)
# take a look at the training dataset
'''
print(train_df.head(12))
print(train_df.describe())
print(train_df.shape)
print(train_df.columns)

print(train_df.Fare.head(10))
print(train_df.Sex.head(10))
print(train_df.Age.head(10))
print(train_df.SibSp.head(10))
print(train_df.Parch.head(10))
print(train_df.Ticket.head(10))
print(train_df.Cabin.head(10))
print(train_df.Embarked.head(10))
'''

## dropping the rows wih missing data reduces the number of records significantly, so it might be worth seeing if an imputer will be better 
from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer()

y = train_df.Survived
#predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
#predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']
x = train_df[predictors]
'''
print(x.shape)
print(x.head(10))
print(x.columns)
print(x.dtypes)
print(y) 

'''

''' 
## get the unique values of each column
print(x.Pclass.unique()) ## 1, 2, 3
print(x.Sex.unique()) ## 'male', 'female'
print(x.Age.unique()) ## numeric 
print(x.SibSp.unique()) ## 0-8
print(x.Parch.unique()) ## 0-6
#print(x.Ticket.unique()) 
print(x.Fare.unique()) 
#print(x.Cabin.unique()) ## too many individual ones
print(x.Embarked.unique()) ## S, C, Q, nan
'''
x_one_hot_encoded = pd.get_dummies(x)
one_hot_encoded_test_predictors = pd.get_dummies(test_df[predictors])
final_train, final_test = x_one_hot_encoded.align(one_hot_encoded_test_predictors,join='inner',  axis=1)
#print(x_one_hot_encoded.shape)
x_imputed = pd.DataFrame(my_imputer.fit_transform(x_one_hot_encoded))
x_imputed.columns = x_one_hot_encoded.columns
#print(x_imputed)
print(x_imputed.dtypes)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

## build a model
titanic_model = RandomForestClassifier()
print(y.head())
## split the dataset into training and validation for testing, so we are not testing on the same dataset
train_x, val_x, train_y, val_y = train_test_split(x_imputed, y,random_state = 0)

print(train_x.shape)
print(train_y.shape)
print(train_y.unique())
titanic_model.fit(train_x, train_y)
predictions = titanic_model.predict(val_x)
print(predictions)
print(pd.DataFrame(predictions).head())
print(predictions.shape)
print(mean_absolute_error(val_y, predictions))

final_test = my_imputer.transform(final_test)
#print(type(final_test))


## build a model
#titanic_test_model = RandomForestRegressor()
#titanic_test_model.fit(x_test, y_test)
random_forest_predictions = titanic_model.predict(final_test)

#print(random_forest_predictions)
## ready to submit
my_submission = pd.DataFrame({'PassengerId': test_df.PassengerId, 'Survived': random_forest_predictions})
my_submission.to_csv('titanic_survival2.csv', index = False)
print(my_submission)
print(my_submission.describe())