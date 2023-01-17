# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Execute the prior cell before this one to load pandas and other modules
training_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
training_df.head(3)
# Type the training_df DataFrame name here to see what it does

training_df.columns
# Code area for Attributes Exercises

# Code area for Column Exercises

# Code area for Row Exercises

# Uncomment the following statements to create the mydata DataFrame
# mydata = training_df.set_index('Ticket')
# mydata.head()
# Code area for more Row Exercises

# Code area for more Display Methods Exercises

# code for Analysis Method Exercises

# code for Modify Method Exercises

# code for Evaluate Method Exercises

# Try the plot() examples here

# X is the traditional training set, with y the solution set.
key_features = ['Pclass','Age','Fare','Sex','SibSp','Parch']
X = training_df.filter(key_features)
y = training_df['Survived']

# Modify the training set with some addition data
X.eval('IsFemale = (Sex == "female")', inplace=True)
X.eval('FamilySize = SibSp + Parch', inplace=True)
X.drop(['Sex','SibSp','Parch'], axis=1, inplace=True)
X.info()
# Only the Age column has null values, so we can fill based on the median Age
X.fillna(X.Age.median(), inplace=True)
X.corrwith(y).sort_values(ascending=False)
# Fit a RandomForestRegressor model to the data.
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(random_state=42, n_estimators=100)
rf_model.fit(X,y)
test_X = test_df.filter(key_features)
test_X.eval('IsFemale = (Sex == "female")', inplace=True)
test_X.eval('FamilySize = SibSp + Parch', inplace=True)
test_X.drop(['Sex','SibSp','Parch'], axis=1, inplace=True)
test_X.isnull().any()
# We have two columns with missing values, so we will treat these separately.
test_X.Age.fillna(test_X.Age.median(), inplace=True)
test_X.Fare.fillna(test_X.Fare.median(), inplace=True)
test_X.isnull().any()
predict = np.around(rf_model.predict(test_X)).astype(int)
output = pd.DataFrame({'PassengerId': test_df.PassengerId,
                      'Survived': predict})

output.to_csv('submission.csv', index=False)
print('Ready to submit. Commit this notebook and go to the Output tab.')