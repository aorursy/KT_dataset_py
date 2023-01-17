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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
train_data = pd.read_csv('../input/train.csv')
train_data
#Converting Sex and Embarked into numeric datatype

train_data[['Sex','Embarked']] = train_data[['Sex','Embarked']].stack().rank(method='dense').unstack()
train_data
# Declaring Features and the Target DataFrames

features = [ 'Sex', 'Age', 'SibSp','Parch','Fare','Embarked']

train_data_features = train_data.loc[:,features]

train_data_target = train_data.loc[:,['Survived']]
### TRAINING DATA CLEANSING  ###





#Finding missing data pre-cleansing

indices = np.where(train_data_features.isna())

print("Following are the missing data indices before cleansing")

print(indices)





#Data cleansing

train_data_features['Age'] = train_data_features['Age'].fillna(train_data_features['Age'].mean())

train_data_features['Embarked'] = train_data_features['Embarked'].fillna(0)



#Finding missing data post-cleansing

print("Following are the missing data indices after cleansing")

indices = np.where(train_data_features.isna())

print(indices)



# import logistic regression model

from sklearn.linear_model import LogisticRegression

logistic_model = LogisticRegression()
#Train the model

logistic_model.fit(train_data_features,train_data_target)
#Read the test data

train_data = pd.read_csv('../input/test.csv')
#Select features from the test data

test_data_features = test_data.loc[:,features]
#Converting Sex and Embarked into numeric datatype

test_data_features[['Sex','Embarked']] = test_data_features[['Sex','Embarked']].stack().rank(method='dense').unstack()
### TEST DATA CLEANSING  ###



#Finding missing data pre-cleansing

indices = np.where(test_data_features.isna())

print("Following are the missing data indices before cleansing")

print(indices)





#Data cleansing

# CAVEAT -: USING TRAINING MEAN TO FILL MISSING TRAINING DATA AS THE TRAINING SIZE IS BIGGER 





test_data_features['Age'] = test_data_features['Age'].fillna(train_data_features['Age'].mean())

test_data_features['Fare'] = test_data_features['Fare'].fillna(train_data_features['Fare'].mean())



#Finding missing data post-cleansing

print("Following are the missing data indices after cleansing")

indices = np.where(test_data_features.isna())

print(indices)

# Below are the predicted values

predicted_survived = logistic_model.predict(test_data_features)

print(predicted_survived)
#Stacking up the passengerid and survived status together as per the competition's prescribed format

final_result = np.column_stack( (np.array(test_data['PassengerId']),np.array(predicted_survived)))

final_result = pd.DataFrame(final_result)

final_result.columns = ['PassengerId', 'Survived']

final_result.set_index('PassengerId')
#Create dataframe to output csv file : Not allowed on kaggle

#final_result.to_csv('../input/result.csv')