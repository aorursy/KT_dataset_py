#This is a simple Getting Started to Submissions notebook. It only uses numerical features and build s linear regression model
#makes predictions on the test data so the user can make a submission
#The user can then do processing of categorical features and otehr preprocessing and try a variety of different algorithms
#Author: Archana Anandakrishnan (10-26-2018)

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
## Read the training data and print a sample few lines
train = pd.read_csv('../input/train.csv')
print("The number of records and features in the training data is {}".format(train.shape))
train.head()
## Read the test data and print a sample few lines
test = pd.read_csv("../input/test.csv")
test.head()
##See what the sample submission should look like
subm = pd.read_csv('../input/sample_submission.csv')
subm.head()
## From the sample submission, we see that we need to predict a numerical value 
# for every id in the test data.
## We will start by building a simpla model. There are 81 features in the data and
# many of them are categorical. These will need some preprocessing before they can
# be used in a model. So for now we will just drop them and start with the numerical
# features

## We can find the data types of each of the columns as follows:
# print(train.dtypes)

keepcols = [] #empty list
## keep only int/float types
for i in train.columns:
    if (train[i].dtype != 'object'):
        keepcols.append(i)
    
print("There are {} numerical columns".format(len(keepcols)))        
print(keepcols)
#Lets start by feeding all these variables into a Linear Regression Model. We can use
#linear regression implementation from scikit learn 
from sklearn import linear_model

# Lets get the data ready
train_X = train[keepcols]
train_X.fillna(0, inplace=True)
train_y = np.log1p(train.SalePrice)
#train_y.fillna(1, inplace=True)
train_X = train_X.drop(['Id', 'SalePrice'], axis=1)

#Lets start with a linear regression model. The model can be easily change to a 
#random forest or gbm
regr = linear_model.LinearRegression()
regr.fit(train_X, train_y)
#Lets check how the model is doing on the training data
#The metric for this challenge as given on the evaluation page is:
#Root-Mean-Squared-Error (RMSE) between the logarithm of the predicted value 
#and the logarithm of the observed sales price.
from sklearn.metrics import mean_squared_error
from math import sqrt

#Predict andtake log
pred = regr.predict(train_X)

print("The RMSE on the training data is {}".format(sqrt(mean_squared_error(pred, train_y))))
#Do the same process and make predictions for the test data

keepcols.remove('SalePrice')

test_X = test[keepcols]
test_X.fillna(0, inplace=True)
#Make predictions with model and write submission.csv
test_X['SalePrice'] = np.expm1(regr.predict(test_X.drop(['Id'], axis=1)))

test_X[['Id', 'SalePrice']].to_csv("submission.csv", index=False)
