# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import linear_model

import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew 


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

sample_submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")
test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

print(train.shape)
print(test.shape)
train.head()
test.head()
test.info()
train.info()
#We are trying to predict the sale price column
target = train.SalePrice

#Get rid of the answer and anything thats not an object
train = train.drop(['SalePrice'],axis=1).select_dtypes(exclude=['object'])

#Split the data into test and validation
train_X, test_X, train_y, test_y = train_test_split(train,target,test_size=0.25)


train_X.fillna( method ='ffill', inplace = True)
test_X.fillna( method ='ffill', inplace = True)

#Simplist XGBRegressor
#my_model = XGBRegressor()
#my_model.fit(train_X, train_y)

my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
my_model.fit(train_X, train_y, early_stopping_rounds=8, 
             eval_set=[(test_X, test_y)], verbose=False)


#Make predictions
predictions = my_model.predict(test_X)

print("Mean absolute error = " + str(mean_absolute_error(predictions,test_y)))


#Getting it to the right format that we used with our model
test = test.select_dtypes(exclude=['object'])
test.fillna( method ='ffill', inplace = True)
#Fill in all the NaN values with ints
test_X = test

#Make predictions
predictions = my_model.predict(test_X)

my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predictions})
# you could use any filename. We choose submission here
my_submission.to_csv('house.csv', index=False)