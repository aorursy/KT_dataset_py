# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
# read train data 
train=pd.read_csv('../input/train.csv')
# set target value
train_Y=train.SalePrice
# set prediction columns
predictor_cols = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']
train_X=train[predictor_cols]
# define data model here it is RandomForestRegressor
my_model=RandomForestRegressor()
# fit 
my_model.fit(train_X,train_Y)
# read train data
test=pd.read_csv('../input/test.csv')
# set prediction columns
test_X=test[predictor_cols]
# pradict salesPrice for test data
predicted_Y=my_model.predict(test_X)
print(predicted_Y)
# creating submission ' .csv ' file with columns Id,salesPrice
my_submission=pd.DataFrame({'Id':test.Id,'SalePrice':predicted_Y})
my_submission.to_csv('my_submission.csv',index=False)

