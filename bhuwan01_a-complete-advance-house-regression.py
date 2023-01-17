# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import sklearn as sns
import matplotlib.pyplot as plt
%matplotlib inline
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
pd.set_option('display.max_row',None)
pd.set_option('display.max_columns',None)

train.head()
test.head()
train.describe()
#cheching %of null values in each features
null_cols = train.isnull().sum()/train.shape[0]*100
#seperate null values more than 10 %
null_cols_rm = null_cols[null_cols>10].keys()
# droping the columns which have more than 10% of null values
train1 = train.drop(columns=null_cols_rm,axis=1)
#after removing the columns having more than 10% null values, checking the null columns
null_cols = train1.isnull().sum()/train1.shape[0]*100
null_cols = null_cols.keys()
null_cols
#seperating numerical variable with categorical variable

num_var = train1[null_cols].select_dtypes(include=['int64', 'float64']).columns
num_var
cat_var = train1[null_cols].select_dtypes(include='object').columns
cat_var
#filling the missing numerical value 
for i in range(len(num_var)):
    train1[num_var[i]] = train1[num_var[i]].fillna(train1[num_var[i]].median())
#filling the missing categorical value
for i in range(len(cat_var)):
    train1[cat_var[i]] = train1[cat_var[i]].fillna(train1[cat_var[i]].mode()[0])
## doing all above steps for testing data
#dropping the columns which have null value more than 10% 
test1 = test.drop(columns=null_cols_rm,axis=True)

# seperating the cat and numerical varibale  
num_var = test1.select_dtypes(include=['int64', 'float64']).columns
cat_var = test1.select_dtypes(include=['object']).columns

#filling numerical variable
for i in range(len(num_var)):
    test1[num_var[i]] = test1[num_var[i]].fillna(test1[num_var[i]].median())
#filling categorical varibale
for i in range(len(cat_var)):
    test1[cat_var[i]] = test1[cat_var[i]].fillna(test1[cat_var[i]].mode()[0])
#preparing data for algorithm
train_y = train1['SalePrice']
train_x = train1.drop(['SalePrice','Id'],axis=1)
test_x = test1.drop('Id',axis=1)
##importing catboost 
import catboost
#creating the model 
model=catboost.CatBoostRegressor(iterations=50, depth=3, learning_rate=0.1, loss_function='RMSE',cat_features=cat_var)
#fitting the traning data
model.fit(train_x,train_y)
#predicting the testing data
p = model.predict(test_x)
#training accuracy
model.score(train_x,train_y)
#submission file
submission = pd.DataFrame()
submission['Id'] = test1['Id']
submission['SalePrice'] = p
submission.to_csv('s.csv',index=False)
