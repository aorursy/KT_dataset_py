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
test = pd.read_csv("../input/hr-prime/Test.csv")

submission = pd.read_csv("../input/hr-prime/sample_submission.csv")

train = pd.read_csv("../input/hr-prime/Train.csv")
train['Range'] = train['High_Cap_Price'] - train['Low_Cap_Price']

train['Date'] = pd.to_datetime(train['Date'])

train['Year'] = train['Date'].dt.year

train['Month'] = train['Date'].dt.month

train['Day'] = train['Date'].dt.day

train
train.dtypes
train.isnull().sum()
test['Date'] = pd.to_datetime(test['Date'])

test['Year'] = test['Date'].dt.year

test['Month'] = test['Date'].dt.month

test['Day'] = test['Date'].dt.day

test
test.dtypes
test.isnull().sum()
train.shape
test.columns
train.columns
print('Test Dataset:',len(train))

print('Test < 1000:',len(train[train['High_Cap_Price'] < 1000]))

print('Test < 20000:',len(train[train['High_Cap_Price'] > 20000]))

train.boxplot(column='High_Cap_Price')
print('Test Dataset:',len(test))

print('Test < 1000:',len(test[test['High_Cap_Price'] < 1000]))

print('Test < 20000:',len(test[test['High_Cap_Price'] > 30000]))

test.boxplot(column='High_Cap_Price')
test_2 = test[(test['High_Cap_Price'] < 20000)&(test['High_Cap_Price'] > 1000)]

train_2 = train[(train['High_Cap_Price'] < 20000)&(train['High_Cap_Price'] > 1000)]

train_2
test_2
min(train['High_Cap_Price']), max(train['High_Cap_Price'])
min(train_2['High_Cap_Price']), max(train_2['High_Cap_Price'])
min(test['High_Cap_Price']), max(test['High_Cap_Price'])
min(test_2['High_Cap_Price']), max(test_2['High_Cap_Price'])
x_train = train_2.iloc[:len(train_2)*9//10][['State_of_Country', 'Market_Category','Product_Category', 'Grade', 'Demand','Year', 'Month', 'Day']]

x_val = train_2.iloc[len(train_2)*9//10:][['State_of_Country', 'Market_Category','Product_Category', 'Grade', 'Demand','Year', 'Month', 'Day']]



y_train = train_2.iloc[:len(train_2)*9//10]['Range']

y_val = train_2.iloc[len(train_2)*9//10:]['Range']



import time

from xgboost import XGBRegressor

ts = time.time()



model = XGBRegressor(

    max_depth=10,

    n_estimators=1000,

    min_child_weight=0.5, 

    colsample_bytree=0.8, 

    subsample=0.8, 

    eta=0.1,

#     tree_method='gpu_hist',

    seed=42)



model.fit(

    x_train, 

    y_train, 

    eval_metric="rmse", 

    eval_set=[(x_train, y_train), (x_val, y_val)], 

    verbose=True, 

    early_stopping_rounds = 20)



time.time() - ts
x_test = test[['State_of_Country', 'Market_Category','Product_Category', 'Grade', 'Demand','Year', 'Month', 'Day']]

Y_pred = model.predict(x_val) #.clip(0, 20)

Y_test = model.predict(x_test) #.clip(0, 20)



results = abs(test['High_Cap_Price'] - Y_test)

results
submission['Low_Cap_Price'] = round(results).astype(int)

submission.to_csv('submission_xgb.csv',index=False)

submission
min(submission['Low_Cap_Price']), max(submission['Low_Cap_Price'])
from IPython.display import display, Image

display(Image(filename='../input/results/HackerEarth Machine Learning challenge- Slashing prices for the biggest sale day.PNG'))