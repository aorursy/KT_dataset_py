# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.ensemble import IsolationForest

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv')

test = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')

submission_sample = pd.read_csv('../input/competitive-data-science-predict-future-sales/sample_submission.csv')

train.head(), test.head(), submission_sample.head()
plt.figure(figsize=(10,6))

sns.set_style('whitegrid')

sns.boxplot(x='item_price',data=train)
train = train[train.item_price<70000]
train.shape
plt.figure(figsize=(10,6))

sns.set_style('whitegrid')

sns.boxplot(x='item_cnt_day',data=train)
train = train[train.item_cnt_day < 1000]
agg = train.groupby(['date_block_num','shop_id','item_id']).item_cnt_day.sum().reset_index(name ='item_cnt_monthly')

agg
agg2 = agg.groupby(['shop_id','item_id']).item_cnt_monthly.mean().reset_index(name='item_cnt_month')

agg2
agg2.isna().sum()
agg2.corr()
X = pd.DataFrame(agg2['item_id'])

y = agg2['item_cnt_month']
X.head()
Xtrain,Xval,ytrain,yval = train_test_split(X,y,train_size= 0.8)
Xtest = test[['item_id']]

model = XGBRegressor()

model.fit(Xtrain,ytrain)

pred = model.predict(Xval)

error = mean_squared_error(pred,yval)

print(error)
pred2 = model.predict(Xtest)
output = pd.DataFrame({'ID': test.ID,'item_cnt_month':pred2})
output.to_csv('submission.csv',index=False)