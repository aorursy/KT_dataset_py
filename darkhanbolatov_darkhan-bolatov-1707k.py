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
import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn import linear_model



import seaborn as sns

from scipy import stats

from scipy.stats import norm, skew 
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error
sample_submission = pd.read_csv("../input/competitive-data-science-predict-future-sales/sample_submission.csv")

test = pd.read_csv("../input/competitive-data-science-predict-future-sales/test.csv")

train = pd.read_csv("../input/competitive-data-science-predict-future-sales/sales_train.csv")

items = pd.read_csv("../input/competitive-data-science-predict-future-sales/items.csv")

item_categories = pd.read_csv("../input/competitive-data-science-predict-future-sales/item_categories.csv")

shops = pd.read_csv("../input/competitive-data-science-predict-future-sales/shops.csv")
print(train.shape)

print(test.shape)

print(items.shape)

print(item_categories.shape)

print(shops.shape)
train.head()
test.head()
items.head()
item_categories.head()
shops.head()
train.item_cnt_day.plot()

plt.title("Number of products sold per day");
l = train.select_dtypes(include = ['float64', 'int64'])

l.hist(figsize=(16, 16), bins=50, xlabelsize=8, ylabelsize=8);

    
unique_dates = pd.DataFrame({'date': train['date'].drop_duplicates()})

unique_dates['date_parsed'] = pd.to_datetime(unique_dates.date, format="%d.%m.%Y")

unique_dates['day'] = unique_dates['date_parsed'].apply(lambda d: d.day)

unique_dates['month'] = unique_dates['date_parsed'].apply(lambda d: d.month)

unique_dates['year'] = unique_dates['date_parsed'].apply(lambda d: d.year)



datess = train.merge(unique_dates, on='date').sort_values('date_parsed')
data = datess.groupby(['year', 'month']).agg({'item_cnt_day': np.sum}).reset_index().pivot(index='month', columns='year', values='item_cnt_day')

data.plot(figsize=(12, 8))
target = train.item_cnt_day
train = train.drop(['item_price','item_cnt_day','date','date_block_num'],axis=1).select_dtypes(exclude=['object'])



train_X, test_X, train_y, test_y = train_test_split(train,target,test_size=0.25)
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)

my_model.fit(train_X, train_y, early_stopping_rounds=8, 

             eval_set=[(test_X, test_y)], verbose=False)
predictions = my_model.predict(test_X)



print("Mean absolute error = " + str(mean_absolute_error(predictions,test_y)))
test = test.select_dtypes(exclude=['object'])

test.fillna( method ='ffill', inplace = True)
test_X = test

test_X = test_X.drop(['ID'],axis=1).select_dtypes(exclude=['object'])
predictions = my_model.predict(test_X)
my_submission = pd.DataFrame({'Id': test.ID, 'item_cnt_month': predictions})

my_submission.to_csv('submi2.csv', index=False)