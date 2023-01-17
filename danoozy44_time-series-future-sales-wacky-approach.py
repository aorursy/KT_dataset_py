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
train = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')

test = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')

shops = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/shops.csv')

items = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')

item_cats = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv')
train
#Dropping null values



train.dropna(inplace=True)
train.dtypes
#Here, df['column'].apply() performs an operation on that column

#Look up datetime.strptime on how it works.

#lambda is used for defining functions with only one statement.



from datetime import datetime



train['date'] = train['date'].apply(lambda x:datetime.strptime(x, '%d.%m.%Y'))
train.dtypes
monthly_sales = train.groupby(['date_block_num','shop_id','item_id'])['date', 'item_price', 'item_cnt_day'].agg({'date':['min','max'],'item_price':'mean', 'item_cnt_day':'sum'})
#Now we have the monthly estimates



aggregation = monthly_sales.reset_index()

aggregation
#Calculate duration

diff = aggregation['date']['max'] - aggregation['date']['min']



#Convert to integers

diff = diff.dt.days.astype('int16')

diff = diff + 1
aggregation['duration'] = diff

aggregation = aggregation.drop(columns=['date'])
#Convert 2D columns to 1D



aggregation['item_cnt_month'] = aggregation['item_cnt_day']['sum']

aggregation['item_cost'] = aggregation['item_price']['mean']

aggregation = aggregation.drop(columns = ['item_cnt_day','item_price'])
#Output first row

#Understand what it means



print(aggregation.iloc[0])



print('\nThis means',aggregation['item_cnt_month'].iloc[0], 'instances of item',aggregation['item_id'].iloc[0] ,'were sold over', aggregation['duration'].iloc[0],'days, for shop',aggregation['shop_id'].iloc[0],'in month',aggregation['date_block_num'].iloc[0],',thereby netting an average of',aggregation['item_cost'].iloc[0],'units of money.')
#We need to get the sum of all durations and items sold, for each tuple of item_id and shop_id.



monthly_sales_new = aggregation.groupby(['item_id','shop_id']).agg({'item_cnt_month':['sum'], 'duration':['sum'], 'item_cost':['mean']}).reset_index()
#We've gotten our main features. Let's transform them.



monthly_sales_new
monthly_sales_new['item_cnt_month_sum'] = monthly_sales_new['item_cnt_month']['sum']/30

monthly_sales_new['duration_sum'] = monthly_sales_new['duration']['sum']

monthly_sales_new['item_cost_mean'] = monthly_sales_new['item_cost']['mean']
monthly_sales_new = monthly_sales_new.drop(columns = ['item_cnt_month','duration','item_cost'])
#Voila!



monthly_sales_new
#The wacky approach: Using item_id and shop_id (!!!) to predict item_cost

#Then using that to predict duration

#Then using the whole thing to predict item_cnt_month



X = monthly_sales_new[['item_id','shop_id']]

y = monthly_sales_new[['item_cost_mean']]



#If we don't include this following code, there's an error. I don't really know why, some feature name mismatch.

#Apparently the monthly_sales_new has a space after item_id and shop_id.

#OR XGB adds a space after the feature names? I don't really know.



test.fillna(0, inplace = True)

test.columns = ['ID', 'shop_id ', 'item_id ']

test = test.reindex(columns = ['ID', 'item_id ', 'shop_id '])

test
from xgboost import XGBRegressor



xgb = XGBRegressor(learning_rate=0.01, n_estimators = 1000, max_depth=3, subsample=0.8, colsample_bytree=1, gamma=1)
xgb.fit(X, y)
pred = xgb.predict(test.drop(columns = ['ID']))
test['item_cost_mean '] = pred
X = monthly_sales_new[['item_id', 'shop_id', 'item_cost_mean']]

y = monthly_sales_new[['duration_sum']]



xgb.fit(X, y)

pred = xgb.predict(test.drop(columns = ['ID']))

test['duration_sum '] = pred
X = monthly_sales_new[['item_id','shop_id', 'item_cost_mean', 'duration_sum']]

y = monthly_sales_new[['item_cnt_month_sum']]



xgb.fit(X, y)

pred = xgb.predict(test.drop(columns = ['ID']))
pred
result = pd.DataFrame({'ID':test['ID'], 'item_cnt_month':pred})
result.to_csv('submission.csv', index=False)