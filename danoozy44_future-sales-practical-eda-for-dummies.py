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
#Library importer



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from datetime import datetime as dt

import gc

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from xgboost import XGBRegressor
train = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')

items = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')

item_cats = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv')

test = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')

shops = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/shops.csv')
train['date'] = train['date'].apply(lambda x: dt.strptime(x, '%d.%m.%Y'))
def downcast(df):

    cols = df.dtypes.index.tolist()

    types = df.dtypes.values.tolist()

    for i,t in enumerate(types):

        if 'int' in str(t):

            if df[cols[i]].min() > np.iinfo(np.int8).min and df[cols[i]].max() < np.iinfo(np.int8).max:

                df[cols[i]] = df[cols[i]].astype(np.int8)

            elif df[cols[i]].min() > np.iinfo(np.int16).min and df[cols[i]].max() < np.iinfo(np.int16).max:

                df[cols[i]] = df[cols[i]].astype(np.int16)

            elif df[cols[i]].min() > np.iinfo(np.int32).min and df[cols[i]].max() < np.iinfo(np.int32).max:

                df[cols[i]] = df[cols[i]].astype(np.int32)

            else:

                df[cols[i]] = df[cols[i]].astype(np.int64)

        elif 'float' in str(t):

            if df[cols[i]].min() > np.finfo(np.float16).min and df[cols[i]].max() < np.finfo(np.float16).max:

                df[cols[i]] = df[cols[i]].astype(np.float16)

            elif df[cols[i]].min() > np.finfo(np.float32).min and df[cols[i]].max() < np.finfo(np.float32).max:

                df[cols[i]] = df[cols[i]].astype(np.float32)

            else:

                df[cols[i]] = df[cols[i]].astype(np.float64)

        elif t == np.object:

            if cols[i] == 'date':

                df[cols[i]] = pd.to_datetime(df[cols[i]], format='%Y-%m-%d')

            else:

                df[cols[i]] = df[cols[i]].astype('category')

    return df
train = downcast(train)

test = downcast(test)

shops = downcast(shops)

items = downcast(items)

item_cats = downcast(item_cats)
sns.boxplot(x=train.item_cnt_day)
sns.boxplot(x=train.item_price)
train = train[train.item_price > 0].reset_index(drop=True)

train[train.item_cnt_day <= 0].item_cnt_day.unique()

train.loc[train.item_cnt_day < 1, 'item_cnt_day'] = 0
# Якутск Орджоникидзе, 56

train.loc[train.shop_id == 0, 'shop_id'] = 57

test.loc[test.shop_id == 0, 'shop_id'] = 57

# Якутск ТЦ "Центральный"

train.loc[train.shop_id == 1, 'shop_id'] = 58

test.loc[test.shop_id == 1, 'shop_id'] = 58

# Жуковский ул. Чкалова 39м²

train.loc[train.shop_id == 11, 'shop_id'] = 10

test.loc[test.shop_id == 11, 'shop_id'] = 10



train.loc[train.shop_id == 40, 'shop_id'] = 39

test.loc[test.shop_id == 40, 'shop_id'] = 39
new_test = pd.merge(pd.merge(pd.merge(test, items),shops),item_cats)

new_test
new_train = pd.merge(pd.merge(pd.merge(train, items),shops),item_cats)

new_train
aggr = new_train.groupby(['item_category_id']).agg({'item_price':'sum'})

aggr = aggr.reset_index()



fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

ax.plot(aggr['item_category_id'], aggr['item_price'])
aggr = new_train.groupby(['item_category_id']).agg({'item_cnt_day':'mean'})

aggr = aggr.reset_index()



fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

ax.plot(aggr['item_category_id'], aggr['item_cnt_day'], color='red')
aggr = train.groupby(['date_block_num']).agg({'item_cnt_day':'mean'})

aggr = aggr.reset_index()



fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

ax.plot(aggr['date_block_num'], aggr['item_cnt_day'], color='brown')
aggr = train.groupby(['date_block_num']).agg({'item_price':'mean'})

aggr = aggr.reset_index()



fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

ax.plot(aggr['date_block_num'], aggr['item_price'], color='green')
aggr = train.groupby(['date_block_num']).agg({'item_price':'mean'})

aggr = aggr.reset_index()



sns.scatterplot(data=aggr, x='date_block_num', y='item_price')
sns.set(style='ticks')



aggr = train.drop(columns = ['date','shop_id','item_id'])

sns.pairplot(aggr)
#Delete unwanted dataframes



del new_train

del test



gc.collect()
aggr = train.groupby(['shop_id','item_id','date_block_num'])['item_price','item_cnt_day','date'].agg({'item_price':['mean'], 'item_cnt_day':['sum'], 'date':['min','max']})



aggr = aggr.reset_index()

aggr
aggr['duration'] = (aggr['date']['max'] - aggr['date']['min']).dt.days.astype(np.int16)

aggr['duration'] += 1

aggr['item_cnt_day_sum'] = aggr['item_cnt_day']['sum']

aggr['item_price_mean'] = aggr['item_price']['mean']



aggr = aggr.drop(columns = ['date', 'item_cnt_day', 'item_price'])



aggr
monthly_sales = aggr.groupby(['shop_id', 'item_id']).agg({'item_cnt_day_sum':['sum'], 'item_price_mean':['mean'], 'duration':['sum']})



monthly_sales = monthly_sales.reset_index()

monthly_sales
monthly_sales['item_cnt_month'] = monthly_sales['item_cnt_day_sum']['sum']/30

monthly_sales['item_price_month'] = monthly_sales['item_price_mean']['mean']

monthly_sales['sales_duration'] = monthly_sales['duration']['sum']



monthly_sales = monthly_sales.drop(columns = ['item_cnt_day_sum','item_price_mean','duration'])

monthly_sales
#Delete unwanted dataframes

del aggr



gc.collect()
def colnamecheck(df):

    cols = []

    

    for x in df.columns:

        if type(x)!= str:

            cols.append(''.join(x))

        else:

            cols.append(x)

            

    df.columns = cols
df = pd.merge(monthly_sales, items, on='item_id')

df = df.drop(columns = ['item_id'])



colnamecheck(df)



df1 = pd.merge(df, shops, on='shop_id')

df2 = pd.merge(df1, item_cats, on='item_category_id')



monthly_sales = df2



del df, df1, df2

gc.collect()
def EncodeColumn(df, old_col, new_col):

    

    enc = LabelEncoder()

    

    df[new_col] = enc.fit_transform(df[old_col])
EncodeColumn(monthly_sales, 'item_name', 'item_name_enc')

EncodeColumn(new_test, 'item_name', 'item_name_enc')



EncodeColumn(monthly_sales, 'shop_name', 'shop_name_enc')

EncodeColumn(new_test, 'shop_name', 'shop_name_enc')



EncodeColumn(monthly_sales, 'item_category_name', 'item_category_name_enc')

EncodeColumn(new_test, 'item_category_name', 'item_category_name_enc')
monthly_sales = monthly_sales.drop(columns = ['item_name'])

monthly_sales = monthly_sales.drop(columns = ['shop_name'])

monthly_sales = monthly_sales.drop(columns = ['item_category_name'])



new_test = new_test.drop(columns = ['item_name'])

new_test = new_test.drop(columns = ['shop_name'])

new_test = new_test.drop(columns = ['item_category_name'])
xgb = XGBRegressor(

    learning_rate=0.01,

    max_depth=3,

    n_estimators=1000, 

    colsample_bytree=0.8, 

    subsample=0.8,     

)
X = monthly_sales.drop(columns = ['item_price_month', 'sales_duration', 'item_cnt_month'])

y = monthly_sales['item_price_month']



xgb.fit(X, y)



preds = xgb.predict(new_test.drop(columns=['ID']))

new_test['item_price_month'] = preds

monthly_sales = monthly_sales.drop(columns = ['item_price_month'])

monthly_sales['item_price_month'] = y
X = monthly_sales.drop(columns = ['sales_duration', 'item_cnt_month'])

y = monthly_sales['sales_duration']



xgb.fit(X, y)



preds = xgb.predict(new_test.drop(columns=['ID']))

new_test['sales_duration'] = preds

monthly_sales = monthly_sales.drop(columns = ['sales_duration'])

monthly_sales['sales_duration'] = y
X = monthly_sales.drop(columns = ['item_cnt_month'])

y = monthly_sales['item_cnt_month']



xgb.fit(X, y)



preds = xgb.predict(new_test.drop(columns=['ID']))

new_test['item_cnt_month'] = preds
result = pd.DataFrame({'ID':new_test['ID'], 'item_cnt_month':new_test['item_cnt_month']})
result
result.to_csv('submission.csv', index=False)