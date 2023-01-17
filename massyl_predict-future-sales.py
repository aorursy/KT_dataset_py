# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose

import gc



plt.rcParams["figure.figsize"] = (10,6)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
test = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')

test.info()
test.head()
sales = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')

sales.info()
sales.head()
sales = sales.rename(columns={'date_block_num':'month_num'})

sales['date'] = pd.to_datetime(sales['date'], dayfirst=True)

sales['month'] = sales['date'].dt.month

time_dim = ['date', 'month', 'month_num']
label= 'item_cnt_day'

y_label= 'item_cnt_month'
sales.describe()
sns.distplot(sales[label], hist=False)
min_thresh, max_thresh = np.percentile(sales[label], [0.5,99.9995])

sales= sales[sales[label].between(min_thresh, max_thresh)]

print (min_thresh, max_thresh)
sns.distplot(sales['item_price'], hist=False)
min_thresh, max_thresh = np.percentile(sales['item_price'], [0.5,99.5])

sales= sales[sales['item_price'].between(min_thresh, max_thresh)]

print (min_thresh, max_thresh)
nunique = [len(sales[i].unique()) for i in sales.columns]

sns.barplot(sales.columns, nunique)

nunique
pd.pivot_table( sales, 'month_num', 'shop_id', aggfunc='max').plot()
sales.groupby('date')[label].sum().plot(title="sales by date")
tmp=sales.groupby('shop_id')[label].count()

tmp.plot(kind='bar', title="sales by shop")

top = tmp.sort_values().tail().index.tolist()

worst = tmp.sort_values().head().index.tolist()

top
tmp=sales.groupby('shop_id')['date'].nunique()

tmp.plot(kind='bar', title="unique dates by shop")

top = tmp.sort_values().tail().index.tolist()

worst = tmp.sort_values().head().index.tolist()

top
set(test['shop_id'].unique()).issubset(sales['shop_id'].unique())

set(test['item_id'].unique()).issubset(sales['item_id'].unique())

tmp=sales[sales['shop_id'].isin(test['shop_id'].unique())] .groupby('shop_id')['date'].nunique()

tmp.plot(kind='bar', title="unique dates by shop from test set")
sales = pd.pivot_table(sales, label, [ 'month_num', 'shop_id', 'item_id', 'item_price', 'month'], aggfunc=['mean', 'sum']).reset_index()

sales.columns = sales.columns.droplevel(1)

sales.head()
g= sns.barplot('month', 'mean', data=sales, palette='tab20', n_boot=100)

g.set(title="Average sales by Month (date & store)")
g= sns.barplot('month', 'sum', data=sales, palette='tab20', n_boot=100)

g.set(title="Average sales by Month (date & store)")
tmp=sales.groupby('month')['shop_id'].nunique()

tmp.plot(kind='bar', title="unique shops by month")
tmp=sales.groupby('month_num')['shop_id'].nunique()

tmp.plot(kind='bar', title="unique shops by month num")
label = 'sum'
tmp = sales[sales['shop_id'].isin(top)].copy()

tmp = pd.pivot_table(tmp, label, ['month_num', 'shop_id'], aggfunc='sum').reset_index()



g= sns.FacetGrid(tmp, row='shop_id', aspect = 6)

g.map(plt.plot, 'month_num', label)

# sales.groupby('date')[label].sum().plot(title="sales by date")
tmp = sales[sales['shop_id'].isin(worst)].copy()

tmp = pd.pivot_table(tmp, label, ['month_num', 'shop_id'], aggfunc='sum').reset_index()



g= sns.FacetGrid(tmp, row='shop_id', aspect = 3)

g.map(plt.plot, 'month_num', label)

# sales.groupby('date')[label].sum().plot(title="sales by date")
tmp = test[['ID', 'shop_id', 'item_id']] .copy()

tmp = tmp.loc[tmp.index.repeat(34)]

tmp ['month_num'] = [np.mod(i, 34) for i in range(7282800)]

merge = pd.merge(tmp, sales, 'left', ['shop_id', 'item_id', 'month_num'], suffixes=('_y', ''))

merge ['sum'] = merge ['sum'].fillna(0)

merge ['mean'] = merge ['mean'].fillna(0)

merge

print ([merge[i].count() for i in merge .columns])
tmp= pd.pivot_table(merge.dropna(), 'month_num', ['shop_id', 'item_id'], aggfunc=[max, min]).reset_index()

tmp.columns = tmp.columns.droplevel(1)

tmp['status'] = tmp.apply(lambda x: True if (x['max'] >30) & (x['max'] - x['min']>13) else False,1)
merge = pd.merge(merge, tmp[['shop_id', 'item_id', 'status']], 'left', ['shop_id', 'item_id'])

merge.info()
shop=59

item=5037

tmp= merge[(merge['shop_id']==shop) & (merge['item_id']==item)].copy()

sns.lineplot('month_num' , 'sum', data=tmp)
tmp = pd.pivot_table (sales, 'item_price', ['shop_id', 'item_id'] ).reset_index()

# tmp.columns = tmp.columns.droplevel(1)

# tmp.columns = ['shop_id', 'item_id', 'mean_price']

merge = pd.merge(merge[['ID', 'shop_id', 'item_id', 'month_num', 'month','mean','sum', 'status']] , tmp, 'left', ['shop_id', 'item_id'])

merge.head()
merge['month'] = np.mod(merge['month_num']+1, 12)

merge['status'] = merge['status'].fillna(False)

merge['item_price'] = merge['item_price'].fillna(0)
X = merge[['shop_id', 'item_id', 'month_num', 'month', 'status', 'item_price']].values

y = merge['mean'].values



X_test= test.copy()

X_test = pd.merge(X_test, merge[['shop_id', 'item_id', 'status', 'item_price']], 'left', ['shop_id', 'item_id'])

X_test.drop_duplicates(inplace=True)

X_test['month_num']= 34

X_test['month']= 11

X_test
X_test_data = X_test[['shop_id', 'item_id', 'month_num', 'month', 'status', 'item_price']] .values
from sklearn.ensemble import RandomForestRegressor



reg = RandomForestRegressor(random_state=1, n_estimators=50)

reg.fit(X, y)

X_test [y_label] = reg.predict(X_test_data)
tmp = X_test[['ID', y_label]].copy()

print (len(tmp))

tmp.to_csv('sub3_rf', index=False)
[['shop_id', 'item_id', 'month_num', 'month', 'status', 'item_price']]

reg.feature_importances_