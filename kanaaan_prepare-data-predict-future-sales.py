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
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import numpy.random as nr
import math

%matplotlib inline
data = pd.read_csv('../input/sales_train.csv')
print(data.head(3))
data.info()
# Check for negative item_cnt_day (could be a return ?, invalid values, I don't think so)
data[data['item_cnt_day']<0]['item_cnt_day'].value_counts()
plt.plot(data[data['item_cnt_day']<0]['item_id'].value_counts().sort_index())
data_filtered=data.loc[data['item_cnt_day']>0]
data_filtered.info()
data=data_filtered
item_categories = pd.read_csv('../input/items.csv')
item_categories.head(3)
dt=pd.merge(data, item_categories, how='inner')
dt.sort_values(by=['date'], inplace=True)
dt.head(3)
## Drop column 
columns=['date','item_price','item_name']
for c in columns:
    if c in dt:
        dt.drop(c, axis = 1, inplace = True)
dt[(dt['item_cnt_day']>0)].head(3)
dtf=dt.groupby(['date_block_num', 'shop_id','item_id'])[["item_cnt_day"]].sum().reset_index()
print(data.size)
print(dtf.size)
dtf.hist(figsize=(15,20))
plt.figure()
pd.plotting.scatter_matrix(dtf[['item_cnt_day','item_id','shop_id','date_block_num']],figsize=(10,10))
plt.figure()
dtf[(dtf['item_id']==2929) & (dtf['shop_id']==0)]
dt[(dt['item_id']==2929) & (dt['shop_id']==0)]
test_shop_id=dt.groupby(['shop_id'])[["item_cnt_day"]].sum().reset_index()
test_shop_id.head()
plt.bar(test_shop_id['shop_id'],test_shop_id ["item_cnt_day"])


test_item_id=dt.groupby(['item_id'])[["item_cnt_day"]].sum().reset_index()
plt.plot(test_item_id[(test_item_id['item_id']!=20949)]['item_id'],test_item_id[(test_item_id['item_id']!=20949)] ["item_cnt_day"])
plt.plot(test_item_id[(test_item_id['item_cnt_day']<=10000)]['item_id'],test_item_id[(test_item_id['item_cnt_day']<=10000)]["item_cnt_day"])

print(test_item_id[(test_item_id['item_id']!=20949)]['item_id'].describe())
print(test_item_id[(test_item_id['item_cnt_day']>12000)]['item_id'].value_counts())
test_item_id=dt.groupby(['item_category_id'])[["item_cnt_day"]].sum().reset_index()
plt.plot(test_item_id['item_category_id'],test_item_id["item_cnt_day"])

plt.plot(dt.groupby(['date_block_num'])[["item_cnt_day"]].sum())
dt_filtered=dt.loc[(dt['date_block_num'] ==9) | (dt['date_block_num'] ==10) | (dt['date_block_num'] ==21)| (dt['date_block_num'] ==22) | (dt['date_block_num'] ==33)]
print(dt_filtered.size)
dt_filtered['date_block_num'].value_counts()
pd.options.mode.chained_assignment = None  # default='warn'

idx=dt_filtered.loc[(dt_filtered['date_block_num'] ==9)].index.values
dt_filtered.at[idx,'date_block_num']=0
dt_filtered.at[idx,'year']=1

idx=dt_filtered.loc[(dt_filtered['date_block_num'] ==10)].index.values
dt_filtered.at[idx,'date_block_num']=1
dt_filtered.at[idx,'year']=1

idx=dt_filtered.loc[(dt_filtered['date_block_num'] ==21)].index.values
dt_filtered.at[idx,'date_block_num']=0
dt_filtered.at[idx,'year']=2

idx=dt_filtered.loc[(dt_filtered['date_block_num'] ==22)].index.values
dt_filtered.at[idx,'date_block_num']=1
dt_filtered.at[idx,'year']=2

idx=dt_filtered.loc[(dt_filtered['date_block_num'] ==33)].index.values
dt_filtered.at[idx,'date_block_num']=0
dt_filtered.at[idx,'year']=3
print(dt_filtered['date_block_num'].value_counts())
print(dt_filtered['year'].value_counts())
plt.plot(dt_filtered.groupby(['date_block_num'])[["item_cnt_day"]].sum())
print(dt_filtered.head())

dt_filtered.to_csv('sales_train_trans_filtered.csv', sep=',',index=False)
dt.to_csv('sales_train_trans.csv', sep=',',index=False)
sales_test = pd.read_csv('../input/test.csv')
sales_test.head(3)
sales_test1=pd.merge(sales_test, item_categories, how='inner')
sales_test1.sort_values(by=['ID'], inplace=True)
sales_test1.head(3)
sales_test1['shop_id'].value_counts()
sales_test1.info()
sales_test1.isnull().sum()
sales_test1['item_id'].value_counts().count()
sales_test1['item_category_id'].value_counts().count()
dt['item_category_id'].value_counts().count()
#pd.concat([pd.unique(sales_test1['item_category_id']),pd.unique(sales_test1['item_category_id'])]).drop_duplicates(keep=False)
#print("sales_test1['item_category_id']-->",pd.unique(sales_test1['item_category_id']))
#print("dt['item_category_id']-->",pd.unique(dt['item_category_id']))
#print("concatenate-->", np.concatenate((pd.unique(sales_test1['item_category_id']),pd.unique(dt['item_category_id'])),axis=0))
np.unique(np.concatenate((pd.unique(sales_test1['item_category_id']),pd.unique(dt['item_category_id'])),axis=0))

a=set(pd.unique(dt['item_category_id']));
b=set(pd.unique(sales_test1['item_category_id']));

list(a-b)

sales_test1.drop('item_name', axis = 1, inplace = True)
sales_test1.to_csv('sales_test1.csv', sep=',',index=False)