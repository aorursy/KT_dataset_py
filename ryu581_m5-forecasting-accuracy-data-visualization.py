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
import numpy as np
import pandas as pd
from pandas import DataFrame, Series

import matplotlib.pyplot as plt
import seaborn as sns

import datetime
#データの読み込み
calendar = pd.read_csv('../input/m5-forecasting-accuracy/calendar.csv')
sales = pd.read_csv('../input/m5-forecasting-accuracy/sales_train_validation.csv')
price = pd.read_csv('../input/m5-forecasting-accuracy/sell_prices.csv')
# salesのデータフレームを見てみる
sales.head(10)
sales.groupby('cat_id').mean()
# ランダムにインデックス番号を作る。
# 適当にアイテムを指定して時系列での売上の推移を見てみる。
index_list = []
while len(index_list) <= 30:
    i = np.random.randint(30000)
    index_list.append(i)
    set(index_list)
    list(index_list)
sales_by_id = sales.copy().iloc[index_list, :].set_index('id')
sales_by_id.drop(['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], axis=1, inplace=True)
sales_by_id = sales_by_id.T
sales_by_id
sales_by_id.plot(legend=True, figsize=(30, 8))
# もとのsalesデータフレームの、日付ごとにsumを実行する
sales_agg = sales.copy().set_index('id')
sales_agg.drop(['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], axis=1, inplace=True)
sales_agg = sales_agg.T
sales_agg['day total'] = sales_agg.sum(axis=1)
sales_agg = pd.DataFrame(sales_agg['day total'])
sales_agg.plot(legend=True, figsize=(30, 8))
# まずは、日毎のデータを月ごとに集計する。
# その前に、売上データフレームの日付ラベルを標準的な表記方法に変換したい カレンダー見て始点を確認する
calendar.head()
# salesデータフレームをいじっていく まずは、もとのsalesデータフレームを転置して、indexを標準日付表記に変換する
sales_ver1 = sales.copy().set_index('id')
sales_ver1 = sales_ver1.transpose()
sales_ver1 = sales_ver1.drop(['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'])
indexes = calendar['date'].drop(range(1913, 1969, 1))
sales_ver1 = sales_ver1.set_index(indexes)
# 1月の売上データは3日分しか無いので、とりあえずそれらは省く
sales_ver1 = sales_ver1.drop(['2011-01-29', '2011-01-30', '2011-01-31'])
sales_ver1
# 日毎のすべての売上を集計する
sales_ver1['day total'] = sales_ver1.sum(axis=1)
sales_ver1.index = pd.to_datetime(sales_ver1.index)
sales_ver1.resample(rule='M').sum()
sales_by_state = pd.DataFrame(sales.groupby('state_id').sum())
sales_by_state = sales_by_state.transpose()
sales_by_state.set_index(indexes)
sales_by_state['day total'] = sales_by_state.sum(axis=1)
sales_by_state
sales_by_state.set_index(indexes, inplace=True)
sales_by_state.drop(['2011-01-29', '2011-01-30', '2011-01-31'], inplace=True)
# 週毎の、月ごとの売上データを集計することができた
sales_by_state.index = pd.to_datetime(sales_by_state.index)
sales_by_state.resample(rule='M').sum()
sales_by_state.index = pd.to_datetime(sales_by_state.index)
sales_by_state = pd.DataFrame(sales_by_state.resample(rule='M').sum())
sales_by_state
sales_by_state.plot(legend=True, figsize=(20, 8))
sales_by_cat = pd.DataFrame(sales.groupby('cat_id').sum())
sales_by_cat = sales_by_cat.transpose()
sales_by_cat.set_index(indexes, inplace=True)
sales_by_cat.drop(['2011-01-29', '2011-01-30', '2011-01-31'], inplace=True)
sales_by_cat.index = pd.to_datetime(sales_by_cat.index)
sales_by_cat = pd.DataFrame(sales_by_cat.resample(rule='M').sum())
sales_by_cat.plot(legend=True, figsize=(20, 8))
sns.countplot(sales['cat_id'])
avg = pd.DataFrame(sales.groupby('cat_id').mean())
avg = avg.transpose()
avg.set_index(indexes, inplace=True)
avg.drop(['2011-01-29', '2011-01-30', '2011-01-31'], inplace=True)
# avg.index = pd.to_datetime(avg.index)
# avg = pd.DataFrame(avg.resample(rule='M').mean())
avg.plot(figsize=(20, 8))
# お店ごとの時系列データに編集する。
sales_by_store = sales.groupby('store_id').sum()
sales_by_store = sales_by_store.transpose()
sales_by_store.set_index(indexes, inplace=True)
sales_by_store.drop(['2011-01-29', '2011-01-30', '2011-01-31'], inplace=True)
sales_by_store.index = pd.to_datetime(sales_by_store.index)
agg_sales_by_store = pd.DataFrame(sales_by_store.resample(rule='M').sum())
sales_CA = agg_sales_by_store[['CA_1', 'CA_2', 'CA_3', 'CA_4']]
sales_TX = agg_sales_by_store[['TX_1', 'TX_2', 'TX_3']]
sales_WI = agg_sales_by_store[['WI_1', 'WI_2', 'WI_3']]
fig, axes = plt.subplots(ncols=3)
sales_CA.plot(ax=axes[0], figsize=(20,10), yticks=range(50000, 225001, 25000), ylim=[25000, 225000]);axes[0].set_title('CA')
sales_TX.plot(ax=axes[1], figsize=(20,10), yticks=range(50000, 225001, 25000), ylim=[25000, 225000]);axes[1].set_title('TX')
sales_WI.plot(ax=axes[2], figsize=(20,10), yticks=range(50000, 225001, 25000), ylim=[25000, 225000]);axes[2].set_title('WI')

plt.figure(figsize=(20, 8))
plt.show()
agg_sales_by_store
# まずはCAだけのデータを集める。
sales_CA_df = sales[sales['state_id'] == 'CA'].set_index('id')
sales_CA_df.drop(['item_id', 'cat_id', 'store_id', 'state_id'], axis=1, inplace=True)
sales_CA_df = pd.DataFrame(sales_CA_df.groupby('dept_id').sum()).T
sales_CA_df.set_index(indexes, inplace=True)
sales_CA_df.drop(['2011-01-29', '2011-01-30', '2011-01-31'], inplace=True)
sales_CA_df.describe()
sales_CA_df.index = pd.to_datetime(sales_CA_df.index)
sales_CA_df = sales_CA_df.resample(rule='M').sum()
sales_TX_df = sales[sales['state_id'] == 'TX'].set_index('id')
sales_TX_df.drop(['item_id', 'cat_id', 'store_id', 'state_id'], axis=1, inplace=True)
sales_TX_df = pd.DataFrame(sales_TX_df.groupby('dept_id').sum()).T
sales_TX_df.set_index(indexes, inplace=True)
sales_TX_df.drop(['2011-01-29', '2011-01-30', '2011-01-31'], inplace=True)
sales_TX_df.describe()
sales_TX_df.index = pd.to_datetime(sales_TX_df.index)
sales_TX_df = sales_TX_df.resample(rule='M').sum()
sales_WI_df = sales[sales['state_id'] == 'WI'].set_index('id')
sales_WI_df.drop(['item_id', 'cat_id', 'store_id', 'state_id'], axis=1, inplace=True)
sales_WI_df = pd.DataFrame(sales_WI_df.groupby('dept_id').sum()).T
sales_WI_df.set_index(indexes, inplace=True)
sales_WI_df.drop(['2011-01-29', '2011-01-30', '2011-01-31'], inplace=True)
sales_WI_df.index = pd.to_datetime(sales_WI_df.index)
sales_WI_df = sales_WI_df.resample(rule='M').sum()
fig, axes = plt.subplots(nrows=3,ncols=7, sharey=True)
for i in list(sales_CA_df.columns):
    sales_CA_df[i].plot(ax=axes[0][list(sales_CA_df.columns).index(i)], figsize=(30, 12));axes[0][list(sales_CA_df.columns).index(i)].set_title(i)
for i in list(sales_TX_df.columns):
    sales_TX_df[i].plot(ax=axes[1][list(sales_TX_df.columns).index(i)], figsize=(30, 12))
for i in list(sales_WI_df.columns):
    sales_WI_df[i].plot(ax=axes[2][list(sales_WI_df.columns).index(i)], figsize=(30, 12))

plt.figure(figsize=(20, 8))
plt.show()
sales_CA_df['FOODS_1'].plot(ax=axes[0], legend=False);axes[0].set_title('FOODS_1')
sales_CA_df['FOODS_2'].plot(ax=axes[1], legend=False);axes[1].set_title('FOODS_2')

plt.figure(figsize=(20, 8))
plt.show()
sales_global = sales.copy().set_index('id').T
sales_global.drop(['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], inplace=True)
sales_global['day total'] = sales_global.sum(axis=1)
sales_global.set_index(indexes, inplace=True)
sales_global.drop(['2011-01-29', '2011-01-30', '2011-01-31'], inplace=True)
sales_global.index = pd.to_datetime(sales_global.index)
sales_global = pd.DataFrame(sales_global['day total'])
sales_global.plot(figsize=(20, 8), ylim=(15000, 60000))
sales_global['MA_7'] = sales_global.rolling(150).mean()
for i in range(0, 150, 1):
    sales_global['MA_7'][i] = sales_global['day total'][0:i+1].mean()
sales_global['MA_7'][0] = sales_global['day total'][0]
sales_global
sales_global.plot(figsize=(20, 8))
sales_month_week = sales.set_index('id').transpose()
sales_month_week = sales_month_week.drop(['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'])
sales_month_week.set_index(indexes, inplace=True)
sales_month_week.index = pd.to_datetime(sales_month_week.index)
sales_month_week['day of week'] = sales_month_week.index.weekday
sales_month_week.head()



