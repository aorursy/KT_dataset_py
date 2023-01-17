# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
plt.gcf().set_size_inches(20, 10)
import tensorflow as tf
print(tf.__version__)
%matplotlib inline
data = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv')
data.head()
calender = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')
calender.head()
data.dtypes
data = pd.merge(data,calender,on='wm_yr_wk')
data.head()
data_stats = data.describe().T
data_stats.reset_index(inplace=True)
data_stats.rename({'index':'fields'},axis=1,inplace=True)
data_stats
# sns.boxplot(x=data_stats.fields,y=data_stats.iloc[1])
data_stats.iloc[1]
data.isna().sum()/data.count() * 100
plt.figure(figsize=(15,5))
plt.title('store wise sales')
sns.boxplot(x='store_id',y='sell_price',data=data)
plt.figure(figsize=(15,5))
sns.heatmap(data.corr(),annot=True,)
year_sell_price = pd.DataFrame(data.groupby(['year','store_id'])['sell_price'].sum().reset_index())
plt.figure(figsize=(15,5))
plt.title('year wise sales of stores')
sns.barplot(x='year',y='sell_price',data=year_sell_price,hue='store_id')
month_sell_price = pd.DataFrame(data.groupby(['month','store_id'])['sell_price'].sum().reset_index())
plt.figure(figsize=(15,5))
sns.barplot(x='month',y='sell_price',hue='store_id',data=month_sell_price,estimator=np.sum)
item_store_sell_price=pd.DataFrame(data.groupby(['item_id','store_id'])['sell_price'].sum().reset_index())
plt.figure(figsize=(15,4))
sns.catplot(x="item_id",y='sell_price',data=item_store_sell_price)
#explore the store CA_1
CA_1_data = data[data.store_id == 'CA_1']
CA_1_data.head()
plt.figure(figsize=(15,5))
day_wise_sum = pd.DataFrame(CA_1_data.groupby('date')['item_id'].count().reset_index())
sns.lineplot(x='date',y='item_id',data=day_wise_sum)
plt.figure(figsize=(15,5))
day_wise_sum = pd.DataFrame(data[data.store_id == 'TX_1'].groupby('date')['item_id'].count().reset_index())
plt.title('TX_1 day wise plot')
sns.lineplot(x='date',y='item_id',data=day_wise_sum)
plt.figure(figsize=(20,10))
month_wise_sum = pd.DataFrame(CA_1_data.groupby('month')['sell_price'].sum().reset_index())
plt.subplot(4,2,1)
plt.title('store CA_1 month wise sale')
sns.lineplot(x='month',y='sell_price',data=month_wise_sum)
plt.subplot(4,2,2)
month_wise_sum = pd.DataFrame(data[data.store_id == 'CA_2'].groupby('month')['sell_price'].sum().reset_index())
plt.title('store CA_2 month wise sale')
sns.lineplot(x='month',y='sell_price',data=month_wise_sum)
plt.subplot(4,2,3)
month_wise_sum = pd.DataFrame(data[data.store_id == 'CA_3'].groupby('month')['sell_price'].sum().reset_index())
plt.title('store CA_3 month wise sale')
sns.lineplot(x='month',y='sell_price',data=month_wise_sum)
plt.subplot(4,2,4)
month_wise_sum = pd.DataFrame(data[data.store_id == 'CA_4'].groupby('month')['sell_price'].sum().reset_index())
plt.title('store CA_4 month wise sale')
sns.lineplot(x='month',y='sell_price',data=month_wise_sum)
plt.figure(figsize=(20,10))
month_wise_sum = pd.DataFrame(data[data.store_id == 'TX_1'].groupby('month')['sell_price'].sum().reset_index())
plt.subplot(3,1,1)
plt.title('store TX_1 month wise sale')
sns.lineplot(x='month',y='sell_price',data=month_wise_sum)
plt.subplot(3,1,2)
month_wise_sum = pd.DataFrame(data[data.store_id == 'TX_2'].groupby('month')['sell_price'].sum().reset_index())
plt.title('store TX_2 month wise sale')
sns.lineplot(x='month',y='sell_price',data=month_wise_sum)
plt.subplot(3,1,3)
month_wise_sum = pd.DataFrame(data[data.store_id == 'TX_3'].groupby('month')['sell_price'].sum().reset_index())
plt.title('store TX_3 month wise sale')
sns.lineplot(x='month',y='sell_price',data=month_wise_sum)
plt.figure(figsize=(20,10))
month_wise_sum = pd.DataFrame(data[data.store_id == 'WI_1'].groupby('month')['sell_price'].sum().reset_index())
plt.subplot(3,1,1)
plt.title('store WI_1 month wise sale')
sns.lineplot(x='month',y='sell_price',data=month_wise_sum)
plt.subplot(3,1,2)
month_wise_sum = pd.DataFrame(data[data.store_id == 'WI_2'].groupby('month')['sell_price'].sum().reset_index())
plt.title('store WI_2 month wise sale')
sns.lineplot(x='month',y='sell_price',data=month_wise_sum)
plt.subplot(3,1,3)
month_wise_sum = pd.DataFrame(data[data.store_id == 'WI_3'].groupby('month')['sell_price'].sum().reset_index())
plt.title('store WI_3 month wise sale')
sns.lineplot(x='month',y='sell_price',data=month_wise_sum)
data.store_id.value_counts()
