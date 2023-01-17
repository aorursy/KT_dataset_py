# This Python 3 environment comes with many helpful analytics libraries installed



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import time

from datetime import datetime

import os



# read data

junuary = pd.read_csv('/kaggle/input/tripdata/green_tripdata_2018-01.csv')

february = pd.read_csv('/kaggle/input/tripdata/green_tripdata_2018-02.csv')

march = pd.read_csv('/kaggle/input/tripdata/green_tripdata_2018-03.csv')

april = pd.read_csv('/kaggle/input/tripdata/green_tripdata_2018-04.csv')

may = pd.read_csv('/kaggle/input/tripdata/green_tripdata_2018-05.csv')

june = pd.read_csv('/kaggle/input/tripdata/green_tripdata_2018-06.csv')



stock_market = pd.read_csv('/kaggle/input/tripdata/stock_maket.csv')
# 查看数据维度

print(junuary.shape)

print(february.shape)

print(march.shape)

print(april.shape)

print(may.shape)

print(june.shape)

print(stock_market.shape)
# 将lpep_pickup_datetime、lpep_dropoff_datetime分为 时、日，星期

junuary['lpep_pickup_datetime'] = pd.to_datetime(junuary['lpep_pickup_datetime'])

february['lpep_pickup_datetime'] = pd.to_datetime(february['lpep_pickup_datetime'])

march['lpep_pickup_datetime'] = pd.to_datetime(march['lpep_pickup_datetime'])

april['lpep_pickup_datetime'] = pd.to_datetime(april['lpep_pickup_datetime'])

may['lpep_pickup_datetime'] = pd.to_datetime(may['lpep_pickup_datetime'])

june['lpep_pickup_datetime'] = pd.to_datetime(june['lpep_pickup_datetime'])



junuary['lpep_dropoff_datetime'] = pd.to_datetime(junuary['lpep_dropoff_datetime'])

february['lpep_dropoff_datetime'] = pd.to_datetime(february['lpep_dropoff_datetime'])

march['lpep_dropoff_datetime'] = pd.to_datetime(march['lpep_dropoff_datetime'])

april['lpep_dropoff_datetime'] = pd.to_datetime(april['lpep_dropoff_datetime'])

may['lpep_dropoff_datetime'] = pd.to_datetime(may['lpep_dropoff_datetime'])

june['lpep_dropoff_datetime'] = pd.to_datetime(june['lpep_dropoff_datetime'])



junuary['hour1'] = junuary['lpep_pickup_datetime'].dt.hour

junuary['hour2'] = junuary['lpep_dropoff_datetime'].dt.hour

february['hour1'] = february['lpep_pickup_datetime'].dt.hour

february['hour2'] = february['lpep_dropoff_datetime'].dt.hour

march['hour1'] = march['lpep_pickup_datetime'].dt.hour

march['hour2'] = march['lpep_dropoff_datetime'].dt.hour

april['hour1'] = april['lpep_pickup_datetime'].dt.hour

april['hour2'] = april['lpep_dropoff_datetime'].dt.hour

may['hour1'] = may['lpep_pickup_datetime'].dt.hour

may['hour2'] = may['lpep_dropoff_datetime'].dt.hour

june['hour1'] = june['lpep_pickup_datetime'].dt.hour

june['hour2'] = june['lpep_dropoff_datetime'].dt.hour



junuary['day1'] = junuary['lpep_pickup_datetime'].dt.day

junuary['day2'] = junuary['lpep_dropoff_datetime'].dt.day

february['day1'] = february['lpep_pickup_datetime'].dt.day

february['day2'] = february['lpep_dropoff_datetime'].dt.day

march['day1'] = march['lpep_pickup_datetime'].dt.day

march['day2'] = march['lpep_dropoff_datetime'].dt.day

april['day1'] = april['lpep_pickup_datetime'].dt.day

april['day2'] = april['lpep_dropoff_datetime'].dt.day

may['day1'] = may['lpep_pickup_datetime'].dt.day

may['day2'] = may['lpep_dropoff_datetime'].dt.day

june['day1'] = june['lpep_pickup_datetime'].dt.day

june['day2'] = june['lpep_dropoff_datetime'].dt.day



junuary['month'] = junuary['lpep_pickup_datetime'].dt.month

february['month'] = february['lpep_pickup_datetime'].dt.month

march['month'] = march['lpep_pickup_datetime'].dt.month

april['month'] = april['lpep_pickup_datetime'].dt.month

may['month'] = may['lpep_pickup_datetime'].dt.month

june['month'] = june['lpep_pickup_datetime'].dt.month



# 合并1~6月数据

all_data = pd.concat([junuary, february, march, april, may, june], axis=0)

all_data.shape
# 将lpep_pickup_datetime、lpep_dropoff_datetime分 日期、 时刻,时刻格式化后仅保留 时：分

all_data['pickup_date'] = pd.to_datetime(all_data['lpep_pickup_datetime'].dt.normalize())

all_data['dropoff_date'] = pd.to_datetime(all_data['lpep_dropoff_datetime'].dt.normalize())



all_data['pickup_time'] = all_data['lpep_pickup_datetime'].dt.time

all_data['pickup_time'] = all_data['pickup_time'].apply(lambda x:x.strftime('%H:%M'))

all_data['dropoff_time'] = all_data['lpep_pickup_datetime'].dt.time

all_data['dropoff_time'] = all_data['dropoff_time'].apply(lambda x:x.strftime('%H:%M'))

stock_market['Date'] = pd.to_datetime(stock_market['Date'])

stock_market.head(5)
# 将出租车数据与股市数据合并

all_data = all_data.merge(stock_market, left_on='pickup_date', right_on='Date',how='left')
all_data.loc[all_data.day1==3]
# 1~6月中，每分钟的乘客总数

fig, axes = plt.subplots(3,2,figsize=(17, 10), dpi=100)

for i in range(0,6):

    d = all_data[all_data['lpep_pickup_datetime'].dt.month == i+1][['pickup_time', 'passenger_count']].set_index('pickup_time')

    d = d.groupby('pickup_time').sum()

    d.plot(ax=axes[i%3][i//3], 

    alpha=0.8, color='tab:blue').set_ylabel('passenger_count', fontsize=13);

    

    axes[i%3][i//3].legend();

    axes[i%3][i//3].set_title('month {}'.format(i+1), fontsize=13);

    plt.subplots_adjust(hspace=0.45)
# 1~6月，每日小费均值变化

fig, axes = plt.subplots(3,2,figsize=(15, 13), dpi=100)

for i in range(0,6):

    d = all_data[all_data['lpep_pickup_datetime'].dt.month == i+1][['pickup_date', 'tip_amount']]

    dm = d.groupby(['pickup_date']).mean()[3:-1]

    dm.plot(ax=axes[i%3][i//3]).set_ylabel('tip mean', fontsize=13);

    

    axes[i%3][i//3].set_title('month {}'.format(i+1), fontsize=13);
# 1~6月，每月小费的相关统计量

mm=[]

m5=[]

m7=[]

for i in range(1,7):

    mm.append(all_data.loc[(all_data.month==i) & (all_data.tip_amount>0),'tip_amount'].describe()['mean'])

    m5.append(all_data.loc[(all_data.month==i) & (all_data.tip_amount>0),'tip_amount'].describe()['50%'])

    m7.append(all_data.loc[(all_data.month==i) & (all_data.tip_amount>0),'tip_amount'].describe()['75%'])



d = {'mean':mm,'50%':m5,'75%':m7}

d= pd.DataFrame(d)

fig,ax1 = plt.subplots(figsize=(17, 6))

d.plot(kind='bar', ax=ax1)



plt.xlabel('month')

plt.ylabel('tip_amount')
# 每分钟乘客数 画出折线图

group1 = all_data['passenger_count'].groupby(all_data['pickup_time']).sum()

plt.figure(figsize=(17,7))

group1.plot()

plt.title('passengers Amount per min')

plt.xlabel('pickup time')

plt.ylabel('passenger_count')
# 每小时乘客数，画出条形图

group2 = all_data['passenger_count'].groupby(all_data['hour1']).sum()

plt.figure(figsize=(15,7))

group2.plot(kind='bar')

plt.title('passengers Amount per hour')

plt.xlabel('pickup hour')

plt.ylabel('passenger_count')
# 以高峰时段17时到19时的各个地区的乘客量画条形图

a = all_data.loc[(all_data.hour1==17) | (all_data.hour1==18) | (all_data.hour1==19)]

group3 = a['passenger_count'].groupby(a['PULocationID']).sum()[:100]

plt.figure(figsize=(30,7))

group3.plot(kind='bar')

plt.title('passengers Amount at 5pm to 7pm')

plt.xticks(size=8)

plt.xlabel('pick up location ID')

plt.ylabel('passenger_count')
# 以高峰时段17时到19时的乘客量画条形图，取乘客量最多的前十个地点ID

a = all_data.loc[(all_data.hour1==17) | (all_data.hour1==18) | (all_data.hour1==19)]

group3 = a['passenger_count'].groupby(a['PULocationID']).sum().sort_values(ascending=False)[:10]

plt.figure(figsize=(15,7))

group3.plot(kind='bar')

plt.title('passengers Amount at 5pm to 7pm')

plt.xlabel('pick up location ID')

plt.ylabel('passenger_count')
a = all_data.loc[all_data.Close.notnull()]

# 每天的乘客总量

group4 = a['passenger_count'].groupby(a['pickup_date']).sum()

group5 = a['Close'].groupby(a['pickup_date']).mean()

openprice = a[['Open']].groupby(a['pickup_date']).mean()
# 收盘价与当日乘客总数的趋势关系

fig, ax1 = plt.subplots(figsize=(17,7))

ax2 = ax1.twinx()

ax1.plot(group4.index, group4.values, c='blue',linestyle='--',label='passenger_count')

ax1.legend(loc=2)



ax2.plot(group5.index, group5.values, c='red',label='close_price')

ax2.legend(loc=1)

ax1.tick_params(axis='y', colors='blue')

ax2.tick_params(axis='y', colors='red')

ax1.set_ylabel('passenger_count',size=13)

ax2.set_ylabel('Close_Price',size=13)

plt.title('close price and passenger amount')



plt.show()
b = a.loc[ (a.hour1==17) | (a.hour1==18) | (a.hour1==19)]
a = all_data.loc[all_data.Close.notnull()]

a .drop(['VendorID', 'store_and_fwd_flag', 'trip_distance', 'fare_amount', 'improvement_surcharge',

         'payment_type','tolls_amount','mta_tax', 'Volume','ehail_fee','trip_type','total_amount','RatecodeID'],

        axis=1,inplace=True)

day_tip_sum = a[['tip_amount']].groupby(a['pickup_date']).sum()

day_tip_mean = a[['tip_amount']].groupby(a['pickup_date']).mean()

day_tip_max = a[['tip_amount']].groupby(a['pickup_date']).max()

tip_count = all_data.loc[(all_data['Open'].notnull()==True) &(all_data['lpep_pickup_datetime'].dt.year == 2018)  & (all_data.tip_amount>0)][['pickup_date']]

tip_count = pd.value_counts(tip_count.pickup_date)

tip_count = tip_count.reset_index()

tip_count = tip_count.sort_values(by='index')
# 每日所有单子小费均值 与 每日开盘价的趋势关系

fig, ax = plt.subplots(3,2,figsize=(22,10))

ax2 = ax[0][0].twinx()

ax[0][0].plot(day_tip_mean.index, day_tip_mean.values, c='blue',linestyle='--',label='tip_mean')

ax[0][0].legend(loc=2)



ax2.plot(openprice.index,openprice.values, c='red',label='open_price')

ax2.legend(loc=1)

ax[0][0].tick_params(axis='y', colors='blue')

ax2.tick_params(axis='y', colors='red')

ax[0][0].set_ylabel('tip_mean',size=13)

ax2.set_ylabel('Open_Price',size=13)

plt.title('open price and tips mean')





# 每日所有单子小费总数 与 每日开盘价的趋势关系

ax2 = ax[0][1].twinx()

ax[0][1].plot(day_tip_sum.index, day_tip_sum.values, c='blue',linestyle='--',label='tip_sum')

ax[0][1].legend(loc=2)



ax2.plot(openprice.index,openprice.values, c='red',label='open_price')

ax2.legend(loc=1)

ax[0][1].tick_params(axis='y', colors='blue')

ax2.tick_params(axis='y', colors='red')

ax[0][1].set_ylabel('tip_sum',size=13)

ax2.set_ylabel('Open_Price',size=13)

plt.title('open price and tips sum')





# 每日最高小费 与 每日开盘价的趋势关系

ax2 = ax[1][0].twinx()

ax[1][0].plot(day_tip_max.index, day_tip_max.values, c='blue',linestyle='--',label='tip max')

ax[1][0].legend(loc=2)



ax2.plot(openprice.index,openprice.values,c='red',label='open_price')

ax2.legend(loc=1)

ax[1][0].tick_params(axis='y', colors='blue')

ax2.tick_params(axis='y', colors='red')

ax[1][0].set_ylabel('tip_max',size=13)

ax2.set_ylabel('Open_Price',size=13)

plt.title('open price and tips max')





# 去除离群点，绘制1~6月小费箱线图

sns.boxplot(x='month', y='tip_amount',data=all_data.loc[(all_data.month<7) & (all_data.tip_amount>0)& (all_data.tip_amount<5),['tip_amount','month']],ax=ax[1][1])

ax[1][1].legend(loc=2)



ax[1][1].tick_params(axis='y', colors='blue')

ax[1][1].set_ylabel('tip_mean',size=13)

plt.title('open_price and tips mean')





# 以部分离群点为数据，绘制1~6月小费箱线图

sns.boxplot(x='month', y='tip_amount',data=all_data.loc[(all_data.month<7) & (all_data.tip_amount>50)& (all_data.tip_amount<400),['tip_amount','month']],ax=ax[2][0])

ax[2][0].legend(loc=2)



ax[2][0].tick_params(axis='y', colors='blue')

ax[2][0].set_ylabel('tip_mean',size=13)

plt.title('open_price and tips mean')



# 每日乘客付小费的单子数目 与 每日开盘价的趋势关系

ax[2][1].plot(tip_count['index'], tip_count['pickup_date'], c='blue',linestyle='--',label='tip_mean')

ax2 = ax[2][1].twinx()

ax2.plot(openprice.index,openprice.values, c='red',label='open_price')



ax2.legend(loc=1)

ax[2][1].tick_params(axis='y', colors='blue')

ax2.tick_params(axis='y', colors='red')

ax[2][1].set_ylabel('tip_count',size=13)

ax2.set_ylabel('Open_Price',size=13)
