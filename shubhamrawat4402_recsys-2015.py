import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

%matplotlib inline

plt.style.use('seaborn-white')

import plotly.plotly as py

import plotly.tools as tls



import os

# Input data files are available in the "../input/" directory.

print(os.listdir("../input"))



# Load Datasets

clicks_df=pd.read_csv('../input/yoochoose-data/yoochoose-clicks.dat',

                      names=['session_id','timestamp','item_id','category'],

                      dtype={'category': str})

display("Clicks Data",)

display(clicks_df.head())



buys_df = pd.read_csv('../input/yoochoose-data/yoochoose-buys.dat', names=['session_id', 'timestamp', 'item_id', 'price', 'quantity'])

display("Buys Data",)

display(buys_df.head())
# Explore data



# display(clicks_df.describe())

# display(buys_df.describe())

## The results indicate that buys data session_id and item_id are a subset of clicks data





# (clicks_df.groupby(['session_id', 'item_id']).count()).head(10)

## There can be multiple clicks on the same item in a particular session



(buys_df.groupby(['session_id', 'item_id']).count()).head(10)

## There can be multiple buys of the same item in a particular session.

## Notice this is different from quantity bought.

buys_df[buys_df['session_id']==11]



# Merge clicks and buys data by session and item ids

# merge by left join bcoz clicks_df's session and item ids are a superset of those of buys_df's

df = pd.merge(clicks_df, buys_df, on=['session_id','item_id'], how='left' , suffixes=('_click','_buy'))

print(len(df), len(clicks_df) + len(buys_df))

df.head()
display(clicks_df[clicks_df.session_id==420374])



display(buys_df[buys_df.session_id==420374])



display(df[df.session_id==420374])
clicks_df['timestamp']=pd.to_datetime(clicks_df.timestamp)

clicks_df['hour']=clicks_df.timestamp.dt.hour

clicks_df['weekday']=clicks_df['timestamp'].dt.dayofweek.astype(int)+1



click_hour_info = clicks_df.groupby(['hour'])['session_id'].nunique().reset_index(name='count_h_c')

click_weekday_info = clicks_df.groupby(['weekday'])['session_id'].nunique().reset_index(name='count_w_c')
clicks_df.head()
buys_df['timestamp']=pd.to_datetime(buys_df.timestamp)

buys_df['hour']=buys_df.timestamp.dt.hour

buys_df['weekday']=buys_df['timestamp'].dt.dayofweek.astype(int)+1



buys_hour_info = buys_df.groupby(['hour'])['session_id'].nunique().reset_index(name='count_h_b')

buys_weekday_info = buys_df.groupby(['weekday'])['session_id'].nunique().reset_index(name='count_w_b')
fig = plt.figure(figsize=(15,6))

fig.suptitle('Number of Clicks for Time', fontsize=20)



ax1 = fig.add_subplot(121)

ax2 = fig.add_subplot(122)



#ax1.scatter(click_hour_info['hour'],click_hour_info['count_h_c'],color='b')

#ax1.scatter(buys_hour_info['hour'],buys_hour_info['count'],color='r')

ax1.bar(click_hour_info['hour'],click_hour_info['count_h_c'],width=0.5,color='b')

#ax1.bar(buys_hour_info['hour'],buys_hour_info['count'],width=0.2,color='r')





ax1.set_xlabel('$hour$', fontsize=17)

ax1.set_ylabel('$number of clicks$', fontsize=17)



#ax2.scatter(click_weekday_info['weekday'],click_weekday_info['count_w_c'],color='b')

#ax2.scatter(buys_weekday_info['weekday'],buys_weekday_info['count'],color='r')

ax2.bar(click_weekday_info['weekday'],click_weekday_info['count_w_c'],width=0.5,color='b')

#ax2.bar(buys_weekday_info['weekday'],buys_weekday_info['count'],width=0.03,color='r')



ax2.set_xlabel('$weekday$', fontsize=17)

ax2.set_ylabel('$number of clicks $', fontsize=17)

plt.show()
fig = plt.figure(figsize=(15,6))

fig.suptitle('Number of Clicks for Time', fontsize=20)



ax1 = fig.add_subplot(121)

ax2 = fig.add_subplot(122)



#ax1.scatter(click_hour_info['hour'],click_hour_info['count'],color='b')

#ax1.scatter(buys_hour_info['hour'],buys_hour_info['count_h_b'],color='r')

#ax1.bar(click_hour_info['hour'],click_hour_info['count'],width=0.2,color='b')

ax1.bar(buys_hour_info['hour'],buys_hour_info['count_h_b'],width=0.5,color='r')





ax1.set_xlabel('$hour$', fontsize=17)

ax1.set_ylabel('$number of buys$', fontsize=17)



#ax2.scatter(click_weekday_info['weekday'],click_weekday_info['count'],color='b')

#ax2.scatter(buys_weekday_info['weekday'],buys_weekday_info['count_w_b'],color='r')

#ax2.bar(click_weekday_info['weekday'],click_weekday_info['count'],width=0.03,color='b')

ax2.bar(buys_weekday_info['weekday'],buys_weekday_info['count_w_b'],width=0.5,color='r')



ax2.set_xlabel('$weekday$', fontsize=17)

ax2.set_ylabel('$number of buys $', fontsize=17)

plt.show()
# result_hours = pd.merge([click_hour_info,buys_hour_info], join = 'inner')

result_hours = pd.merge(click_hour_info,buys_hour_info,on='hour', how='inner')

# df = pd.merge(clicks_df, buys_df, on=['session_id','item_id'], how='left' , suffixes=('_click','_buy'))



result_hours['ratio'] = result_hours['count_h_b']/result_hours['count_h_c']



#result_weekdays = pd.concat([click_weekday_info,buys_weekday_info], axis = 1, join = 'inner')

result_weekdays = pd.merge(click_weekday_info,buys_weekday_info,on='weekday', how='inner')

result_weekdays['ratio'] = result_weekdays['count_w_b']/result_weekdays['count_w_c']
result_hours.head()

result_weekdays.head()
fig = plt.figure(figsize=(15,6))

fig.suptitle(' Buy Ratio Averaged for Time', fontsize=20)



ax1 = fig.add_subplot(121)

ax2 = fig.add_subplot(122)



#ax1.scatter(result_hours['hour'],result_hours['ratio'],color='b')

ax1.bar(result_hours['hour'],result_hours['ratio'],width=0.5,color='b')





ax1.set_xlabel('$hour$', fontsize=17)

ax1.set_ylabel('$buy/click ratio$', fontsize=17)



#ax2.scatter(result_weekdays['weekday'],result_weekdays['ratio'],color='b')

ax2.bar(result_weekdays['weekday'],result_weekdays['ratio'],width=0.5,color='b')

ax2.set_xlabel('$weekday$', fontsize=17)

ax2.set_ylabel('$buy/click ratio$', fontsize=17)

plt.show()