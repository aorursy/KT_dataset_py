import pandas as pd
import numpy as np
import time
from datetime import datetime
#import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import os
import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize

def load_df(csv_path='../input/train.csv', nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    
    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'}, # Important!!
                     nrows=nrows)
    
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
    return df

#print(os.listdir("input"))
%%time
train_df = load_df()
train_df.columns
# Let's check quickly how the data looks like after the conversion

train_df.head()
#Temporarily take the dataframe into another variable and this sd variable is the one we will use it throughout the notebook 

sd = train_df
#let's check the info on all columns, how many null and not null values are and their associated data types
sd.info()
sd.sessionId.nunique()
#removing duploicate session Id from the dataframe
df = sd[sd.duplicated(subset='sessionId')]
sd.drop_duplicates(subset='sessionId', keep="first", inplace=True)
#new session id count after removing all duplicates
sd.sessionId.count()
# Let's see how first record looks like
sd.loc[0]
cols =     ['device.browserSize', 'device.browserVersion', 'device.language', 'device.mobileDeviceBranding',  'device.mobileDeviceInfo', 
            'device.mobileDeviceMarketingName', 'device.mobileDeviceModel', 'device.mobileInputSelector', 'device.operatingSystemVersion',
            'device.screenColors', 'device.screenResolution', 'device.flashVersion','geoNetwork.cityId', 'geoNetwork.latitude', 'geoNetwork.longitude','geoNetwork.networkLocation',
           'trafficSource.campaignCode']
# Drop unwanted columns from dataframe which does not have useful information (null values etc)
sd.drop(columns=cols, axis=1, inplace=True)
#Also there are many columns which have numeric values and currently they are Object, so let's convert them into numberic values
sd[[ 
       'trafficSource.adwordsClickInfo.page',
         'totals.bounces', 'totals.hits',
       'totals.newVisits', 'totals.pageviews', 'totals.transactionRevenue', 'totals.visits']] = sd[[ 
       'trafficSource.adwordsClickInfo.page',
         'totals.bounces', 'totals.hits',
       'totals.newVisits', 'totals.pageviews', 'totals.transactionRevenue', 'totals.visits']].apply(pd.to_numeric)
# If you see the transaction amount, they are too high value
sd['totals.transactionRevenue'].max()
#fixing the transactionRevenue column data in dataframe
sd['totals.transactionRevenue'] = sd['totals.transactionRevenue']/1000000
# Some useful stats around transactionRevenue count and transactionRevenue amount data

print("Total transactions count over 0 dollars " +  sd[sd['totals.transactionRevenue'] > 0]['totals.transactionRevenue'].count().astype(str))
print("Total transactions amount over 0 dollars " +  sd[sd['totals.transactionRevenue'] > 0]['totals.transactionRevenue'].sum().astype(str))
print("Minimum transactions amount over 0 dollars " +  sd[sd['totals.transactionRevenue'] > 0]['totals.transactionRevenue'].min().astype(str))
print("Maximum transactions amount over 0 dollars " +  sd[sd['totals.transactionRevenue'] > 0]['totals.transactionRevenue'].max().astype(str))
print("Total transactions count over 1000 dollars " +  sd[sd['totals.transactionRevenue'] > 1000]['totals.transactionRevenue'].count().astype(str))
print("Total transactions amoount over 1000 dollars " +  sd[sd['totals.transactionRevenue'] > 1000]['totals.transactionRevenue'].sum().astype(str))
sd[(sd['totals.transactionRevenue'] > 0) & (sd['totals.transactionRevenue'] < 1000)]['totals.transactionRevenue'].plot(kind='hist', bins=120, sort_columns=False, figsize=[14,7])
sd['date'] = pd.to_datetime(sd['date'].astype(str), format='%Y%m%d')
# Showing a line graph with transaction revenue generated per day for the entire dataset between August 2017 - August 2018
# It's clear that during the month of Oct-Nov-Dec 2017 the overall mean level of the transactions were higher than other months

sd.groupby(['date'])['totals.transactionRevenue'].sum().plot(kind='line', sort_columns=False, figsize=[14,7], title='Daily Revenue (USD)', label='Transaction Revenue $ Amount', markevery=500, legend=True)
# Below graph shows # of transactions done by customers. It's visible that during Christmas time there were most number of transactions
# and again during May 2018... could be because schools are off and people do shopping a lot before summer vacations ?? :) 
sd.groupby(['date'])['totals.transactionRevenue'].count().plot(kind='line', sort_columns=False, figsize=[14,7], title='Daily number of session', label='Session count', markevery=100, legend=True)
# Let's add a weekday name like sunday, monday, tuesday....saturday for further analysis
sd['day_of_week'] = sd['date'].dt.weekday_name
# Let's add day number for sorting purpose
sd['day_number'] = sd['date'].dt.weekday
# Let's see which day of the week store get's most revenue in a year. It seems Tuesday is the winner. Saturday/Sunday people dont shop much online it seems 

ax = sd.groupby(['day_number'])['totals.transactionRevenue'].sum().plot(kind='bar', sort_columns=False, figsize=[14,7], title='Revenue per weekday (USD)', label='Transaction Revenue $ Amount',legend=True)
ax.set_xticklabels(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])    
ax.set_ylabel('Revenue(USD)')
ax.set_xlabel('Day of the week')
# Most number of transactions are done on Monday and very low transaction rate is found on Saturday and Sunday

ax = sd.groupby(['day_number'])['totals.transactionRevenue'].count().plot(kind='bar', sort_columns=False, figsize=[14,7], title='Number of Session on weekday', label='Number of sessions',legend=True)
ax.set_xticklabels(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])    
ax.set_ylabel('Number of Sessions')
ax.set_xlabel('Day of the week')
# Now which month of the year get's most revenue... it turns out that April, August and December are the months
# April = Summer vacation
# August = School about to start
# December = Christmas Vacations
ax = sd.groupby([sd['date'].dt.month])['totals.transactionRevenue'].sum().plot(kind='bar', width=.9, sort_columns=False, figsize=[14,7], title='Revenue per month', label='Revenue',legend=True)
ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])    
ax.set_ylabel('Revenue per month (USD)')
ax.set_xlabel('Months')
# most number of transactions are done in December

ax = sd.groupby([sd['date'].dt.month])['totals.transactionRevenue'].count().plot(kind='bar', width=.9, sort_columns=False, figsize=[14,7], title='Number of Session per month', label='Number of sessions',legend=True)
ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])    
ax.set_ylabel('Number of Sessions per month')
ax.set_xlabel('Months')
# Transaction revenue per channel grouping

ax = sd.groupby([sd['channelGrouping']])['totals.transactionRevenue'].sum().plot(kind='bar', width=.9, sort_columns=True, figsize=[14,7], title='Revenue per Channel', label='Revenue',legend=True)
ax.set_ylabel('Revenue (USD)')
ax.set_xlabel('Channel Name')
# Most number of sessions per channel grouping

ax = sd['channelGrouping'].value_counts().plot(kind='bar', width=.9, sort_columns=True, figsize=[14,7], title='Number of Sessions per Channel', label='Number of sessions',legend=True)
ax.set_ylabel('Number of Sessions')
ax.set_xlabel('Channel Name')
# revenue generated by source
sd.groupby([sd['trafficSource.source']])['totals.transactionRevenue'].sum().sort_values(ascending=False)[:20]
#Similarly plot for the revenue generated by source

ax = sd.groupby([sd['trafficSource.source']])['totals.transactionRevenue'].sum().sort_values(ascending=False)[:20].sort_values().plot(kind='barh', width=.9, sort_columns=True, figsize=[14,7], title='Revenue per Source', label='Revenue',legend=True)
ax.set_ylabel('Source Name')
ax.set_xlabel('Revenue (USD)')
ax = sd['trafficSource.source'].value_counts()[:20].sort_values().plot(kind='barh', width=.9, sort_columns=False, figsize=[14,7], title='Number of Sessions per Source', label='Number of sessions',legend=True)
ax.set_ylabel('Source Name')
ax.set_xlabel('Number of Sessions')
ax = sd.groupby([sd['device.deviceCategory']])['totals.transactionRevenue'].sum().sort_values(ascending=False)[:20].plot(kind='bar', width=.9, sort_columns=True, figsize=[14,7], title='Revenue per Device Category', label='Revenue',legend=True)
ax.set_xlabel('Device Category')
ax.set_ylabel('Revenue (USD)')
ax = sd['device.deviceCategory'].value_counts()[:20].plot(kind='bar', width=.9, sort_columns=False, figsize=[14,7], title='Number of Sessions per Device Category', label='Number of sessions',legend=True)
ax.set_xlabel('Device Category')
ax.set_ylabel('Number of Sessions')
ax = sd.groupby([sd['device.operatingSystem']])['totals.transactionRevenue'].sum().sort_values(ascending=False)[:7].plot(kind='bar', width=.9, sort_columns=True, figsize=[14,7], title='Revenue per Operating System', label='Revenue',legend=True)
ax.set_xlabel('Operating System')
ax.set_ylabel('Revenue (USD)')
ax = sd['device.operatingSystem'].value_counts()[:7].plot(kind='bar', width=.9, sort_columns=False, figsize=[14,7], title='Number of Sessions per Operating System', label='Number of sessions',legend=True)
ax.set_xlabel('Operating System')
ax.set_ylabel('Number of Sessions')
ax = sd.groupby([sd['device.browser']])['totals.transactionRevenue'].sum().sort_values(ascending=False)[:9].plot(kind='bar', width=.9, sort_columns=True, figsize=[14,7], title='Revenue per Browser', label='Revenue',legend=True)
ax.set_xlabel('Browser')
ax.set_ylabel('Revenue (USD)')
ax = sd['device.browser'].value_counts()[:9].plot(kind='bar', width=.9, sort_columns=False, figsize=[14,7], title='Number of Sessions per Browser', label='Number of sessions',legend=True)
ax.set_xlabel('Browser')
ax.set_ylabel('Number of Sessions')
ax = sd.groupby([sd['totals.pageviews']])['totals.transactionRevenue'].sum()[:75].plot(kind='bar', width=.9, sort_columns=True, figsize=[14,7], title='Revenue Pageviews', label='Revenue',legend=True)
ax.set_xlabel('Pageviews')
ax.set_ylabel('Revenue (USD)')
ax = sd['totals.pageviews'].value_counts()[:30].plot(kind='bar', width=.9, sort_columns=False, figsize=[14,7], title='Number of Sessions per pageviews', label='Number of sessions',legend=True)
ax.set_xlabel('Pageviews')
ax.set_ylabel('Number of Sessions')
ax = sd[(sd['totals.pageviews'] <= 100) & (sd['totals.transactionRevenue'] > 0)]['totals.pageviews'].value_counts().sort_values(ascending=False)[:50].plot(kind='bar', width=.9, sort_columns=True, figsize=[14,7], title='Number of Sessions per pageviews', label='Number of sessions',legend=True)
ax.set_xlabel('Pageviews')
ax.set_ylabel('Number of Sessions with transaction revenue')
print('Total number of bounces ' + sd[(sd['totals.bounces'] == 1)].count().iloc[1].astype(str))
print('Total revenue (USD) generated from bounces ' + sd[(sd['totals.bounces'] == 1) & (sd['totals.transactionRevenue'] > 0)]['totals.transactionRevenue'].sum().astype(str))
print('Total number of single pageviews ' + sd[(sd['totals.pageviews'] == 1)].count().iloc[1].astype(str))
print('Total revenue (USD) generated from single pageviews ' + sd[(sd['totals.pageviews'] == 1) & (sd['totals.transactionRevenue'] > 0)]['totals.transactionRevenue'].sum().astype(str))
ax = sd.groupby([sd[(sd['totals.hits'] <200 )]['totals.hits']])['totals.transactionRevenue'].sum()[:100].plot(kind='bar', width=.9, sort_columns=True, figsize=[14,7], title='Revenue page hits', label='Revenue',legend=True)
ax.set_xlabel('Pagehits')
ax.set_ylabel('Revenue (USD)')
ax.set_xmargin(.5)
print(sd.groupby([sd[(sd['totals.transactionRevenue'] > 1 )]['geoNetwork.country']])['totals.transactionRevenue'].sum().sort_values(ascending=False)[:10])
ax = sd.groupby([sd[(sd['totals.transactionRevenue'] > 1 )]['geoNetwork.country']])['totals.transactionRevenue'].sum().sort_values(ascending=False)[:10].plot(kind='bar', width=.9, sort_columns=True, figsize=[14,7], title='Revenue per country', label='Revenue',legend=True)
ax.set_xlabel('Country')
ax.set_ylabel('Revenue (USD)')
ax.set_xmargin(.5)
print('All visits by Country  - top 10')
print(sd.groupby(['geoNetwork.country'])['totals.visits'].sum().sort_values(ascending=False)[:10])
print('\nAll Revenue making visits by Country - top 10')
print(sd.groupby([sd[(sd['totals.transactionRevenue'] > 1 )]['geoNetwork.country']])['totals.visits'].sum().sort_values(ascending=False)[:10])

sd.groupby(['geoNetwork.country'])['totals.visits'].sum().sort_values(ascending=False)[:10].plot(kind='line', sort_columns=True, figsize=[14,7], label='Total Visits (right)',legend=True, subplots=True, sharey=False, secondary_y=True, colormap='rainbow')
ax = sd.groupby([sd[(sd['totals.transactionRevenue'] > 1 )]['geoNetwork.country']])['totals.visits'].sum().sort_values(ascending=False)[:10].plot(kind='bar', width=.9, sort_columns=True, figsize=[14,7], title='Revenue per country', label='Revenue making visits (left)',legend=True)
ax.set_xlabel('Country')
ax.set_ylabel('Revenue (USD)')
ax.set_xmargin(.5)
print('\nMean Transaction Revenue by Country where Transaction Revenue is > $0 - top 10')
print(sd.groupby([sd[(sd['totals.transactionRevenue'] > 0 )]['geoNetwork.country']])['totals.transactionRevenue'].mean().sort_values(ascending=False)[:10])

ax = sd.groupby([sd[(sd['totals.transactionRevenue'] > 0 )]['geoNetwork.country']])['totals.transactionRevenue'].mean().sort_values(ascending=False)[:10].plot(kind='bar', width=.9, sort_columns=True, figsize=[14,7], title='Mean Revenue per country with transaction amount > $0', label='Mean Revenue',legend=True)
ax.set_xlabel('Country')
ax.set_ylabel('Revenue (USD)')
ax.set_xmargin(.5)

