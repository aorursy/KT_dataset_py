# Import libraries to use

# Common imports
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
#from preprocess import read_data, json_read

from datetime import datetime # To access datetime

import warnings                # To ignore the warning
warnings.filterwarnings("ignore")

# To plot pretty figures
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
import os # it's a operational system library, to set some informations

def read_data(file_path, file_name, data_format):
    """
    Parameters:
    -----------
    file_path: str 
               where the datafile is
    file_name: str
               file name of the datafile
    data_format: dict
                 format of data to import
                 
    Return
    ------
    df: dataframe
        the df after preprocessing
    """
    # Load the data
    df = pd.read_csv(file_path + file_name, dtype = data_format)
    
    # Printing the shape of dataframes that was imported     
    print("Loaded file at {}, and dataframe with shape {}".format(file_path + file_name, df.shape))
    
    return df


def json_read(df, field_name, extract_field, new_field_name):
    """
    Read semi-structured JSON data
    
    Parameters:
    ----------
    df: dataframe
        the dataframe needed to process
    field_name: str
                column to read json data
    extract_field: str 
                   info to extract from json data format
    new_field_name: str
                    add a column for data extract from json data format
                    
    Return
    ------
    df: dataframe
        the df after extracting data from json data format and adding to a new column, called new_field_name
    """
    df[new_field_name] = pd.DataFrame(df[field_name].apply(json.loads).tolist())[[extract_field]]
    return df
chunksize = 600000
for chunk in pd.read_csv('../input/train_v2.csv', chunksize=chunksize):
    df_train = chunk
    break
    
df_train.head()
# Check data type in each column
df_train.dtypes
df_train.describe()
# Check memory usage in MB
df_train.memory_usage(deep=True)* 1e-6
# Estimate total memory usage
usage = df_train.memory_usage(deep=True).sum() * 1e-6
print('Memory usage is {} Gb'.format(usage/1000))
chunksize = 100
for chunk in pd.read_csv('../input/test_v2.csv', chunksize=chunksize):
    df_test = chunk
    break

df_test.head()
del df_test
df_submit = pd.read_csv('../input/sample_submission_v2.csv')
df_submit.head()
del df_submit
df_train.loc[0,'geoNetwork']
df_train.loc[10,'device']
df_train.loc[101,'device']
df_train.loc[80,'totals']
df_train.loc[102,'trafficSource']
# Extract revenue from transactionRevenue from totals field
field_name = 'totals' 
extract_field = 'transactionRevenue'
new_field_name = 'revenue'

json_read(df_train, field_name, extract_field, new_field_name).head()
# Estimate total memory usage
usage = df_train.memory_usage(deep=True).sum() * 1e-6
print('Memory usage is {} Gb'.format(usage/1000))
# Check missing values in revenue field
df_train['revenue'].isnull().sum()
# Fill in missing data with zeros
df_train['revenue'] = df_train['revenue'].fillna(0)
df_train.head(7)
df_train['revenue'] = df_train['revenue'].astype('int64')
df_train.dtypes
df_train['Buy'] = df_train['revenue'].apply(lambda x: 1 if x != 0 else 0)
df_train.head()
# Estimate total memory usage
usage = df_train.memory_usage(deep=True).sum() * 1e-6
print('Memory usage is {} Gb'.format(usage/1000))
df_train.channelGrouping.value_counts().plot(kind="bar",title="channelGrouping distribution",figsize=(8,8),rot=25,colormap='Paired')
df_channel = df_train[['channelGrouping','revenue', 'Buy']]
(df_channel.set_index('channelGrouping').groupby(level=0)['revenue'].agg({'mean': np.average, 'median':np.median, 'std':np.std, 
                                                                          'max': np.max, 'min': np.min}) )
# Visualize the data by drawing boxplot grouped by a categorical variable:
sns.boxplot(x='revenue', y='channelGrouping', data=df_channel)
Channel_Buy = (df_channel.groupby(['channelGrouping', 'Buy'])['revenue'].agg({'Count':'count'}))
Channel_Buy['Relative Frequency'] = Channel_Buy.groupby(level=0).apply(lambda x: 100*x/float(x.sum()))
Channel_Buy
# boxplot for categorical variable (channelGrouping) vs. numerical variable (revenue)
df_channel_Buy = df_channel[df_channel['Buy'] == 1]
sns.boxplot(x='revenue', y='channelGrouping', data=df_channel_Buy)
# Transform revenue by log function
df_channel_Buy['log(revenue)'] = df_channel_Buy['revenue'].apply(np.log)
sns.boxplot(x='log(revenue)', y='channelGrouping', data=df_channel_Buy)
# Delete useless dataframe
del df_channel_Buy, Channel_Buy, df_channel
df_train['date'] = pd.to_datetime(df_train['date'],format="%Y%m%d") 
df_train.head()['date']
df_time = df_train[['date', 'revenue', 'Buy']]
df_time['year'] = pd.DatetimeIndex(df_time['date']).year
df_time['month'] = pd.DatetimeIndex(df_time['date']).month
df_time['day'] = pd.DatetimeIndex(df_time['date']).day
df_time.head()
df_time['DayOfWeek']=df_time['date'].dt.dayofweek
temp = df_time['date']
def applyer(row):
    if row.dayofweek == 5 or row.dayofweek == 6:
        return 1
    else:
        return 0

temp2 = df_time['date'].apply(applyer)
df_time['weekend']=temp2
df_time.index = df_time['date'] # indexing the Datetime to get the time period on the x-axis.
df_time.head()
df_time_buy = df_time[df_time['Buy'] == 1]
df_time_buy.head()
# using an html hex string for color
color = '#0099ff'
df_time.groupby(['year','month']).size().plot.bar(rot = 55, color=color)
plt.ylabel('Visit Count')
df_time.groupby(['year','month'])['Buy'].mean().plot.bar(color=color)
plt.ylabel('Purchasing Rate')
df_time.groupby('weekend')['Buy'].mean().plot.bar(color=color)
plt.ylabel('Purchasing Rate')
daily_transaction_per_visit_df = df_time_buy[['date','revenue']].groupby(by=['date'],axis=0).mean()
fig, axes = plt.subplots(figsize=(20,10))
axes.set_title("Daily Transaction per Visit")
axes.set_ylabel("Transaction per Visit")
axes.set_xlabel("date")
axes.plot(daily_transaction_per_visit_df["revenue"])
# Clean dataframe
del daily_transaction_per_visit_df, df_time_buy, df_time
df_train.iloc[10,2]
df_device = df_train[['device', 'revenue', 'Buy']]
# Extract deviceCategory from device field
field_name = 'device' 
extract_field = 'deviceCategory'
new_field_name = 'DeviceCategory'

json_read(df_device, field_name, extract_field, new_field_name).head()
# Check whether there's missing values in DeviceCategory column
pd.isna(df_device['DeviceCategory']).sum()
# using an html hex string for color
color = '#0099ff'
df_device.groupby(['DeviceCategory']).size().plot.bar(rot = 0, color=color)
plt.ylabel('Visit Count')
# Extract deviceCategory from device field
field_name = 'device' 
extract_field = 'isMobile'
new_field_name = 'IsMobile'

json_read(df_device, field_name, extract_field, new_field_name).head()
# Estimate total memory usage
usage = df_device.memory_usage(deep=True).sum() * 1e-6
print('Memory usage is {} Gb'.format(usage/1000))
# Check data type in IsMobile
df_device['IsMobile'].dtypes
df_device.groupby('IsMobile')['Buy'].mean().plot.bar(color=color, rot = 0)
plt.ylabel('Purchasing Rate')
df_device_Buy = df_device[df_device['Buy']==1]
df_device_Buy.head()
# Transform revenue by log function
df_device_Buy['log(revenue)'] = df_device_Buy['revenue'].apply(np.log)
sns.boxplot(x='IsMobile', y='log(revenue)', data=df_device_Buy)
del df_device_Buy, df_device
df_train.loc[21,'geoNetwork']
df_geo = df_train[['geoNetwork', 'revenue', 'Buy']]
df_geo.head()
# Estimate total memory usage
usage = df_geo.memory_usage(deep=True).sum() * 1e-6
print('Memory usage is {} Gb'.format(usage/1000))
# Extract continent from geoNetwork field
field_name = 'geoNetwork' 
extract_field = 'continent'
new_field_name = 'continent'

json_read(df_geo, field_name, extract_field, new_field_name).head()
df_geo.groupby('continent').size().plot.bar(color=color, rot = 0)
plt.ylabel('visit counts')
df_geo.groupby('continent')['Buy'].mean().plot.bar(color=color, rot = 0)
plt.ylabel('Purchasing Rate')
# Extract country from geoNetwork field
field_name = 'geoNetwork' 
extract_field = 'country'
new_field_name = 'country'

json_read(df_geo, field_name, extract_field, new_field_name).head()
df_americas = df_geo[df_geo['continent'] == 'Americas'] 
df_americas.groupby('country')['Buy'].mean().sort_values(ascending=False).head(10)
# How many visitors are from Anguilla?
len(df_americas[df_americas['country']=='St. Lucia'])
df_americas.groupby('country').size().sort_values(ascending=False).head(25)
df_geo_Buy = df_geo[df_geo['Buy']==1]
df_geo_Buy.head()
# Transform revenue by log function
df_geo_Buy['log(revenue)'] = df_geo_Buy['revenue'].apply(np.log)
sns.boxplot(x='continent', y='log(revenue)', data=df_geo_Buy)
len(df_geo_Buy[df_geo_Buy['continent']== 'Africa'])
del df_americas, df_geo, df_geo_Buy
# using an html hex string for color
color = '#0099ff'
df_train.groupby(['socialEngagementType']).size().plot.bar(rot = 0, color=color)
plt.ylabel('Visits')
df_train['socialEngagementType'].count()
df_train.loc[101,'trafficSource']
df_train.loc[102,'trafficSource']
df_train.loc[103,'trafficSource']
df_train.loc[80,'totals']
df_train.loc[1001,'totals']
df_train.loc[302,'totals']
df_totals = df_train[['totals', 'revenue', 'Buy']]
df_totals.head()
# View momory usage including objects
df_totals.info(memory_usage='deep')
# Extract visits from totals field
field_name = 'totals' 
extract_field = 'visits'
new_field_name = 'visits'

json_read(df_totals, field_name, extract_field, new_field_name).head()
df_totals.groupby('visits').size()
# Drop unuseful feature to release memory
df_totals.drop(columns='visits', inplace=True)
df_totals.head()
# Extract hits from totals field
field_name = 'totals' 
extract_field = 'hits'
new_field_name = 'hits'

json_read(df_totals, field_name, extract_field, new_field_name).head()
df_totals.groupby('hits').size().head()
# Extract pageviews from totals field
field_name = 'totals' 
extract_field = 'pageviews'
new_field_name = 'pageviews'

json_read(df_totals, field_name, extract_field, new_field_name).head()
df_totals.groupby('pageviews').size().head()
# Check data types for values in each column
df_totals.dtypes
# Convert desired columns to numeric type
df_totals[['hits', 'pageviews']] = df_totals[['hits', 'pageviews']].apply(pd.to_numeric) 
df_totals.dtypes
temp_df = df_totals[['hits','pageviews', 'Buy','revenue']]

# Calculate correlations
corr = temp_df.corr()

# Heatmap
sns.heatmap(corr)
