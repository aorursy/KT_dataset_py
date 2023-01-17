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
# Getting data and importing libraries
import matplotlib.pyplot as plt
from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()

(market_df, news_df) = env.get_training_data()
market_df.info()
market_df.head()
market_df.describe()
# Market df shape
print('Number of records:' ,market_df.shape[0])
print('Number of features:' ,market_df.shape[1])

# Check number of assetes - market df
n_assets = len(market_df.assetName.unique().categories)
print('Number of assets:', n_assets)
news_df.info()
news_df.head()
news_df.describe()
# News df shape
print('Number of records:' ,news_df.shape[0])
print('Number of features:' ,news_df.shape[1])

n_assets = len(news_df.assetName.unique().categories)
print("Total news about %d assets" % (n_assets))
days = env.get_prediction_days()
(market_test_df, news_test_df, predictions_template_df) = next(days)
print('Done!')
predictions_template_df.head()
# Check for mssing values/nan values - market df
rows_with_nan_values = market_df.isnull().any(axis = 1)
n_rows_with_nan_values = sum(rows_with_nan_values)

print('Number of rows with nan values:', n_rows_with_nan_values)
print('Number toal rows:', market_df.shape[0])

# Remove nan values
market_new_df = market_df.dropna(axis = 0)
print('New dimensions:', market_new_df.shape)
# Check for missing values/nan values - news df
rows_with_nan_values = news_df.isnull().any(axis = 1)
n_rows_with_nan_values = sum(rows_with_nan_values)

print('Number of rows with nan values:', n_rows_with_nan_values)
print('Number toal rows:', news_df.shape[0])
# There are no nan values
# Visualization of random asset

# Get any asset
asset_sample = market_df.assetCode.sample(1, random_state = 10).iloc[0]
asset_market = market_df[market_df['assetCode'] == asset_sample]
asset_market.index = asset_market.time

# Plotting close price 
fig,axes = plt.subplots(2,1,figsize=(15,5))
axes[0].set_ylabel('Close price', fontsize=18)
axes[1].set_ylabel('Volume', fontsize=18)
axes[1].set_xlabel('Date', fontsize=18)
axes[0].set_title(asset_sample, fontsize=18)
axes[0].plot(asset_market.index, asset_market['close'])
axes[1].plot(asset_market.index, asset_market['volume'])

# Define close to open relation
close_to_open_yield = market_new_df['close']/market_new_df['open']
market_new_df['close_to_open_ratio'] = close_to_open_yield

min_val = np.min(close_to_open_yield)
max_val = np.max(close_to_open_yield)
print('values range:', '[', min_val, max_val ,']')

# Seems that we have some outliers - lets check it
print((close_to_open_yield > 1.2).sum())
print((close_to_open_yield < 0.8).sum())

plt.hist(close_to_open_yield, bins = 100, range = [0.8, 1.2])
plt.ylabel('Count', fontsize =16)
plt.xlabel('Close to open yield', fontsize = 16)
# Define outliers - market df
rows_to_remove_above  = close_to_open_yield > 1.2
rows_to_remove_under  = close_to_open_yield < 0.8
rows_to_remove = rows_to_remove_above | rows_to_remove_under

# Remove outliers
ind_to_remove = np.where(rows_to_remove)[0]
market_new_df = market_new_df.drop(market_new_df.index[ind_to_remove])
market_new_df.returnsOpenNextMktres10.plot(figsize=(10,5))
plt.ylabel('returnsOpenNextMktres10', fontsize = 14)
plt.xlabel('Observation index', fontsize = 14)
plt.show()
plt.hist(market_new_df['returnsOpenNextMktres10'], bins =1000, range = [-0.5, 0.5])
plt.title('Label distribution', fontsize =16)
plt.xlabel('returnsOpenNextMktres10', fontsize =14)
plt.ylabel('Count', fontsize =14)
time_series_df = market_new_df[["time"]].groupby(by=["time"]).size()
time_series_df.index = pd.to_datetime(time_series_df.index)

#Plot market trends
fig,axes = plt.subplots(1,1,figsize=(10,5))
axes.set_ylabel('Number of records', fontsize = 14)
axes.set_xlabel('Date', fontsize = 14)
axes.plot(time_series_df)
# Using only daily data
market_new_df['time'] = pd.to_datetime(market_new_df.time.astype('datetime64').dt.date, utc=True)
news_df['time'] = pd.to_datetime(news_df.time.astype('datetime64').dt.date, utc=True)

# Define times features
def addTimesFeatures(df):
    df['year'] = pd.to_datetime(df['time']).dt.year
    df['month'] = pd.to_datetime(df['time']).dt.month
    df['day'] = pd.to_datetime(df['time']).dt.day
    df['day_of_week'] = pd.to_datetime(df['time']).dt.dayofweek
    return df

market_new_df = addTimesFeatures(market_new_df)
news_df.head(3)
from sklearn.model_selection import train_test_split

# Divide data set into training and validation sets
train_indices, val_indices = train_test_split(market_new_df.index.values,test_size = 0.25, shuffle=False, random_state = 10)
market_train = market_new_df.loc[train_indices]
market_val = market_new_df.loc[val_indices]

time_training = market_train.time[train_indices]
asset_code_training = market_train.assetCode[train_indices]

# We want to get the news that only related to the training market samples
news_df.sort_values(by = 'time')
initial_time = time_training[0]
end_time = time_training[-1]

market_train = market_train.set_index(['time', 'assetCode'], drop=False)
news_df = news_df.set_index(['time', 'assetCodes'],  drop=False)

# X = market_train.merge(news_df, how='left', on=['time', 'assetCode'], left_index=True)
# initial_ind = news_df[news_df['time'] == initial_time].index.tolist()[0]
# end_ind = news_df[news_df['time'] == end_time].index.tolist()[-1]

# news_train_ind = np.linspace(initial_ind,end_ind,num = end_ind - initial_ind)
# print(news_train_ind)
# news_train = news_df[news_train_ind, ]
# news_train.head()
# market_train[['time','assetCode']] 

# Get numerical features names - market df
market_numeric_cols = []
for col in market_train.columns:
    if col not in ['universe', 'time', 'assetCode', 'assetName']: 
        market_numeric_cols.append(col)

print(market_numeric_cols)
news_cols_numeric = ['urgency', 'takeSequence', 'wordCount', 'sentenceCount', 'companyCount',
                     'marketCommentary', 'relevance', 'sentimentNegative', 'sentimentNeutral',
                     'sentimentPositive', 'sentimentWordCount', 'noveltyCount12H', 'noveltyCount24H',
                     'noveltyCount3D', 'noveltyCount5D', 'noveltyCount7D', 'volumeCounts12H',
                     'volumeCounts24H', 'volumeCounts3D', 'volumeCounts5D', 'volumeCounts7D']

# Scale numeric features
from sklearn.preprocessing import StandardScaler

# Standardization - market df   
scaler = StandardScaler()
market_train[market_numeric_cols] = scaler.fit_transform(market_train[market_numeric_cols])





