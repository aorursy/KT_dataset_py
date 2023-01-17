# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json
import matplotlib.pyplot as plt


from pandas.io.json import json_normalize

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
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
train_df = load_df()#nrows=20000)
test_df = load_df("../input/test.csv")#,nrows=20000)
train_df.head()
features_not_available = []
for a,b in zip(train_df.loc[0], train_df.columns):
    if a == 'not available in demo dataset':
        features_not_available.append(b)
train_df = train_df.drop(features_not_available, axis=1)
train_df.head()
channel_density = train_df.groupby(['channelGrouping']).channelGrouping.count().sort_values()
plt.xticks(rotation=60)
plt.xlabel('Channels')
plt.ylabel('Visits')
plt.title('Channel Density')
x_den = channel_density.index
y_den = channel_density
for a,b in zip(x_den, y_den):
    plt.text(a, b+60, str(round(100*b/train_df.shape[0]))+'%')
plt.bar(x_den, y_den)
train_df["totals.transactionRevenue"] = train_df["totals.transactionRevenue"].astype('float')
channel_revenue = train_df.groupby('channelGrouping')['totals.transactionRevenue'].sum().sort_values().reset_index()
plt.xticks(rotation=60)
plt.xlabel('Channels')
plt.ylabel('Revenue')
plt.title('Revenue per channel')
plt.bar(channel_revenue['channelGrouping'], channel_revenue['totals.transactionRevenue'])
device_browser = train_df.groupby('device.browser')['totals.transactionRevenue'].agg(['count', 'size', 'mean']).sort_values('count', ascending=False)[0:8]
device_browser.head()
plt.xticks(rotation=60)
plt.xlabel('Browser')
plt.ylabel('Count')
plt.title('Device browser count')
plt.bar(device_browser.index, device_browser['count'])
plt.xticks(rotation=60)
plt.xlabel('Browser')
plt.ylabel('Revenue')
plt.title('Device browser total revenue')
plt.bar(device_browser.index, device_browser['size'])
plt.xticks(rotation=60)
plt.xlabel('Browser')
plt.ylabel('Mean revenue')
plt.title('Device browser mean revenue')
plt.bar(device_browser.index, device_browser['mean'])