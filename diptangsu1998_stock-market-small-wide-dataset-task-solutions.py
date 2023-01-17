# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df1=pd.read_csv('../input/stock-market-small-wide-dataset/bar.csv')
df1.head()
df2=pd.read_csv('../input/stock-market-small-wide-dataset/quote.csv')
df2.head()
temp_df1=df1[df1['time'].str.lower().str.contains('2020-08-05')]
temp_df2=df2[df2['time'].str.lower().str.contains('2020-08-05')]
dataframe1=pd.merge(temp_df1, temp_df2, on='time')
dataframe1.info()
dataframe1=dataframe1.tail(30)
dataframe1
plt.figure()
df=dataframe1[['bid_price','ask_price','average_price']].plot(figsize=(14, 6), marker='o', grid=True, markersize=4)
plt.xlabel('Time')
plt.ylabel('Price')
df3=pd.read_csv('../input/stock-market-small-wide-dataset/rating.csv')
df3.head()
dataframe2=pd.merge(df3, df1, how='inner', on='symbol', left_on=None, right_on=None, left_index=False, right_index=False, sort=True, suffixes=('_x', '_y'), copy=True, indicator=False, validate=None)
dataframe2[['symbol','ratingBuy','ratingScaleMark','consensusStartDate','consensusEndDate','average_price']]
plt.figure(figsize=(40,30))
plt.subplots_adjust(hspace = 0.5)
plt.subplot(2,1,1)
sns.lineplot(data=dataframe2, x="consensusStartDate", y="average_price", marker='o', color='blue',linestyle=':')
plt.xticks(rotation=90)
plt.xlabel('consensusStartDate',fontsize=20)
plt.ylabel('average_price',fontsize=20)
plt.subplot(2,1,2)
sns.lineplot(data=dataframe2, x='consensusEndDate', y='average_price', marker='^', color='green', markersize=8, linestyle='dashed')
plt.xticks(rotation=90)
plt.xlabel('consensusEndDate', fontsize=20)
plt.ylabel('average_price', fontsize=20)
df4=pd.read_csv('../input/stock-market-small-wide-dataset/event.csv')
for index, row in df4.iterrows():
    print(row.symbol, len(df1[(df1.symbol == row.symbol)]))
print('Therefore, no data was found in bar.csv dataset for symbols in event.csv dataset')
df5=pd.read_csv('../input/stock-market-small-wide-dataset/target.csv')
df5.head()
dataframe3=df5[df5['updatedDate'].str.lower().str.contains('2020-08-31')]
dataframe4=dataframe3[['updatedDate','priceTargetAverage']]
dataframe5=dataframe4.rename(columns={'updatedDate':'reportDate'})
dataframe5.head()
dataframe6=pd.merge(dataframe5, df4, on='reportDate')
print(dataframe6)
dataframe7=df1[df1['time'].str.lower().str.contains('2020-08-31')]
temp_df1=dataframe7['time'].str.split(" ", n = 1, expand = True)
dataframe7['updatedDate']=temp_df1[0]
dataframe7['time2']=temp_df1[1]
dataframe7.head()
dataframe8=dataframe7[['updatedDate','average_price']]
print(dataframe8.head(),'\n', dataframe4.head())
dataframe9=pd.merge(dataframe8, dataframe4, on='updatedDate')
dataframe9.drop('priceTargetAverage', axis=1, inplace=True)
print(dataframe9)
dataframe9.plot(figsize=(22,8), color='cyan', grid=True)
plt.xlabel('Time', fontsize=15)
plt.ylabel('average_price', fontsize=15)
df6=pd.read_csv('../input/stock-market-small-wide-dataset/news.csv')
df6.columns
df1.columns
dataframe7["time"].head(2)
df6['datetime'].head(2)
dataframe10=dataframe7
dataframe10=dataframe10.rename(columns={'updatedDate':'date', 'time':'datetime', 'time2':'time'})
dataframe10.columns
dataframe10=dataframe10[['datetime','average_price','time']]
dataframe11=pd.merge(df6, dataframe10, on='datetime')
dataframe11.head()
dataframe11.info()
plt.figure()
dataframe11.plot(figsize=(20, 8), c='magenta')
plt.xlabel('Time', fontsize=14)
plt.ylabel('Average Price',fontsize=14)
plt.title('Average Price Distribution', fontsize=18)