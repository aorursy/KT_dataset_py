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
# take 

df = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv", parse_dates=['Date'])

df.head()
df['Country'].value_counts().reindex(['China', 'Mainland China'])  # There `China` 34 records but `Miainland China` has 801.
df.loc[df['Country'] == 'China', 'Date'].describe()  # eveny records where Country is China
df.loc[df['Country'] == 'Mainland China', 'Date'].describe()  # looks like the first day is China then it switch to Mainland China
series_county_backup = df['Country'].copy()

df['Country'].where(~(df['Country'] == 'China'), other="Mainland China", inplace=True,)  # verify

df['Country'].value_counts().reindex(['China', 'Mainland China'])  # all China gone
df.loc[df['Country'] == 'Mainland China', :].groupby('Province/State')['Date'].describe()  # There are several non-China get inside, like Taiwan, Hong Kong, Macau

# The reason is clear that the One China policy only enforce the first date then Taiwan, Hong Kong, Macau
df['Country'] = series_county_backup  # roll back

df['Country'].where(~((df['Country'] == 'China') & (~df['Province/State'].isin(['Taiwan', 'Hong Kong', 'Macau']))), other="Mainland China", inplace=True,)  # verify

df.loc[df['Country'] == 'Mainland China', :].groupby('Province/State')['Date'].describe()  # this time is right,
df.loc[df['Province/State'] == 'Taiwan', 'Country'].value_counts() 
df.loc[df['Province/State'] == 'Hong Kong', 'Country'].value_counts()  # Hong Kong become a country after the second day
# some fun in time series

series_china = df.loc[df['Country'] == 'Mainland China', :].groupby("Date")[['Confirmed', 'Deaths', 'Recovered']].apply(lambda x: x.sum())

series_china
series_china.plot(kind='line', grid=True, figsize=(16, 6))
series_china.resample('3D').mean().plot(kind='line', grid=True, figsize=(16, 6))  # resample by 3 days by mean
series_china['Confirmed'].pct_change().rename('daily confirmed change').plot(kind='line', grid=True, legend=True, figsize=(16, 6))
series_china.pct_change().ewm(com=0.5).mean().plot(kind='line', grid=True, figsize=(16, 6))  # emwa on pct change
series_china.resample('1D').mean().pct_change().rolling(3).mean().plot(kind='line', grid=True, figsize=(16, 6))  # resample by daily and ma(3) on pct change