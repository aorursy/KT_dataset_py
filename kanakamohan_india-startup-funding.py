import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

from datetime import datetime

import os

%matplotlib inline

df = pd.read_csv('../input/Startup-Funding-Data/startup_funding_v4.csv')
df.drop(['Unnamed: 0'], axis=1, inplace=True)
df.Date = pd.to_datetime(df.Date, infer_datetime_format=True)
plt.xlabel('Year')

plt.ylabel('count')

plt.title('Startups Funded Per Year')

df.Date.dt.year.value_counts().sort_index().plot(kind='bar', figsize=(8,6), fontsize=14)
ser = df.Date.groupby([df.Date.dt.year, df.Date.dt.month]).count().unstack(level=-1)

ser.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

ser.plot(kind='bar', stacked=True, figsize=(12,10), legend='reverse');
df['Industry'].value_counts()[:3].append(df['Industry'].value_counts()[4:10]).plot(kind='pie', figsize=(8,8))
City_Count = df['City'].value_counts()

City_Count[:4].append(City_Count[5:16]).sort_values(ascending=True).plot(kind='barh', fontsize=14, figsize=(12, 8))