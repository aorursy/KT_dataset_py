import numpy as np

import pandas as pd

import datetime as dt

import matplotlib.pyplot as plt
data = pd.read_csv('../input/globalterrorismdb_0617dist.csv', encoding='ISO-8859-1', low_memory=False)

data.info()
data.head()
data.describe()
data[data['country_txt'] == 'India'].tail()
data['country_txt'].value_counts()[0:30].plot("bar",figsize=(12,6),title="Top 30 countries which faced terrorist events")
data[data['iyear'] > 1996]['country_txt'].value_counts()[0:30].plot("bar",figsize=(12,6),title="Top 30 countries faced terrorist events in last 10 years")
data[data['country_txt'] == 'United States'].tail()
data_part = data[['nkill', 'nkillus', 'nkillter', 'nwound', 'nwoundus', 'nwoundte','country_txt','location']]

data_part.fillna(0.0)

data_part.head()

recent_data = data[data['iyear'] > 1996]

recent_data_part = recent_data[['nkill', 'nkillus', 'nkillter', 'nwound', 'nwoundus', 'nwoundte','country_txt','location']]

recent_data_part.head()

d = recent_data_part.groupby('country_txt').sum()

nkill = d.reindex().sort_values(by='nkill',ascending=False)['nkill']

s=nkill[0:30].plot("bar",figsize=(12,6))

s.set_title("Top 30 countries in last 10 years",color='r',fontsize=30)

s.set_xlabel("Country",color='m',fontsize=20)

s.set_ylabel("No. of wounded/ killed people in last 10 years",color='m',fontsize=20)