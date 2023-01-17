# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import time
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/us-counties-covid-19-dataset/us-counties.csv', parse_dates=['date'], index_col='date')
print(data.shape, data.columns)
data1 = data
data2 = data
data3 = data
data1.head()
data1.isnull().sum().sort_values()
data1.describe()
print(data1['county'].unique(), data1['county'].nunique())
print(data1['state'].unique(), data1['state'].nunique())
data1.groupby(['state', 'county']).describe()
data1.drop(['fips'], axis=1).groupby(['state', 'county']).sum()
data1.drop(['fips'], axis=1).groupby(['state']).sum()
data1.drop(['fips'], axis=1).groupby(['county']).sum()
print(data1['fips'].unique(), data1['fips'].nunique())
print(data1['county'].unique(), data1['county'].nunique())
data1[['county', 'fips']].groupby(['county']).nunique()
data1[data1['county'] == 'Adair']
data1[data1['county'] == 'Adair']['fips'].unique()
data1.sort_values(by=['state'], ascending=True).reset_index(drop=True)
data_states = data1.groupby(['state','county', 'date'])['deaths','cases'].apply(lambda x: x.sum())
data_states = data_states.reset_index()
data_states = data_states.sort_values(by='date', ascending=False)
data_states = data_states.reset_index(drop=True)
data_states
data_states_date = data_states.groupby(['state','date'])['deaths','cases'].apply(lambda x: x.sum()).reset_index()
data_states_date
data_states_total = data_states_date.groupby('state')['cases'].sum()
data_states_total = data_states_total.reset_index()
data_states_total = data_states_total.sort_values(by=['cases'], ascending=False)
data_states_total
import plotly.express as px
fig = px.line(data_states_date, x='date', y='deaths', color='state')
fig.update_layout(showlegend=False)
fig.show()
fig = px.line(data_states_date, x='date', y='cases', color='state')
fig.update_layout(showlegend=False)
fig.show()
cases_by_date = data_states = data.groupby('date')['cases'].apply(lambda x: x.sum())
cases_by_date = cases_by_date.reset_index()
cases_by_date['date'] = pd.to_datetime(cases_by_date['date'], infer_datetime_format=True)
cases_by_date
type(cases_by_date['date'][0])
indexed_cases_by_date = cases_by_date.set_index(['date'])
indexed_cases_by_date
indexed_cases_by_date
indexed_cases_by_date.index
indexed_cases_by_date['2020-01']
plt.xlabel('date')
plt.ylabel('cases')
plt.plot(indexed_cases_by_date)
