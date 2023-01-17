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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
df_all_data = pd.read_csv("../input/2019-coronavirus-dataset-01212020-01262020/2019_nCoV_20200121_20200131.csv")

df_all_data.head()
df_all_data.shape
df_all_data.isnull().sum()
df_all_data.info()
# Rename Countries

df_all_data.replace('Mainland China', 'China', inplace=True)

df_all_data.replace('Hong Kong', 'China', inplace=True)

df_all_data.replace('Macau', 'China', inplace=True)
# Handle NaNs

df_all_data['Province/State'].fillna('Unknown', inplace=True)

df_all_data['Suspected'].fillna(0, inplace=True)

df_all_data['Recovered'].fillna(0, inplace=True)

df_all_data['Death'].fillna(0, inplace=True)
df_all_data['Last Update'].unique()
df_all_data['Last Update'].max()
df_latest_data = df_all_data[df_all_data['Last Update'] == df_all_data['Last Update'].max()]
df_latest_data.shape
df_latest_data.info()
df_latest_data['Country/Region'].unique()
df_latest_data.info()
def convert_last_update_date_format(x):

    return pd.to_datetime(x.split(' ')[0])
df_all_data['Last Update'] = df_all_data['Last Update'].apply(convert_last_update_date_format)
df_all_data_by_date = df_all_data.groupby(['Last Update']).sum().reset_index()

df_all_data_by_date
df_latest_data_by_country = df_latest_data.groupby(['Country/Region']).agg('sum').reset_index()

df_latest_data_by_country
df_latest_data_by_state = df_latest_data.groupby(['Province/State']).agg('sum').reset_index()

df_latest_data_by_state
plt.figure(figsize=(25,15))

sns.barplot(x='Country/Region', y='Confirmed', data = df_latest_data_by_country)

plt.title('Country wise confirmed cases')
plt.figure(figsize=(30,20))

sns.barplot(x='Death', y='Province/State', data = df_latest_data_by_state)

plt.title('State-wise death rates')
plt.figure(figsize = (10,10))

pie = plt.pie(df_latest_data_by_state['Confirmed'], shadow=True)

plt.legend(df_latest_data_by_state['Province/State'], loc = 0)

plt.title('State-wise confirmed cases')
plt.figure(figsize=(20,10))

plt.plot( 'Last Update', 'Confirmed', data=df_all_data_by_date, marker='o', markerfacecolor='white', markersize=12, color='green', linewidth=4)

plt.plot( 'Last Update', 'Suspected', data=df_all_data_by_date, marker='o', markerfacecolor='white', markersize=11, color='orange', linewidth=3)

plt.plot( 'Last Update', 'Recovered', data=df_all_data_by_date, marker='o', markerfacecolor='white', markersize=9, color='blue', linewidth=2)

plt.plot( 'Last Update', 'Death', data=df_all_data_by_date, marker='o', markerfacecolor='white', markersize=7, color='red', linewidth=1)

plt.title('Conrona Virus Trends')

plt.xlabel('Last Updated Date')

plt.ylabel('Count')

plt.show()
fig= df_all_data.groupby(['Last Update','Country/Region'])['Confirmed'].sum().unstack().plot(figsize=(20,15),linewidth=2)

fig.set_title('Conrona Virus Country wise Confirmed Cases')

fig.set_xlabel('Last Updated Date')

fig.set_ylabel('Count')