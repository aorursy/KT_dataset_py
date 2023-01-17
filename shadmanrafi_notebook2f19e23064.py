# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as py
import seaborn as sns
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
df = pd.read_csv('/kaggle/input/coronavirus-2019ncov/covid-19-all.csv')
df.head()
df['Date'] = pd.to_datetime(df['Date'])
df[['Confirmed','Recovered','Deaths']] = df[['Confirmed','Recovered','Deaths']].fillna(0).astype(int)
df['Still Infected'] = df['Confirmed'] - df['Recovered'] - df['Deaths']
df["Country/Region"].replace({"Mainland China": "China"}, inplace=True)
countries_affected = df['Country/Region'].unique().tolist()
print("\nTotal countries effected by Corona virus: ",len(countries_affected))
df.head()
df_list= df['Country/Region'].unique().tolist()
print ('[%s]' % ', '.join(map(str, df_list)))
del df['Province/State']
df.head()
aggregation_functions = {'Confirmed': 'sum', 'Recovered': 'sum', 'Deaths': 'sum', 'Still Infected': 'sum', }
df_new = df.groupby(['Country/Region', 'Date']).aggregate(aggregation_functions)
df_new.reset_index()
df_new.head()
del df['Latitude']
del df['Longitude']
df.head()
recent_date = df['Date'].max()
latest_entry = df[df['Date'] >= recent_date]
df_stats = df.groupby(['Date','Country/Region'])[['Confirmed','Recovered','Deaths','Still Infected']].sum().reset_index()
df_stats.head()
df_all_agg = df.drop('Date',1)
df_all_agg.head()
df_all_agg = df_all_agg.groupby(['Country/Region'])[['Confirmed','Recovered','Deaths','Still Infected']].sum().reset_index()
df_all_agg.head()
latest_entry.head()
latest_entry.shape
latest_entry = latest_entry.groupby(['Country/Region'])[['Confirmed','Recovered','Deaths','Still Infected']].sum().reset_index()
latest_entry.head()
latest_entry['Death/Case'] = (latest_entry['Deaths']*100)/latest_entry['Confirmed']
latest_entry.head()
latest_entry.shape
latest_entry['Recovery/Case'] = (latest_entry['Recovered']*100)/latest_entry['Confirmed']
latest_entry.head()
latest_entry['Mortality'] = (latest_entry['Deaths']*100)/latest_entry['Recovered']
latest_entry.head()
q1 = latest_entry['Confirmed'].quantile(0.25)
q3 = latest_entry['Confirmed'].quantile(0.75)
iqr = q3 - q1
fence_low = q1 - 1.5 * iqr
fence_high = q3 + 1.5 * iqr
cleaned_data = latest_entry.loc[(latest_entry['Confirmed'] > fence_low) & (latest_entry['Confirmed'] < fence_high)]
cleaned_data.head()
latest_entry = latest_entry.sort_values(["Still Infected", "Deaths", "Recovery/Case", "Mortality"], ascending = (True, True, False, True)).reset_index()

latest_entry.head()
latest_entry.tail()
latest_entry = latest_entry[latest_entry['Mortality'] > 0]
latest_entry.tail()
latest_entry.head()
latest_entry.shape
t = latest_entry[latest_entry['Confirmed'] > 1500]
t.head()
t.tail()