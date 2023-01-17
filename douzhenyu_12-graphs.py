# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas_profiling import ProfileReport



import matplotlib.pyplot as plt

from matplotlib import dates as md

import seaborn as sns

import plotly.graph_objs as go

import plotly

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import cufflinks as cf

cf.set_config_file(offline=True)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_meta = pd.read_csv('/kaggle/input/building-data-genome-project-v1/meta_open.csv')

df_meta
df_meta.pivot_table(index='newweatherfilename',columns='primaryspaceusage', values='uid', aggfunc='count').plot.bar(stacked=True, figsize=(15,5))
df_meta[df_meta['newweatherfilename']=='weather2.csv']
df_powerMeter = pd.read_csv('/kaggle/input/building-data-genome-project-v1/temp_open_utc_complete.csv', index_col='timestamp', parse_dates=True)

df_powerMeter.index = df_powerMeter.index.tz_localize(None)

df_powerMeter = df_powerMeter/df_meta.set_index('uid').loc[df_powerMeter.columns, 'sqm']

df_powerMeter
list_bldg_site2 = df_meta.loc[df_meta['newweatherfilename']=='weather2.csv', 'uid'].to_list()

list_bldg_site2
df_powerMeter_site2 =  df_powerMeter[list_bldg_site2].dropna(how='all')

df_powerMeter_site2
df_Office = df_powerMeter_site2.iloc[:, 0:17]

df_Office['Office'] = df_Office.apply(lambda x: x.mean(), axis=1)

df_Office = df_Office.iloc[:, 17:18]

df_Office
df_Class = df_powerMeter_site2.iloc[:, 17:36]

df_Class['Class'] = df_Class.apply(lambda x: x.mean(), axis=1)

df_Class = df_Class.iloc[:, 19:20]

df_Class
df_Dorm = df_powerMeter_site2.iloc[:, 36:63]

df_Dorm['Dorm'] = df_Dorm.apply(lambda x: x.mean(), axis=1)

df_Dorm = df_Dorm.iloc[:, 27:28]

df_Dorm
df_Lab = df_powerMeter_site2.iloc[:, 63:88]

df_Lab['Lab'] = df_Lab.apply(lambda x: x.mean(), axis=1)

df_Lab = df_Lab.iloc[:, 25:26]

df_Lab
df_site2 = pd.concat([df_Office,df_Class,df_Dorm,df_Lab],axis=1)

df_site2
df_weather2 = pd.read_csv('/kaggle/input/building-data-genome-project-v1/weather2.csv', index_col='timestamp', parse_dates=True)

df_weather2 = df_weather2.select_dtypes(['int', 'float'])



for col in df_weather2.columns:

    df_weather2.loc[df_weather2[col]<-100, col] = np.nan

    df_weather2[col] = df_weather2[col].interpolate()



df_weather2 = df_weather2.reset_index().drop_duplicates(subset=['timestamp'])



df_weather2 = df_weather2.set_index('timestamp').resample('1H').mean()



df_weather2
df_weather2.iplot()
df_site2_merged = df_site2.merge(df_weather2, left_index=True, right_index=True)

df_site2_merged
for col in df_site2_merged.columns[0:4]:

    df_temp = df_site2_merged.resample('D').mean()

    df_temp['weekday'] = df_temp.index.weekday

    sns.regplot(x="TemperatureC", y=col, data=df_temp[df_temp['weekday']>4], order=2)

    sns.regplot(x="TemperatureC", y=col, data=df_temp[df_temp['weekday']<=4], order=2)

    plt.legend(['weekend','weekday'],loc='lower left')

    plt.title(col+' energy consumption - Temp',size=20)

    plt.xlabel('Temp,℃',size=10)

    plt.ylabel('kW/m2',size=10)

    plt.show()
df_site2_merged['hour'] = df_site2_merged.index.hour

df_site2_merged['weekday'] = df_site2_merged.index.weekday

df_site2_merged['timeofweek'] = df_site2_merged.index.weekday*24 + df_site2_merged.index.hour

df_site2_merged
for col in df_site2_merged.columns[0:4]:

    sns.lineplot(data=df_site2_merged, x="timeofweek", y=col)

    plt.title(col+' energy consumption - timeofweek',size=20)

    plt.xlabel('time of week',size=10)

    plt.ylabel('kW/m2',size=10)

    plt.show()
df_schedule2 = pd.read_csv('/kaggle/input/building-data-genome-project-v1/schedule2.csv', header=None)

df_schedule2 = df_schedule2.rename(columns={0:'date',1:'date_type'})

df_schedule2['date'] = pd.to_datetime(df_schedule2['date'])

df_schedule2
df_schedule2['date_type'].value_counts()
df_site2_merged['date'] = pd.to_datetime(df_site2_merged.index.date)

df_site2_merged = df_site2_merged.merge(df_schedule2, on='date')

df_site2_merged
for col in df_site2_merged.columns[0:4]:

    sns.boxplot(x="date_type", y=col, data=df_site2_merged)

    plt.title(col+' energy consumption - datetype',size=20)

    plt.ylabel('kW/m2',size=10)

    plt.xlabel('Date-type',size=10)

    plt.show()