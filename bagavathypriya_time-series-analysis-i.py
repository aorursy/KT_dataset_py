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
!pip install download
from __future__ import absolute_import,division,print_function,unicode_literals

import matplotlib as mpl
import os
from datetime import datetime

import pandas as pd

from download import download

mpl.rcParams['figure.figsize'] = (8,6)
mpl.rcParams['axes.grid'] = False

path = download('https://archive.ics.uci.edu/ml/machine-learning-databases/00501/PRSA2017_Data_20130301-20170228.zip','/kaggle/temp',replace=True, kind="zip")

df = pd.read_csv('/kaggle/temp/PRSA_Data_20130301-20170228/PRSA_Data_Dingling_20130301-20170228.csv', encoding='ISO-8859-1')
df.head()
df.info()
def conv_to_date(x):
    return datetime.strptime(x,"%Y %m %d %H")
a_df=pd.read_csv("/kaggle/temp/PRSA_Data_20130301-20170228/PRSA_Data_Dingling_20130301-20170228.csv",
                 parse_dates=[['year','month','day','hour']],date_parser=conv_to_date,keep_date_col=True)
a_df.head()
a_df.info()
a_df['month']=pd.to_numeric(a_df['month'])
a_df.shape
a_df.columns
a_df.isnull().any()
print('Unique values:',a_df.nunique())
a_df.describe()
df_nonindexed=a_df.copy()
a_df=a_df.set_index('year_month_day_hour')
a_df.index
a_df.head()
a_df.loc['2013-03-01':'2013-03-05']
a_df.loc['2013':'2014']
pm_data=a_df['PM2.5']
pm_data.head()
pm_data.plot(grid=True)
a_df_15=a_df.loc['2015']
pm_15=a_df_15['PM2.5']
pm_15.plot(grid=True)
a_df_15=a_df.loc['2016']
pm_15=a_df_15['PM2.5']
pm_15.plot(grid=True)
import plotly.express as px
fig = px.line(df_nonindexed,x='year_month_day_hour',y='PM2.5',title='PM 2.5 with slider')

fig.update_xaxes(rangeslider_visible=True)
fig.show()
fig = px.line(df_nonindexed,x='year_month_day_hour',y='PM2.5',title='PM 2.5 with slider')

fig.update_xaxes(rangeslider_visible=True,rangeselector=dict(
                        buttons = list([
                        dict(count = 1,label = '1y',step='year',stepmode = "backward"),
                        dict(count = 2,label = '2y',step='year',stepmode = "backward"),
                        dict(count = 3,label = '3y',step='year',stepmode = "backward"),
                        dict(step= 'all')
                            ])        
                        ))
fig.show()
df_14=a_df['2014'].reset_index()
df_15=a_df['2015'].reset_index()

df_14['month_day_hour']=df_14.apply(lambda x: str(x['month'])+"."+x['day'],axis=1)
df_15['month_day_hour']=df_15.apply(lambda x: str(x['month'])+"."+x['day'],axis=1)

plt.plot(df_14['month_day_hour'],df_14['PM2.5'])
plt.plot(df_15['month_day_hour'],df_15['PM2.5'])

plt.legend(['2014','2015'])
plt.xlabel('Month')
plt.ylabel('PM2.5')

plt.title('Air qulaity 2014 and 2015')
df_14.head()
a_df['2014':'2016'][['month','PM2.5']].groupby('month').describe()
a_df['2014':'2016'][['month','PM2.5','TEMP']].groupby('month').agg({'PM2.5':['min','max'],'TEMP':['min','max']})
df_15=a_df['2015']
a_df_15=df_15[['PM2.5','TEMP']]
a_df_15.plot(subplots=True)
a_df[['PM2.5','TEMP']].hist()
a_df['TEMP'].plot(kind='density')
pd.plotting.lag_plot(a_df['TEMP'],lag=1)
pd.plotting.lag_plot(a_df['TEMP'],lag=10)
pd.plotting.lag_plot(a_df['TEMP'],lag=24)
pd.plotting.lag_plot(a_df['TEMP'],lag=8640)
pd.plotting.lag_plot(a_df['TEMP'],lag=4320)
pd.plotting.lag_plot(a_df['TEMP'],lag=2150)
a_df_15 = a_df['2015']
pm_data_2015 = a_df_15[['PM2.5','TEMP','PRES']]
pm_data_2015.plot(subplots = True)
multi_data = a_df[['PM2.5','TEMP','PRES','RAIN','DEWP']]
multi_data.plot(subplots = True)
multi_data = a_df[['PM2.5','SO2','NO2','O3','CO']]
multi_data.plot(subplots = True)
a_df['2014':'2015'][['PM2.5','O3']].plot(figsize=(15,8),linewidth= 3,fontsize = 15)
plt.xlabel('year_month_day_hour')
g = sns.pairplot(a_df[['PM2.5','SO2','NO2','O3','CO']])
aq_corr = a_df[['PM2.5','SO2','NO2','O3','CO']].corr(method = 'pearson')
aq_corr
import seaborn as sns

sns.heatmap(aq_corr)
a_df.groupby('wd').agg(median=('PM2.5','median'),mean=('PM2.5','mean'),max=('PM2.5','max'),min=('PM2.5','min')).reset_index()
a_dna= a_df.copy()

a_dna=a_dna.dropna()
pd.plotting.autocorrelation_plot(a_dna['2014':'2016']['TEMP'])
a_dna['TEMP'].resample('1m').mean() # as the data is hourly basis, resampling into monthly basis
pd.plotting.autocorrelation_plot(a_dna['2014':'2016']['TEMP'].resample('1m').mean())
pd.plotting.autocorrelation_plot(a_dna['2014':'2016']['PM2.5'].resample('1m').mean())
