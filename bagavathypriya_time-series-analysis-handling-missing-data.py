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
def convert_to_date(x):
    return datetime.strptime(x, '%Y %m %d %H')
a_df = pd.read_csv('/kaggle/temp/PRSA_Data_20130301-20170228/PRSA_Data_Dingling_20130301-20170228.csv', 
                    parse_dates = [['year', 'month', 'day', 'hour']],date_parser=convert_to_date)
a_df.head()
a_df.info()
a_df.describe()
a_df.isnull().sum()
a_df.query('TEMP!=TEMP')
a_df.query('TEMP!=TEMP').count()
a_df[a_df['PM2.5'].isnull()]
a_df[a_df['PM2.5'].isnull()].count()
import plotly.express as px

fig = px.line(a_df, x='year_month_day_hour', y='PM2.5', title='PM2.5 with Slider')

fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(count=2, label="2y", step="year", stepmode="backward"),
            dict(count=3, label="3y", step="year", stepmode="backward"),
            dict(step="all")
        ])
    )
)
fig.show()
fig = px.line(a_df, x='year_month_day_hour', y='TEMP', title='TEMP with Slider')

fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(count=2, label="2y", step="year", stepmode="backward"),
            dict(count=3, label="3y", step="year", stepmode="backward"),
            dict(step="all")
        ])
    )
)
fig.show()
a_df=a_df.set_index('year_month_day_hour')
a_df.head()
a_df.loc['2015-02-02':'2015-02-04']
a_dfna=a_df.copy()

a_dfna=a_dfna.dropna()
pd.plotting.autocorrelation_plot(a_dfna['2014':'2016']['TEMP'])
a_df['2015-02-21 10':'2015-02-21 20']
a_df_imp=a_df['2015-02-21 10':'2015-02-21 23'][['TEMP']]
a_df_imp
a_df_imp['TEMP_FILL']=a_df_imp['TEMP'].fillna(method='ffill')
a_df_imp
a_df_imp['Temp_bfill']=a_df_imp['TEMP'].fillna(method='bfill')
a_df_imp
a_df_imp['Temp_roll']=a_df_imp['TEMP'].rolling(window=2,min_periods=1).mean()
a_df_imp
a_df.loc[a_df_imp.index+pd.offsets.DateOffset(years=-1)]['TEMP']
a_df_imp.index
a_df_imp=a_df_imp.reset_index()
a_df_imp['Temp_prev']=a_df_imp.apply(lambda x:a_df.loc[x['year_month_day_hour']-pd.offsets.DateOffset(years=-1)]['TEMP'] 
                                     if pd.isna(x['TEMP']) else x['TEMP'],axis=1)
a_df_imp
