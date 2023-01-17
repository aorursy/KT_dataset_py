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
!pip install chart_studio
!pip install plotly-geo
# Import Libraries
import numpy as np 
import pandas as pd 
#Import plotly libraries
import chart_studio.plotly as py
import plotly.tools as tls
import plotly.graph_objs as go
import plotly
import plotly.figure_factory as ff
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
plotly.offline.init_notebook_mode(connected=True)
majordir='/kaggle/input/covid19-us-county-trend/'
datadir=majordir+'csse_covid_19_daily_reports/'
date_today=27
covid_data_world_daily_0322=pd.read_csv(datadir+'03-22-2020.csv')
covid_data_world_daily_0322.rename(columns={'Confirmed':'Confirmed_0322'},inplace=True)
covid_data_world_daily_0322.head()
covid_data_us_daily_0322=covid_data_world_daily_0322[covid_data_world_daily_0322['Country_Region']=='US'].copy()
covid_data_us_daily_0322.shape
for i in range(23,date_today):
    dataset=datadir+'03-'+str(i)+'-2020.csv'
    col='Confirmed_03'+str(i)
    covid_data_world_daily=pd.read_csv(dataset)
    covid_data_world_daily.rename(columns={'Confirmed':col},inplace=True)
    covid_data_us_daily=covid_data_world_daily[covid_data_world_daily['Country_Region']=='US'].copy()

    if i==23:        
        covid_data_us_dailytrend=covid_data_us_daily_0322[['FIPS','Confirmed_0322']].merge(covid_data_us_daily[['FIPS',col]],on='FIPS').dropna()
    else:
        print(i)
        covid_data_us_dailytrend=covid_data_us_dailytrend.merge(covid_data_us_daily[['FIPS',col]],on='FIPS').dropna()
covid_data_us_dailytrend.shape
covid_data_us_dailytrend_nonan=covid_data_us_dailytrend[covid_data_us_dailytrend['FIPS'].notna()].copy()
covid_data_us_dailytrend_nonan.FIPS=covid_data_us_dailytrend_nonan.FIPS.astype(str).str.split('.',expand=True)[0]
census_df_fips = pd.read_excel(majordir+'PopulationEstimates_us_county_level_2018.xlsx',skiprows=1)
census_df_fips.head()
fips = covid_data_us_dailytrend_nonan.FIPS.tolist()
values =covid_data_us_dailytrend_nonan.Confirmed_0322.tolist()

fig = ff.create_choropleth(fips=fips, values=values)
fig.layout.template = None
fig.show()
col='Confirmed_03'+str(date_today-1)
fips = covid_data_us_dailytrend_nonan.FIPS.tolist()
values =covid_data_us_dailytrend_nonan[col].tolist()

fig = ff.create_choropleth(fips=fips, values=values)
fig.layout.template = None
fig.show()
census_df_fips_covid=census_df_fips.merge(covid_data_us_dailytrend,on='FIPS') 
census_df_fips_covid.shape
census_df_fips_covid.head()
for i in range(22,date_today):
    col='Confirmed_03'+str(i)
    col_10000='Confirmed_per10000_03'+str(i)
    census_df_fips_covid[col_10000]=10000*(census_df_fips_covid[col]/census_df_fips_covid['POP_ESTIMATE_2018'])
census_df_fips_covid.head()
fips = census_df_fips_covid.FIPS.tolist()
values =census_df_fips_covid.Confirmed_per10000_0322.tolist()

fig = ff.create_choropleth(fips=fips, values=values)
fig.layout.template = None
fig.show()
col='Confirmed_per10000_03'+str(date_today-1)
fips = census_df_fips_covid.FIPS.tolist()
values =census_df_fips_covid[col].tolist()

fig = ff.create_choropleth(fips=fips, values=values)
fig.layout.template = None
fig.show()