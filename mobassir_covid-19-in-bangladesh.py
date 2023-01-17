from IPython.core.display import HTML





HTML('''<div class="flourish-embed flourish-cards" data-src="visualisation/1884138" data-url="https://flo.uri.sh/visualisation/1884138/embed"><script src="https://public.flourish.studio/resources/embed.js"></script></div>''')



HTML('''<div class="flourish-embed flourish-cards" data-src="visualisation/1477361" data-url="https://public.flourish.studio/visualisation/1477361/embed"><script src="https://public.flourish.studio/resources/embed.js"></script></div>''')
import os

from pathlib import Path

import random

import sys



from tqdm.notebook import tqdm

import numpy as np

import pandas as pd

import scipy as sp





import matplotlib.pyplot as plt

import seaborn as sns



from IPython.core.display import display, HTML



# --- plotly ---

from plotly import tools, subplots

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.express as px

import plotly.figure_factory as ff

import plotly.io as pio

pio.templates.default = "simple_white"



# --- models ---

from sklearn import preprocessing

from sklearn.model_selection import KFold

import lightgbm as lgb

import xgboost as xgb

import catboost as cb



# --- setup ---

pd.set_option('max_columns', 50)
a = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/train.csv')

a
country_df = a.groupby(['Date', 'Country_Region'])[['ConfirmedCases', 'Fatalities']].sum().reset_index()



country_df.tail()



target_date = country_df['Date'].max()



print('Date: ', target_date)

for i in [1, 10, 100, 1000, 10000]:

    n_countries = len(country_df.query('(Date == @target_date) & ConfirmedCases > @i'))

    print(f'{n_countries} countries have more than {i} confirmed cases')

    

    

top_country_df = country_df.query('(Date == @target_date) & (ConfirmedCases > 1000)').sort_values('ConfirmedCases', ascending=False)

top_country_melt_df = pd.melt(top_country_df, id_vars='Country_Region', value_vars=['ConfirmedCases', 'Fatalities'])



#from https://www.kaggle.com/corochann/covid-19-eda-with-recent-update-on-april

fig = px.bar(top_country_melt_df.iloc[::-1],

             x='value', y='Country_Region', color='variable', barmode='group',

             title=f'Confirmed Cases/Deaths on {target_date}', text='value', height=2000, orientation='h', template='plotly_dark')

fig.show()

a = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/train.csv')

country_df = a.groupby(['Date', 'Country_Region'])[['ConfirmedCases', 'Fatalities']].sum().reset_index()

target_date = country_df['Date'].max()



print('Date: ', target_date)

top_country_df = country_df.query('(Date == @target_date) & (ConfirmedCases > 1000)').sort_values('ConfirmedCases', ascending=False)

top_country_melt_df = pd.melt(top_country_df, id_vars='Country_Region', value_vars=['ConfirmedCases', 'Fatalities'])
#from https://www.kaggle.com/corochann/covid-19-eda-with-recent-update-on-april

fig = px.bar(top_country_melt_df.iloc[::-1],

             x='value', y='Country_Region', color='variable', barmode='group',

             title=f'(After 1 week ) Confirmed Cases/Deaths on {target_date}', text='value', height=2000, orientation='h')

fig.show()
train = a

BD_df = train.query('Country_Region == "Bangladesh"')

BD_df['prev_confirmed'] = BD_df.groupby('Country_Region')['ConfirmedCases'].shift(1)

BD_df['new_case'] = BD_df['ConfirmedCases'] - BD_df['prev_confirmed']

BD_df.loc[BD_df['new_case'] < 0, 'new_case'] = 0.
fig = px.line(BD_df,

              x='Date', y='new_case', color='Country_Region',

              title=f'DAILY NEW Confirmed cases in Bangladesh(Till : 2020-04-7)')

fig.show()
bd_df = pd.read_csv('../input/covid19-in-bangladesh/COVID-19_in_bd.csv')

bd_df.head()


bd_df = bd_df

bd_df['prev_confirmed'] = bd_df['Confirmed'].shift(1)

bd_df['new_case'] = bd_df['Confirmed'] - bd_df['prev_confirmed']

bd_df.loc[bd_df['new_case'] < 0, 'new_case'] = 0.
fig = px.line(bd_df,

              x='Date', y='new_case',

              title=f'DAILY NEW Confirmed cases in Bangladesh')

fig.show()
bd_df = bd_df

bd_df['prev_confirmed'] = bd_df['Deaths'].shift(1)

bd_df['new_case'] = bd_df['Deaths'] - bd_df['prev_confirmed']

bd_df.loc[bd_df['new_case'] < 0, 'new_case'] = 0.
fig = px.line(bd_df,

              x='Date', y='new_case',

              title=f'DAILY NEW Deaths in Bangladesh')

fig.show()
bd_df = bd_df

bd_df['prev_confirmed'] = bd_df['Recovered'].shift(1)

bd_df['new_case'] = bd_df['Recovered'] - bd_df['prev_confirmed']

bd_df.loc[bd_df['new_case'] < 0, 'new_case'] = 0.
fig = px.line(bd_df,

              x='Date', y='new_case',

              title=f'DAILY Recovering rate in Bangladesh')

fig.show()
df1 = pd.read_csv('../input/covid19-bangladesh-dataset/COVID-19-Bangladesh.csv')

df1.head()
# Grouping cases by date 

df = pd.read_csv('../input/covid19-bangladesh-dataset/COVID-19-Bangladesh.csv')

temp = df.groupby('Date')['Confirmed', 'Recovered', 'Deaths','Quarantine'].sum().reset_index() 

# Unpivoting 

temp = temp.melt(id_vars='Date',value_vars = ['Confirmed', 'Recovered', 'Deaths'], var_name='Case', value_name='Count') 

# Visualization

fig = px.line(temp, x='Date', y='Count', color='Case', color_discrete_sequence=['#ff0000', '#FFFF00', '#0000FF' , '#0020FF'], template='presentation') 

fig.update_layout(title="COVID-19 Cases Over Time in Bangladesh")

fig.show()
# Grouping cases by date 

df = pd.read_csv('../input/covid19-bangladesh-dataset/COVID-19-Bangladesh.csv')

temp = df.groupby('Date')['Quarantine','Released From Quarantine'].sum().reset_index() 

# Unpivoting 

temp = temp.melt(id_vars='Date',value_vars = ['Quarantine','Released From Quarantine'], var_name='Case', value_name='Count') 

# Visualization

fig = px.line(temp, x='Date', y='Count', color='Case', color_discrete_sequence=[ '#ff0000', '#FFFF00'], template='ggplot2') 

fig.update_layout(title="COVID-19 Cases Over Time in Bangladesh")

fig.show()
# Grouping cases by date 

df = bd_df

temp = df.groupby('Date')['Confirmed', 'Recovered', 'Deaths'].sum().reset_index() 

# Unpivoting 

temp = temp.melt(id_vars='Date',value_vars = ['Confirmed', 'Recovered', 'Deaths'], var_name='Case', value_name='Count') 



# Visualization

fig = px.line(temp, x='Date', y='Count', color='Case', color_discrete_sequence=['#ff0000', '#FFFF00', '#0000FF'], template='plotly_dark') 

fig.update_layout(title="COVID-19 Cases Over Time in Bangladesh")

fig.show()
# from https://pypi.org/project/COVID19Py/



!pip install COVID19Py
import COVID19Py

covid19 = COVID19Py.COVID19()

loc = covid19.getLocationByCountryCode("BD")

locData = loc[0]

virus = dict(locData['latest'])

print(virus)
latest = covid19.getLatest()

latest
locations = covid19.getLocations(rank_by='recovered')

locations
locations[20] #bd
changes = covid19.getLatestChanges()

changes
locations[20]['latest']['recovered'] #bd
columns = ['country_code','country_population', 'confirmed', 'deaths', 'recovered']

df = pd.DataFrame( columns=columns)

df.iloc[:,2]
for i in range(len(locations)):

    df = df.append({'confirmed': locations[i]['latest']['confirmed'],

                    

                    'deaths': locations[i]['latest']['deaths'],

                    

                    'recovered': locations[i]['latest']['recovered'],

                   

                   'country_code' : locations[i]['country_code'],

                    

                   'country_population' : locations[i]['country_population'],

                   

                    'last_updated' : locations[i]['last_updated'],

                    

                    'latitude' : locations[i]['coordinates']['latitude'],

                    

                    'longitude' : locations[i]['coordinates']['longitude']

  

                   },

                   ignore_index=True)
from datetime import datetime

date = datetime.date(datetime.now())

print(datetime.date(datetime.now()))

df.to_csv(f'covid{date}.csv', index=False)

df
target_date = df['last_updated'].max()



print('Date: ', target_date)

for i in [1, 10, 100, 1000, 10000]:

    n_countries = len(df.query('(last_updated == @target_date) & confirmed > @i'))

    print(f'{n_countries} countries have more than {i} confirmed cases')
top_country_df = df.query(' (deaths > 11)').sort_values('deaths', ascending=True)

top_country_melt_df = pd.melt(top_country_df, id_vars='country_code', value_vars=['deaths'])



top_country_df1 = df.query(' (confirmed > 100)').sort_values('confirmed', ascending=True)

top_country_melt_df1 = pd.melt(top_country_df1, id_vars='country_code', value_vars=['confirmed'])

#from https://www.kaggle.com/corochann/covid-19-eda-with-recent-update-on-april

fig = px.bar(top_country_melt_df.iloc[::-1],

             x='value', y='country_code', color='variable', barmode='group',color_discrete_sequence=['#ff0000'], template='seaborn',

             title=f'Lowest Deaths ', text='value', height=2000, orientation='h')

fig.show()
# Grouping cases by date 



temp = df.groupby('last_updated')['confirmed', 'recovered', 'deaths'].sum().reset_index() 

# Unpivoting 

temp = temp.melt(id_vars='last_updated',value_vars = ['confirmed', 'recovered', 'deaths'], var_name='Case', value_name='Count') 





# Visualization

fig = px.line(temp, x='last_updated', y='Count', color='Case', color_discrete_sequence=['#ff0000', '#FFFF00', '#0000FF'], template='plotly_dark') 

fig.update_layout(title="COVID-19 Cases Over worldwide")

fig.show()



#from https://www.kaggle.com/corochann/covid-19-eda-with-recent-update-on-april

fig = px.bar(top_country_melt_df1.iloc[::-1],

             x='value', y='country_code', color='variable', barmode='group',color_discrete_sequence=['#ff0000'], template='seaborn',

             title=f'Lowest confirmed ', text='value', height=2000, orientation='h')

fig.show()
print(len(df.recovered))

df.recovered.sum()
# from https://github.com/imdevskp/covid_19_jhu_data_web_scrap_and_cleaning

#original author : https://www.kaggle.com/imdevskp/covid-19-analysis-visualization-comparisons



HTML('''<div class="flourish-embed flourish-bar-chart-race" data-src="visualisation/1571387"><script src="https://public.flourish.studio/resources/embed.js"></script></div>''')


HTML('''<div class="flourish-embed flourish-chart" data-src="story/230114"><script src="https://public.flourish.studio/resources/embed.js"></script></div>''')


HTML('''<div class="flourish-embed flourish-map" data-src="story/225979"><script src="https://public.flourish.studio/resources/embed.js"></script></div>''')


HTML('''<div class="flourish-embed flourish-chart" data-src="story/230085"><script src="https://public.flourish.studio/resources/embed.js"></script></div>''')


HTML('''<div class="flourish-embed flourish-chart" data-src="story/230110"><script src="https://public.flourish.studio/resources/embed.js"></script></div>''')


HTML('''<div class="flourish-embed flourish-map" data-src="story/229998"><script src="https://public.flourish.studio/resources/embed.js"></script></div>''')


HTML('''<div class="flourish-embed flourish-table" data-src="story/230195"><script src="https://public.flourish.studio/resources/embed.js"></script></div>''')