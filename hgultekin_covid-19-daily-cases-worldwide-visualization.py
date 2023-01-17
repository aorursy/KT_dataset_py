import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import datetime
from  pandas import json_normalize
#sns.set_style('whitegrid')
#from plotly.offline import init_notebook_mode, iplot, plot
#import plotly as py
#import plotly.graph_objs as go

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#df = pd.read_csv('../input/covid19-coronavirus-dataset/COVID-19-geographic-disbtribution-worldwide-2020-08-02.csv')
d = pd.read_json('/kaggle/input/covid19-stream-data/json')
df = json_normalize(d['records'])
df
df.dateRep = pd.to_datetime(df.dateRep, format="%d/%m/%Y")
df[df.countriesAndTerritories=='']
df.info()
df.isnull().sum()
df.countriesAndTerritories.nunique()
df.countriesAndTerritories.unique()
df[df.countriesAndTerritories == 'United_Kingdom'].sort_values('dateRep')
df.head(10)
df = df.sort_values('dateRep')
df.groupby('continentExp').sum()
px.pie(df, df.groupby('continentExp').sum().index, df.groupby('continentExp').sum().cases,title='Covid-19 Cases w.r.t Continents')
px.pie(df, df.groupby('continentExp').sum().index, df.groupby('continentExp').sum().deaths, title='Covid-19 Mortality w.r.t Continents')
print(df.dateRep.min(), df.dateRep.max())
df['cases_cum'] = df.groupby('countriesAndTerritories').cases.cumsum()
df['deaths_cum'] = df.groupby('countriesAndTerritories').deaths.cumsum()
df
fig = px.line(df, x="dateRep", y="cases_cum", color="continentExp",
              line_group="countriesAndTerritories", hover_name="countriesAndTerritories", title= 'Covid-19 Cases w.r.t. Continents')
fig.show()
fig = px.line(df, x="dateRep", y="deaths_cum", color="continentExp",
              line_group="countriesAndTerritories", hover_name="countriesAndTerritories", title= "Covid-19 Deaths w.r.t. Continents")
fig.show()
df
df.dateRep = df.dateRep.apply(lambda x: x.strftime('%Y-%m-%d'))
fig = px.scatter_geo(df, locations="countryterritoryCode",
                    color="cases_cum",
                    size="cases_cum",
                    hover_name="countriesAndTerritories",
                    #animation_frame="dateRep",
                    title = "Total COVID cases by 2020-08-02",
                    color_continuous_scale=px.colors.cyclical.IceFire,
                    #width =1200,
                    #height=600,
                    projection = 'equirectangular',
                    scope = 'world',
                    size_max=20)


fig.update_layout(margin={'r':0,'t':0,'l':0,'b':0})
fig.show()
fig = px.choropleth(df, locations="countryterritoryCode",
                    color="cases_cum",
                    hover_name="countriesAndTerritories",
                    animation_frame="dateRep",
                    title = "Daily Total COVID cases",
                    color_continuous_scale=px.colors.sequential.Reds,
                    #width =1200,
                    #height=600,
                    projection = 'equirectangular',
                    scope = 'world')


fig.update_layout(margin={'r':0,'t':0,'l':0,'b':0})
fig.show()
fig = px.choropleth(df, locations="countryterritoryCode",
                    color="deaths_cum",
                    hover_name="countriesAndTerritories",
                    animation_frame="dateRep",
                    title = "Daily Total COVID Deaths",
                    color_continuous_scale=px.colors.sequential.Reds,
                    #width =1200,
                    #height=600,
                    projection = 'equirectangular',
                    scope = 'world')


fig.update_layout(margin={'r':0,'t':0,'l':0,'b':0})
fig.show()