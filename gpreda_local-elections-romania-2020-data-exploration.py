import numpy as np 

import pandas as pd

import os

import matplotlib.pyplot as plt

import seaborn as sns 

import datetime as dt

%matplotlib inline

import datetime as dt

import plotly.graph_objs as go

import plotly.figure_factory as ff

from plotly import tools

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

from shapely.geometry import shape, Point, Polygon

import folium

from folium.plugins import HeatMap, HeatMapWithTime

init_notebook_mode(connected=True)
data_folder = "/kaggle/input/local-elections-romania-2020/"

data_files = os.listdir(data_folder)



data_df = pd.DataFrame()

for data_file in data_files:

    date_hour = data_file.split("_")

    date = date_hour[1]

    hour = date_hour[2].split("-")[0]

    print(date, hour)

    _df = pd.read_csv(os.path.join(data_folder, data_file))

    _df['date'] = date

    _df['hour'] = hour

    data_df = data_df.append(_df)

print(list(data_df.columns))
data_df[['Judet', 'UAT', 'Localitate', 'Siruta', 'Nr sectie de votare',

       'Nume sectie de votare', 'Mediu', 'Votanti pe lista permanenta',

       'Votanti pe lista complementara', 'LP', 'LC', 'LS', 'UM', 'LT', 'hour']].head()
lt_hour_judet_df = data_df.groupby(["Judet", "hour"])["LT"].sum().reset_index()

lt_hour_judet_df.columns = ["Judet", "Hour", "Total"]
max_hour = lt_hour_judet_df['Hour'].max()

print("Last hour: ", max_hour)


d_df = lt_hour_judet_df.loc[lt_hour_judet_df.Hour==max_hour]

d_df = d_df.sort_values(by=['Total'], ascending = False)



hover_text = []

for index, row in d_df.iterrows():

    hover_text.append(('Judet: {}<br>'+

                      'Votanti: {}').format(row['Judet'], row['Total']))

d_df['hover_text'] = hover_text



    

trace = go.Bar(

    x = d_df['Judet'],y = d_df['Total'],

    name='Total',

    marker=dict(color='Red'),

    text = hover_text,

)



data = [trace]

layout = dict(title = 'Numar total votanti / judet',

          xaxis = dict(title = 'Judet', showticklabels=True),

          yaxis = dict(title = 'Numar total votanti'),

          hovermode = 'closest',

         )

fig = dict(data=data, layout=layout)

iplot(fig, filename='alegeri-locale-2020')
d_df = lt_hour_judet_df.copy()

d_df = d_df.loc[d_df['Total']>0]

counties = list(d_df.Judet.unique())



data = []

for county in counties:

    dc_df = d_df.loc[d_df.Judet==county]

    traceC = go.Scatter(

        x = dc_df['Hour'],y = dc_df['Total'],

        name=county,

        mode = "markers+lines",

        text=dc_df['Total']

    )

    data.append(traceC)



layout = dict(title = 'Numar total votanti / judet (log scale)',

          xaxis = dict(title = 'Ora', showticklabels=True), 

          yaxis = dict(title = 'Total votanti (log scale)'),

          yaxis_type="log",

          hovermode = 'y',

          height=1000

         )



fig = dict(data=data, layout=layout)

iplot(fig, filename='alegeri-locale-2020')
total_list_df = data_df.loc[data_df['hour']=='12']

total_list_df = total_list_df.groupby(['Judet', 'hour'])['Votanti pe lista permanenta'].sum().reset_index()

total_list_df.columns = ['Judet', 'Hour', 'TLP']

total_list_df.head()
lt_hour_judet_df = lt_hour_judet_df.merge(total_list_df[['Judet', 'TLP']], on=['Judet'])
lt_hour_judet_df['Percent'] = lt_hour_judet_df['Total'] / lt_hour_judet_df['TLP']
d_df = lt_hour_judet_df.loc[lt_hour_judet_df.Hour==max_hour]

d_df = d_df.sort_values(by=['Percent'], ascending = False)



hover_text = []

for index, row in d_df.iterrows():

    hover_text.append(('Judet: {}<br>'+

                      'Procent votanti: {}').format(row['Judet'], row['Percent']))

d_df['hover_text'] = hover_text



    

trace = go.Bar(

    x = d_df['Judet'],y = d_df['Percent'],

    name='Percent',

    marker=dict(color='Red'),

    text = hover_text,

)



data = [trace]

layout = dict(title = 'Procent votanti din numarul total votanti pe lista principala / judet',

          xaxis = dict(title = 'Judet', showticklabels=True),

          yaxis = dict(title = 'Procent votanti'),

          hovermode = 'closest',

         )

fig = dict(data=data, layout=layout)

iplot(fig, filename='alegeri-locale-2020-procent')
d_df = lt_hour_judet_df.copy()

d_df = d_df.loc[d_df['Percent']>0]

counties = list(d_df.Judet.unique())



data = []

for county in counties:

    dc_df = d_df.loc[d_df.Judet==county]

    traceC = go.Scatter(

        x = dc_df['Hour'],y = dc_df['Percent'],

        name=county,

        mode = "markers+lines",

        text=dc_df['Percent']

    )

    data.append(traceC)



layout = dict(title = 'Procent votanti din numarul total de votanti pe liste principale / judet',

          xaxis = dict(title = 'Ora', showticklabels=True), 

          yaxis = dict(title = 'Procent votanti'),

          hovermode = 'y',

          height=1000

         )



fig = dict(data=data, layout=layout)

iplot(fig, filename='alegeri-locale-2020-procent-votanti')
b_df = data_df.loc[data_df.Judet=='B']

lt_hour_df = b_df.groupby(["UAT", "hour"])["LT"].sum().reset_index()

lt_hour_df.columns = ["UAT", "Hour", "Total"]

lt_hour_df.head()
total_b_df = b_df.loc[b_df['hour']=='12']

total_b_df = total_b_df.groupby(['UAT', 'hour'])['Votanti pe lista permanenta'].sum().reset_index()

total_b_df.columns = ['UAT', 'Hour', 'TLP']

total_b_df.head(6)
lt_hour_df = lt_hour_df.merge(total_b_df[['UAT', 'TLP']], on=['UAT'])

lt_hour_df['Percent'] = lt_hour_df['Total'] / lt_hour_df['TLP']
d_df = lt_hour_df.loc[lt_hour_df.Hour==max_hour]

d_df = d_df.sort_values(by=['Percent'], ascending = False)



hover_text = []

for index, row in d_df.iterrows():

    hover_text.append(('Sector: {}<br>'+

                      'Procent votanti: {}').format(row['UAT'], row['Percent']))

d_df['hover_text'] = hover_text



    

trace = go.Bar(

    x = d_df['UAT'],y = d_df['Percent'],

    name='Percent',

    marker=dict(color='Blue'),

    text = hover_text,

)



data = [trace]

layout = dict(title = 'Procent votanti din numarul total votanti pe lista principala / UAT',

          xaxis = dict(title = 'Sector', showticklabels=True),

          yaxis = dict(title = 'Procent votanti'),

          hovermode = 'closest',

         )

fig = dict(data=data, layout=layout)

iplot(fig, filename='alegeri-locale-2020-procent')
d_df = lt_hour_df.copy()

d_df = d_df.loc[d_df['Percent']>0]

uats = list(d_df.UAT.unique())



data = []

for uat in uats:

    dc_df = d_df.loc[d_df.UAT==uat]

    traceC = go.Scatter(

        x = dc_df['Hour'],y = dc_df['Percent'],

        name=uat,

        mode = "markers+lines",

        text=dc_df['Percent']

    )

    data.append(traceC)



layout = dict(title = 'Bucuresti: procent votanti din numarul total de votanti pe liste principale / Sector',

          xaxis = dict(title = 'Ora', showticklabels=True), 

          yaxis = dict(title = 'Procent votanti'),

          hovermode = 'y',

          height=800

         )



fig = dict(data=data, layout=layout)

iplot(fig, filename='alegeri-locale-2020-procent-votanti')
columns = ['Judet', 'UAT', 'Barbati 18-24','Barbati 25-34', 'Barbati 35-44', 'Barbati 45-64', 'Barbati 65+',

 'Femei 18-24', 'Femei 25-34', 'Femei 35-44', 'Femei 45-64', 'Femei 65+','hour']

bsa_df = b_df[columns]
bsa_df
absa_df = bsa_df.groupby(['hour']).sum().reset_index()
d_df = absa_df.copy()

data = []

for column in absa_df.columns[1:6]:

    trace = go.Bar(

        x = d_df['hour'],y = d_df[column],

        name=column,

        marker=dict(color='Blue'),

        text = column,

    )

    data.append(trace)

for column in absa_df.columns[6:]:

    trace = go.Bar(

        x = d_df['hour'],y = d_df[column],

        name=column,

        marker=dict(color='Red'),

        text = column,

    )

    data.append(trace)    

layout = dict(title = 'Votanti Bucuresti - pe grupe de varsta si sex',

          xaxis = dict(title = 'Ora', showticklabels=True),

          yaxis = dict(title = 'Numar votanti - pe grupe de varsta si sex'),

          hovermode = 'closest',

         )

fig = dict(data=data, layout=layout)

iplot(fig, filename='alegeri-locale-2020-procent')
colors = ['lightblue', 'aquamarine', 'blue', 'darkblue', 'black']

colors_r = ['yellow', 'orange', 'tomato', 'red', 'darkred']

d_df = bsa_df.loc[bsa_df.UAT=='BUCUREŞTI SECTORUL 6'].groupby(['hour']).sum().reset_index().copy()

data = []

for i, column in enumerate(absa_df.columns[1:6]):

    trace = go.Bar(

        x = d_df['hour'],y = d_df[column],

        name=column,

        marker=dict(color=colors[i]),

        text = column,

    )

    data.append(trace)

for i, column in enumerate(absa_df.columns[6:]):

    trace = go.Bar(

        x = d_df['hour'],y = d_df[column],

        name=column,

        marker=dict(color=colors_r[i]),

        text = column,

    )

    data.append(trace)    

layout = dict(title = 'Votanti Sectorul 6 - pe grupe de varsta si sex',

          xaxis = dict(title = 'Ora', showticklabels=True),

          yaxis = dict(title = 'Numar votanti - pe grupe de varsta si sex'),

          hovermode = 'closest',

         )

fig = dict(data=data, layout=layout)

iplot(fig, filename='alegeri-locale-2020-procent')
d_df = bsa_df.loc[bsa_df.hour==max_hour].groupby(['UAT']).sum().reset_index().copy()

data = []

for i, column in enumerate(absa_df.columns[1:6]):

    trace = go.Bar(

        x = d_df['UAT'],y = d_df[column],

        name=column,

        marker=dict(color=colors[i]),

        text = column,

    )

    data.append(trace)

for i, column in enumerate(absa_df.columns[6:]):

    trace = go.Bar(

        x = d_df['UAT'],y = d_df[column],

        name=column,

        marker=dict(color=colors_r[i]),

        text = column,

    )

    data.append(trace)    

layout = dict(title = 'Votanti Bucuresti grupati pe sectoare, grupe de varsta si sex',

          xaxis = dict(title = 'Sector', showticklabels=True),

          yaxis = dict(title = 'Numar votanti - pe sectoare, grupe de varsta si sex'),

          hovermode = 'closest',

         )

fig = dict(data=data, layout=layout)

iplot(fig, filename='alegeri-locale-2020-procent')