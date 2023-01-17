# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline 
#from bubbly.bubbly import bubbleplot 
#from __future__ import division
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

IS_LOCAL = False
import os
if(IS_LOCAL):
    PATH="../input/120-years-of-olympic-history-athlets-and-results"
else:
    PATH="../input"
print(os.listdir(PATH))
athlete_events_df = pd.read_csv(PATH+"/athlete_events.csv")
noc_regions_df = pd.read_csv(PATH+"/noc_regions.csv")
athlete_events_df.head(10)
def missing_data(data):
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data(athlete_events_df)
tmp = athlete_events_df.groupby(['Year', 'Sex', 'Season'])['ID'].nunique()
df = pd.DataFrame(data={'Athlets': tmp.values}, index=tmp.index).reset_index()
df_summer = df[df['Season']=='Summer']; 
df_winter = df[df['Season']=='Winter']; 

dfS_women= df_summer[df_summer['Sex']=='F']; 
dfS_men = df_summer[df_summer['Sex']=='M']

traceS_women = go.Scatter(
    x = dfS_women['Year'],y = dfS_women['Athlets'],
    name="Women",
    marker=dict(color="#f64f59"),
    mode = "markers+lines"
)
traceS_men = go.Scatter(
    x = dfS_men['Year'],y = dfS_men['Athlets'],
    name="Men",
    marker=dict(color="#12c2e9"),
    mode = "markers+lines"
)

data = [traceS_women, traceS_men]
layout = dict(title = 'Number of men and women in Summer Olympics since 1896',
          xaxis = dict(title = 'Year', showticklabels=True), 
          yaxis = dict(title = 'Number of athlets'),
         )
fig = dict(data=data, layout=layout)
iplot(fig, filename='events-athlets2')


dfW_women= df_winter[df_winter['Sex']=='F']; 
dfW_men = df_winter[df_winter['Sex']=='M']

traceW_women = go.Scatter(
    x = dfW_women['Year'],y = dfW_women['Athlets'],
    name="Women",
    marker=dict(color="#f64f59"),
    mode = "markers+lines"
)
traceW_men = go.Scatter(
    x = dfW_men['Year'],y = dfW_men['Athlets'],
    name="Men",
    marker=dict(color="#12c2e9"),
    mode = "markers+lines"
)

data = [traceW_women, traceW_men]
layout = dict(title = 'Number of men and women in Winter Olympics since 1896',
          xaxis = dict(title = 'Year', showticklabels=True), 
          yaxis = dict(title = 'Number of athlets'),
         )
fig = dict(data=data, layout=layout)
iplot(fig, filename='events-athlets2')
ae_summer = athlete_events_df[athlete_events_df['Season']=='Summer'].copy()

# getting the age median per sports for each year 
tmp = ae_summer.groupby(['Year', 'Sport'])['Age'].median()

df_summer_sports = pd.DataFrame(data={'Median Age': tmp.values}, index=tmp.index).reset_index()

# plot set up
trace2 = go.Heatmap(z=np.array(df_summer_sports['Median Age']),
                    x=np.array(df_summer_sports['Year']),
                    y=np.array(df_summer_sports['Sport']),
                    colorscale=[[0, '#12c2e9'], [0.5, '#c471ed'], [1, '#f64f59']],
                    xgap=0.9,
                    ygap=0.9,
                    colorbar=dict(
                        title='Age',
                        titleside='right',
                        titlefont=dict(
                            size=20,
                            color='#4b8480'),
                        outlinecolor='white',
                        tickcolor='#4b8480',
                        tickfont=dict(color='#4b8480'),
                        tickmode='array',
                        ticktext=['max', 'average', 'min']
                    )
                   )

data2 = [trace2]
layout2 = go.Layout(
    title='Median age by sport across time (summer)',
    titlefont=dict(
        size=24,
        color='#4b8480'
    ),
    height=800,
    xaxis=dict(title='Year', 
               titlefont=dict(
                   size=16
               ),
               showgrid=False, 
               color='#4b8480'
              ),
    yaxis=dict(
               color='#4b8480',
               tickfont=dict(
                   size=9
               ),
               automargin=True,
               showgrid=True
              )
)
fig2 = go.Figure(data=data2, layout=layout2)

iplot(fig2)



ae_winter = athlete_events_df[athlete_events_df['Season']=='Winter'].copy()

# getting the age median per sports for each year 
tmp = ae_winter.groupby(['Year', 'Sport'])['Age'].median()

df_winter_sports = pd.DataFrame(data={'Median Age': tmp.values}, index=tmp.index).reset_index()

# plot set up
trace2 = go.Heatmap(z=np.array(df_winter_sports['Median Age']),
                    x=np.array(df_winter_sports['Year']),
                    y=np.array(df_winter_sports['Sport']),
                    colorscale=[[0, '#12c2e9'], [0.5, '#c471ed'], [1, '#f64f59']],
                    xgap=0.9,
                    ygap=0.9,
                    colorbar=dict(
                        title='Age',
                        titleside='right',
                        titlefont=dict(
                            size=20,
                            color='#4b8480'),
                        outlinecolor='white',
                        tickcolor='#4b8480',
                        tickfont=dict(color='#4b8480'),
                        tickmode='array',
                        ticktext=['max', 'average', 'min']
                    )
                   )

data2 = [trace2]
layout2 = go.Layout(
    title='Median age by sport across time (winter)',
    titlefont=dict(
        size=24,
        color='#4b8480'
    ),
    height=800,
    xaxis=dict(title='Year', 
               titlefont=dict(
                   size=16
               ),
               showgrid=False, 
               color='#4b8480'
              ),
    yaxis=dict(
               color='#4b8480',
               tickfont=dict(
                   size=9
               ),
               automargin=True,
               showgrid=False
              )
)
fig2 = go.Figure(data=data2, layout=layout2)

iplot(fig2)

gold_medals = athlete_events_df[(athlete_events_df.Medal == 'Gold')]
silver_medals = athlete_events_df[(athlete_events_df.Medal == 'Silver')]
bronze_medals = athlete_events_df[(athlete_events_df.Medal == 'Bronze')]

NOC_gold_medals = gold_medals.NOC.value_counts().reset_index(name='Medal')
NOC_silver_medals = silver_medals.NOC.value_counts().reset_index(name='Medal')
NOC_bronze_medals = bronze_medals.NOC.value_counts().reset_index(name='Medal')

data = [ dict(
        type = 'choropleth',
        locations = NOC_gold_medals['index'],
        locationmode = 'ISO-3',
        z = NOC_gold_medals['Medal'],
        text = 'Gold medals',
        colorscale=[[0, '#12c2e9'], [0.5, '#c471ed'], [1, '#f64f59']],
        autocolorscale = False,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            tickprefix = '',
            title = 'Gold medals'),
      ) ]
layout = dict(
    title = 'Gold medals per countries',
    geo = dict(
        showframe = False,
        showcoastlines = True,
        projection = dict(
            type = 'Mercator'
        )
    )
)
fig = dict( data=data, layout=layout )
iplot( fig, validate=False, filename='world-map')
data = [ dict(
        type = 'choropleth',
        locations = NOC_silver_medals['index'],
        locationmode = 'ISO-3',
        z = NOC_silver_medals['Medal'],
        text = 'Silver medals',
        colorscale=[[0, '#12c2e9'], [0.5, '#c471ed'], [1, '#f64f59']],
        autocolorscale = False,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            tickprefix = '',
            title = 'Silver medals'),
      ) ]
layout = dict(
    title = 'Silver medals per countries',
    geo = dict(
        showframe = False,
        showcoastlines = True,
        projection = dict(
            type = 'Mercator'
        )
    )
)
fig = dict( data=data, layout=layout )
iplot( fig, validate=False, filename='world-map')
data = [ dict(
        type = 'choropleth',
        locations = NOC_bronze_medals['index'],
        locationmode = 'ISO-3',
        z = NOC_bronze_medals['Medal'],
        text = 'Bronze medals',
        colorscale=[[0, '#12c2e9'], [0.5, '#c471ed'], [1, '#f64f59']],
        autocolorscale = False,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            tickprefix = '',
            title = 'Bronze medals'),
      ) ]
layout = dict(
    title = 'Bronze medals per countries',
    geo = dict(
        showframe = False,
        showcoastlines = True,
        projection = dict(
            type = 'Mercator'
        )
    )
)
fig = dict( data=data, layout=layout )
iplot( fig, validate=False, filename='world-map')
tmp = athlete_events_df.groupby(['NOC'])['ID'].nunique()
df = pd.DataFrame(data={'Athlets': tmp.values}, index=tmp.index).reset_index()

data = [ dict(
        type = 'choropleth',
        locations = df['NOC'],
        locationmode = 'ISO-3',
        z = df['Athlets'],
        text = 'Athlets',
        colorscale=[[0, '#12c2e9'], [0.5, '#c471ed'], [1, '#f64f59']],
        autocolorscale = False,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            tickprefix = '',
            title = 'Athlets'),
      ) ]
layout = dict(
    title = 'Number of athlets since 1896 per countries',
    geo = dict(
        showframe = False,
        showcoastlines = True,
        projection = dict(
            type = 'Mercator'
        )
    )
)
fig = dict( data=data, layout=layout )
iplot( fig, validate=False, filename='world-map')
