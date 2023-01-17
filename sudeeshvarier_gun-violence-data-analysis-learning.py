# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import plotly.plotly as py1
import plotly.graph_objs as go
import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
from plotly import tools
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.core.display import display, HTML, Javascript
import IPython.display

import cufflinks as cf
cf.go_offline()
df = pd.read_csv('../input/gun-violence-data_01-2013_03-2018.csv')
df.head()
df.describe()
statedistribution = df.groupby(['state']).size()

state_distribution_category = pd.DataFrame({'labels': statedistribution.index,
                                   'values': statedistribution.values })

state_distribution_category.iplot(kind='pie',
                         labels='labels',
                         values='values', 
                         title='Grouped by state where crime is reported')
states_df = df['state'].value_counts()

statesdf = pd.DataFrame()
statesdf['state'] = states_df.index
statesdf['counts'] = states_df.values

scl = [[0.0, 'rgb(130,240,247)'],[0.2, 'rgb(130,218,235)'],[0.4, 'rgb(130,189,220)'],\
            [0.6, 'rgb(130,154,200)'],[0.8, 'rgb(130,107,177)'],[1.0, 'rgb(130,39,143)']]

state_to_code = {'District of Columbia' : 'dc','Mississippi': 'MS', 'Oklahoma': 'OK', 'Delaware': 'DE', 'Minnesota': 'MN', 'Illinois': 'IL', 'Arkansas': 'AR', 'New Mexico': 'NM', 'Indiana': 'IN', 'Maryland': 'MD', 'Louisiana': 'LA', 'Idaho': 'ID', 'Wyoming': 'WY', 'Tennessee': 'TN', 'Arizona': 'AZ', 'Iowa': 'IA', 'Michigan': 'MI', 'Kansas': 'KS', 'Utah': 'UT', 'Virginia': 'VA', 'Oregon': 'OR', 'Connecticut': 'CT', 'Montana': 'MT', 'California': 'CA', 'Massachusetts': 'MA', 'West Virginia': 'WV', 'South Carolina': 'SC', 'New Hampshire': 'NH', 'Wisconsin': 'WI', 'Vermont': 'VT', 'Georgia': 'GA', 'North Dakota': 'ND', 'Pennsylvania': 'PA', 'Florida': 'FL', 'Alaska': 'AK', 'Kentucky': 'KY', 'Hawaii': 'HI', 'Nebraska': 'NE', 'Missouri': 'MO', 'Ohio': 'OH', 'Alabama': 'AL', 'Rhode Island': 'RI', 'South Dakota': 'SD', 'Colorado': 'CO', 'New Jersey': 'NJ', 'Washington': 'WA', 'North Carolina': 'NC', 'New York': 'NY', 'Texas': 'TX', 'Nevada': 'NV', 'Maine': 'ME'}
statesdf['state_code'] = statesdf['state'].apply(lambda x : state_to_code[x])

data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = statesdf['state_code'],
        z = statesdf['counts'],
        locationmode = 'USA-states',
        text = statesdf['state'],
        marker = dict(
            line = dict (
                color = 'rgb(205,205,205)',
                width = 2
            ) ),
        colorbar = dict(
            title = "Gun Violence Incidents")
        ) ]

layout = dict(
        title = 'State wise number of Gun Violence Incidents',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(66, 165, 245)'),
             )
    
fig = dict( data=data, layout=layout )
iplot( fig, filename='d3-cloropleth-map' )
df['month'] = pd.DatetimeIndex(df['date']).month
monthcount = df['month'].value_counts()


sns.countplot(x = 'month', data=df, palette="husl")

df['gun_type_parsed'] = df['gun_type'].fillna('0:Unknown')
gt = df.groupby(by=['gun_type_parsed']).agg({'n_killed': 'sum', 'n_injured': 'sum', 'state': 'count'}).reset_index().rename(columns={'state':'count'})

results = {}
for i, each in gt.iterrows():
    guntypes = each['gun_type_parsed'].split("||")
    for guntype in guntypes:
        if "Unknown" in guntype:
            continue
        guntype = guntype.replace("::",":").replace("|1","")
        gtype = guntype.split(":")[1]
        if gtype not in results: 
            results[gtype] = {'killed' : 0, 'injured' : 0, 'used' : 0}
        results[gtype]['killed'] += each['n_killed']
        results[gtype]['injured'] +=  each['n_injured']
        results[gtype]['used'] +=  each['count']

gun_names = list(results.keys())
used = [each['used'] for each in list(results.values())]
killed = [each['killed'] for each in list(results.values())]
injured = [each['injured'] for each in list(results.values())]
affected = []
for i, x in enumerate(used):
    affected.append((killed[i] + injured[i]) / x)

sample = go.Bar(x=gun_names, y=used, name='SF Zoo', orientation = 'v',
    marker = dict(color = '#4b42f4', 
        line = dict(color = '#4b42f4', width = 1) ))
data = [sample]
layout = dict(height=400, title='Gun Type Distribution', legend=dict(orientation="h"));
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='marker-h-bar')