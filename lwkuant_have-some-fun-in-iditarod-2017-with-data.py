### Load required packages

import numpy as np 

import pandas as pd 

import scipy as sp

import matplotlib.pyplot as plt 

%matplotlib inline 

import seaborn as sns



import plotly as py

import plotly.graph_objs as go

py.offline.init_notebook_mode(connected=True) # run at the start of every ipython notebook so that we can work offline

import warnings

warnings.filterwarnings('ignore')
### load the dataset 

df = pd.read_csv('../input/report.csv', encoding='utf-8')



print('Shape:')

print(df.shape)



print('\nInformation:')

print(df.info())



print('\nSome examples:')

print(df.head())
### Check the NAs

print('Print out the contestant:')

print(df.ix[df['Elapsed Time'].isnull(), :])



print('\nCount of this contestant:')

print(np.sum(df['Name'] == 'Otto Balogh'))



print('\nCheck the count of other contestants:')

print(df['Name'].value_counts().sort_values(ascending=True).head(10))
### Remove the contestant who cannot participate in the competition

df_rem = df.ix[~df['Elapsed Time'].isnull(), :]

print(df_rem.info()) # for checking
### Count of unique contestants

print('Number of unique contestants (excluding one that has been removed):')

print(len(df_rem['Name'].value_counts()))



### Group by the country

con_country = df_rem['Name'].groupby(df_rem['Country'])

print('\nDistributioin of contestants by country:')

print(con_country.nunique().sort_values(ascending=False))
### Visualize



tb = con_country.nunique().sort_values(ascending=False)



trace = go.Bar(

    x = list(tb.index),

    y = list(tb.values),

    marker = dict(

    color = '#FF3333'))



layout = go.Layout(

    title = 'Count of Contestants by Country')



data = [trace]



fig = go.Figure(data=data, layout=layout)

py.offline.iplot(fig)
### check the classes of status 

print(df_rem['Status'].value_counts()) 



### drop the duplicate data to calculate the true number for both classes

ind_con = df_rem.drop_duplicates(['Name'])[['Name', 'Status']].set_index('Name')

tb = ind_con['Status'].value_counts().sort_values(ascending=False)

print('\nTrue distribution of status:')

print(tb)
### Visualize 



trace = go.Bar(

    x = list(tb.index),

    y = list(tb.values),

    width=0.3,

    marker = dict(

    color = '#FF3333'))



layout = go.Layout(

    title = 'Count of Status')



data = [trace]



fig = go.Figure(data=data, layout=layout)

py.offline.iplot(fig)
### Create the table 

group_checkpoint = df_rem.groupby(['Checkpoint'])



checkpoint_tb = df_rem.drop_duplicates(['Checkpoint'])[['Checkpoint', 'Latitude', 'Longitude']].set_index('Checkpoint')

checkpoint_tb = pd.merge(

    checkpoint_tb,

    pd.DataFrame(group_checkpoint['Speed'].mean()),

    right_index = True,

    left_index = True

)



print(checkpoint_tb.head()) # for checking
### Get the level of toughness (measured by the average speed)

checkpoint_tb.fillna(100, inplace=True)



### rank the toughness

checkpoint_tb['Level'] = checkpoint_tb['Speed'].rank(ascending=False)

checkpoint_tb['Level'] = checkpoint_tb['Level'].astype(int)



### Sort the dataframe by the order of checkpoint

cps = df_rem.ix[df['Name'] == 'Mitch Seavey', :]['Checkpoint'].values

ind = np.arange(len(cps))



cps_order = dict(zip(cps, ind))

checkpoint_tb['Checkpoint'] = checkpoint_tb.index

checkpoint_tb['Order'] = checkpoint_tb['Checkpoint'].map(cps_order)

checkpoint_tb = checkpoint_tb.sort_values(['Order'])

print(checkpoint_tb.head()) # for checking
### Visualize

mapbox_access_token = 'pk.eyJ1IjoieG5pcGVyIiwiYSI6ImNqMDR6cXR0aDBoNm4ycWxzcTF2Z3ZxbGsifQ.dAlvq0ZttViD4l3HRbqeYw'



scl = [[0, 'rgb(255, 200, 180)'], [1, 'rgb(255, 0, 0)']]



data = go.Data([    



    go.Scattermapbox(

        lat=list(checkpoint_tb['Latitude'].values),

        lon=list(checkpoint_tb['Longitude'].values),

        mode="lines+markers",

        marker=go.Marker(

            size=15,

            color=list(checkpoint_tb['Level'].values),

                        colorscale=scl,

        cmin=0,

        cmax=checkpoint_tb['Level'].values.max(),

        colorbar=dict()

        ),

        line = go.Line(

            width=1.2,

            color='#444444'

            ),

        text=list(checkpoint_tb.index),

    ),])



layout = go.Layout(

    showlegend=False,

    autosize=False,

    width=800,

    height=600,

    title='The Distribution of Toughness of Route in the Contest',

    hovermode='closest',

    margin=go.Margin(

        l=50,

        r=10,

        b=50,

        t=100,

        pad=2

    ),

    mapbox=dict(

        accesstoken=mapbox_access_token,

        bearing=0,

        center=dict(

            lat=np.mean(checkpoint_tb['Latitude']),

            lon=np.mean(checkpoint_tb['Longitude'])

        ),

        pitch=0,

        zoom=4.5

    ),

)



fig = dict(data=data, layout=layout)

py.offline.iplot(fig, validate=False)
### Determine the ranks of winners 



## create a column for the complete date_time

from datetime import datetime

df_rem['Arrival Date'].fillna('03/06/2017', inplace=True)

df_rem['Arrival Time'].fillna('11:00:00', inplace=True)

df_rem['Arrival_datetime'] = df_rem.ix[:, ['Arrival Date', 'Arrival Time']].apply(lambda x: datetime.strptime(x[0]+'/'+x[1],

                                                                                                             '%m/%d/%Y/%H:%M:%S'),

                                                                                                             axis=1)

## The people who complete the race and ranks

# get the names

names = np.array(df_rem['Name'].value_counts().index)[df_rem['Name'].value_counts().values>=17]



df_com = df_rem.ix[df_rem['Name'].isin(names), :]



# rank the winners

name_group = df_com.groupby(['Name'])

rank_tb = pd.DataFrame(name_group['Arrival_datetime'].max().sort_values())

rank_tb['Rank'] = np.array(range(len(rank_tb)))+1
### Get the number of dogs used when departure for winners and losers



## get the names for winners and losers

win_los_names = list(rank_tb.index)[:3]

win_los_names.extend(list(rank_tb.index)[-3:])



win_los_dogs = []

for name in win_los_names:

    win_los_dogs.append(list(df_com['Departure Dogs'][df_com['Name'] == name])[:-1])



print('The top 3 winners and the last 3 losers: (ordered by the ranks)')

print(win_los_names)
### Get the names of the names of departure checkpoint

checkpoints = list(checkpoint_tb.index)[:-1]
### Visualize 

"""

win_los_dogs_df = pd.DataFrame(columns = win_los_names, index = checkpoints)

for ind, name in enumerate(win_los_names):

    win_los_dogs_df[name] = win_los_dogs[ind]



axes = win_los_dogs_df.plot(alpha=0.8, figsize=[20, 10], fontsize=15, linewidth=2)

axes.set_ylim([5, 20])

axes.set_title('Number of Dogs Used by Winners and Losers', fontsize=20)

axes.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=15)

"""
### Visualize



trace_1 = go.Scatter(

    mode = 'lines',

    x = checkpoints[0],

    y = win_los_dogs[0],

    name = win_los_names[0],

    line=dict(color = '#006400'))



trace_2 = go.Scatter(

    mode = 'lines',

    x = checkpoints[1],

    y = win_los_dogs[1],

    name = win_los_names[1],

    line=dict(color = '	#41A541'))



trace_3 = go.Scatter(

    mode = 'lines',

    x = checkpoints[2],

    y = win_los_dogs[2],

    name = win_los_names[2],

    line=dict(color = '#6DD66D'))



trace_4 = go.Scatter(

    mode = 'lines',

    x = checkpoints[3],

    y = win_los_dogs[3],

    name = win_los_names[3],

    line=dict(color = '#B90000'))



trace_5 = go.Scatter(

    mode = 'lines',

    x = checkpoints[4],

    y = win_los_dogs[4],

    name = win_los_names[4],

    line=dict(color = '#B94646'))



trace_6 = go.Scatter(

    mode = 'lines',

    x = checkpoints[5],

    y = win_los_dogs[5],

    name = win_los_names[5],

    line=dict(color = '#FF8C8C'))



layout = go.Layout(

    title = 'Number of Dogs Used by Winners and Losers')



data = [trace_1, trace_2, trace_3, trace_4, trace_5, trace_6]



fig = go.Figure(data=data, layout=layout)

py.offline.iplot(fig)