import numpy as np

import pandas as pd

import matplotlib

import matplotlib.pyplot as plt

from matplotlib import cm

%matplotlib inline



from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly.graph_objs as go



from mpl_toolkits.basemap import Basemap



import seaborn as sns

sns.set(style="white", palette="muted", color_codes=True)



init_notebook_mode()

df = pd.read_csv("../input/nyc-elevators.csv", low_memory = False)

df.head(2)
def missing_values_table(df): 

    mis_val = df.isnull().sum()

    mis_val_percent = 100 * df.isnull().sum()/len(df)

    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    mis_val_table_ren_columns = mis_val_table.rename(columns = {0 : 'Missing Values', 1 : '% of Total Values'})

    return mis_val_table_ren_columns 

    

missing_values_table(df)
df = df.drop('Unnamed: 26', axis=1)
def group_by_cnt(df, col_nm):

    out = df.groupby(col_nm).size().reset_index(name='Counts').sort_values(['Counts'], ascending = True).reset_index(drop=True)

    out['Percent'] = out['Counts']/sum(out['Counts']) * 100

    out['Text'] = out[col_nm] + ": " + out['Counts'].astype(str)

    return(out)
df['ELEVATOR_TYPE'] = df['Device Type'].str.rstrip(')').str.split('(').str[0]

ele_type = group_by_cnt(df, 'ELEVATOR_TYPE')

ele_type
data = [go.Bar(x=ele_type['Counts'],

               y=ele_type['ELEVATOR_TYPE'], 

               text = ele_type['Text'],

               textposition = 'auto',

               marker=dict(color = 'rgba(55, 128, 191, 0.7)'),

               orientation = 'h')]

layout = dict(

    title='The elevator type in New York',

    xaxis=dict(

        type='log',

        autorange=True,

        showgrid=False,

        zeroline=False,

        showline=False,

        autotick=True,

        ticks='',

        showticklabels=False

    ), 

    yaxis=dict(

        autorange=True,

        showgrid=False,

        zeroline=False,

        showline=False,

        autotick=True,

        ticks='',

    ),

    font = dict( color = "black", size = 10 ),

    autosize = True)

fig = dict(data=data, layout=layout )

iplot(fig, filename='Elevator-type')
ele_status = group_by_cnt(df, 'DV_DEVICE_STATUS_DESCRIPTION')

ele_status
data = [go.Bar(x=ele_status['Counts'],

               y=ele_status['DV_DEVICE_STATUS_DESCRIPTION'], 

               text = ele_status['Text'],

               textposition = 'auto',

               marker=dict(color = 'rgba(55, 128, 191, 0.7)'),

               orientation = 'h')]

layout = dict(

    title='The elevator status in New York',

    xaxis=dict(

        type='log',

        autorange=True,

        showgrid=False,

        zeroline=False,

        showline=False,

        autotick=True,

        ticks='',

        showticklabels=False

    ), 

    yaxis=dict(

        autorange=True,

        showgrid=False,

        zeroline=False,

        showline=False,

        autotick=True,

        ticks='',

    ),

    font = dict( color = "black", size = 10 ),

    autosize = True)

fig = dict(data=data, layout=layout )

iplot(fig, filename='Elevator-status')
x = df.groupby(['ELEVATOR_TYPE', 'DV_DEVICE_STATUS_DESCRIPTION']).size().reset_index(name='Counts')

data = [go.Heatmap(z=x['Counts'], 

                   x=x['ELEVATOR_TYPE'],

                   y=x['DV_DEVICE_STATUS_DESCRIPTION'],

                   colorscale=[[0.0, 'rgb(165,0,38)'], [0.1111111111111111, 'rgb(215,48,39)'], [0.2222222222222222, 'rgb(244,109,67)'], [0.3333333333333333, 'rgb(253,174,97)'], [0.4444444444444444, 'rgb(254,224,144)'], [0.5555555555555556, 'rgb(224,243,248)'], [0.6666666666666666, 'rgb(171,217,233)'], [0.7777777777777778, 'rgb(116,173,209)'], [0.8888888888888888, 'rgb(69,117,180)'], [1.0, 'rgb(49,54,149)']])

       ]



layout = go.Layout(title='Operating status of each elevator type in NY',

                   xaxis = dict(ticks='', nticks=45),

                   yaxis = dict(ticks='' )

                  )



fig = go.Figure(data=data, layout=layout)

iplot(fig, filename='datetime-heatmap')
west, south, east, north = -74.03, 40.63, -73.77, 40.85

tmp = df

tmp = tmp[(tmp.LATITUDE> south) & (tmp.LONGITUDE < north)]

tmp = tmp[(tmp.LATITUDE> west) & (tmp.LONGITUDE < east)]
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(20,10))

tmp.loc[(tmp['Device Status'] == 'A') & 

        (tmp['ELEVATOR_TYPE'] == 'Passenger Elevator ')].plot(kind='scatter', 

                                                              x='LONGITUDE', 

                                                              y='LATITUDE',

                                                              color='black', s=.05, alpha=.75,

                                                              subplots=True, ax=ax1)

ax1.set_title("Active Passenger Elevator Only", fontsize=18)

ax1.set_facecolor('#f9f9f9') 



tmp.loc[(tmp['Device Status'] == 'A') & 

        (tmp['ELEVATOR_TYPE'] == 'Escalator ')].plot(kind='scatter', 

                                                   x='LONGITUDE', 

                                                   y='LATITUDE',

                                                   color='black', s=.5, alpha=.75,

                                                    subplots=True, ax=ax2)

ax2.set_title("Active Escalator Only", fontsize=18)

ax2.set_facecolor('#f9f9f9') 





plt.show()
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(20,10))



tmp.loc[(tmp['Device Status'] == 'A') & 

        ((tmp['ELEVATOR_TYPE'] == 'Private Elevator ') | 

         (tmp['ELEVATOR_TYPE'] == 'Public Elevator '))].plot(kind='scatter', 

                                                              x='LONGITUDE', 

                                                              y='LATITUDE',

                                                              color='black', 

                                                              subplots=True, ax=ax1)

ax1.set_title("Active Public and Private Elevators", fontsize=18)

ax1.set_facecolor('#f9f9f9') 



tmp.loc[(tmp['Device Status'] == 'A') & 

        (tmp['ELEVATOR_TYPE'] == 'Handicap Lift ')].plot(kind='scatter', 

                                                         x='LONGITUDE', 

                                                         y='LATITUDE',

                                                         color='black',

                                                         subplots=True, ax=ax2)

ax2.set_title("Active Handicap Lift Only", fontsize=18)

ax2.set_facecolor('#f9f9f9')



plt.show();