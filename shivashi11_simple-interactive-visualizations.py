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
import pandas as pd

import numpy as np

from datetime import date, datetime, timedelta

import time

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('fivethirtyeight')



# for interactive visualizations

import plotly.offline as py

from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objs as go

from plotly import tools

init_notebook_mode(connected = True)

import plotly.figure_factory as ff
from IPython.core.display import Image, display

display(Image('https://i.imgur.com/0N9ktSe.png', width=700, unconfined=True))
df = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv",)



df.rename(columns={'Last Update': 'LastUpdate',

                   'Province/State': 'PS'},

         inplace=True)

df['Date'] = pd.to_datetime(df['Date']).dt.date



virus_cols=['Confirmed', 'Deaths', 'Recovered']



df = df[df[virus_cols].sum(axis=1)!=0]



df['Country'] = np.where(df['Country']=='Mainland China', 'China', df['Country'])

df.dropna(inplace=True)



usecols=['Province/State', 'Country/Region', 'Lat', 'Long']

path= '/kaggle/input/novel-corona-virus-2019-dataset/time_series_2019_ncov_'

csvs=['confirmed.csv', 'deaths.csv', 'recovered.csv']

coords_df = pd.concat([pd.read_csv(path + csv, usecols=usecols) for csv in csvs])

coords_df.rename(columns={'Country/Region': 'Country',

                          'Province/State': 'PS'}, 

                inplace=True)

coords_df['Country'] = np.where(coords_df['Country']=='Mainland China', 'China', coords_df['Country'])

coords_df = coords_df.drop_duplicates()

df = pd.merge(df, coords_df, on=['Country', 'PS'], how='left')
df = df.groupby(['PS', 'Country', 'Date']).agg({'Confirmed': 'sum',

                                                'Deaths': 'sum',

                                                'Recovered': 'sum',

                                                'Lat': 'max',

                                                'Long': 'max'}).reset_index()
df['Date'] = df['Date'].apply(lambda x: str(x))

# adjusting sizes so that they become visible

df['New_Confirmed'] = df['Confirmed']**(1/2.7)+3
import plotly.express as px



px.set_mapbox_access_token('pk.eyJ1IjoiMWFkaXR5YTEiLCJhIjoiY2s2NWZmYTRzMGFmdzNrbzRrMnB4eGt5YiJ9._nTBIf6oi7v22jqg5XT9Xw')



fig = px.scatter_geo(df, lat="Lat", lon="Long", size = 'New_Confirmed',color = 'Confirmed',projection = 'natural earth',

                        opacity=1, size_max=15, text = 'PS', range_color = [0, 30000],

                       animation_frame = 'Date'

                        )

fig.show()
fig = px.scatter_mapbox(df, lat="Lat", lon="Long", size = 'New_Confirmed',color = 'Confirmed',

                        opacity=1, size_max=15, text = 'PS', range_color = [0, 30000],

                       animation_frame = 'Date', center = {'lat' : 30, 'lon' : 112}, zoom = 2

                        )

fig.show()
data = df.groupby(['PS','Date']).agg({'Confirmed': 'sum',

                                                'Deaths': 'sum',

                                                'Recovered': 'sum',

                                                }).reset_index()

fig = px.bar(data[data['PS']!='Hubei'].sort_values('Confirmed'), y = 'PS', x = 'Confirmed', 

             orientation = 'h', animation_frame='Date', width = 800, height = 900, )

fig.show()
fig = px.scatter(df[df['PS']!='Hubei'], x = 'Confirmed', y = 'Deaths', size = 'New_Confirmed', color = 'PS',

                animation_frame = 'Date', range_y = [0,8], range_x = [0, 1500])



fig.show()
data = df.groupby(['PS']).agg({'Confirmed': 'sum',

                                                'Deaths': 'sum',

                                                'Recovered': 'sum',

                                                }).reset_index()
fig = px.bar(data[data['PS']!='Hubei'].sort_values('Confirmed',).tail(30), y = 'PS', x = 'Confirmed', 

             orientation = 'h', width = 800, height = 600, )

fig.show()
fig = px.bar(data[data['PS']!='Hubei'].sort_values('Deaths',).tail(30), y = 'PS', x = 'Deaths', 

             orientation = 'h', width = 800, height = 600, )

fig.show()
fig = px.bar(data[data['PS']!='Hubei'].sort_values('Recovered').tail(30), y = 'PS', x = 'Recovered', 

             orientation = 'h', width = 800, height = 600, )

fig.show()
from IPython.core.display import Image, display

display(Image('https://i.imgur.com/Tj7wc0f.jpg', width=700, unconfined=True))