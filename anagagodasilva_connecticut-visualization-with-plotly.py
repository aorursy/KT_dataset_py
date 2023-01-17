# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.express as px #graphic library 



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("/kaggle/input/us-accidents/US_Accidents_June20.csv").query("State=='CT'")
df.head(5)
fig = px.density_mapbox(df, lat='Start_Lat', lon='Start_Lng', radius=5, zoom=7,

                        color_continuous_scale=px.colors.sequential.YlOrRd,

                        mapbox_style="stamen-terrain")

fig.update_layout(

        title = 'Mapbox Density Heatmap - Accidents',

)

fig.show()
df['Start_Time']
df['Start_year']=pd.to_datetime(df['Start_Time'], format='%Y-%m-%d %H:%M:%S').dt.year

df['Start_month']=pd.to_datetime(df['Start_Time'], format='%Y-%m-%d %H:%M:%S').dt.month

df['Start_day']=pd.to_datetime(df['Start_Time'], format='%Y-%m-%d %H:%M:%S').dt.day

df['Start_hour']=pd.to_datetime(df['Start_Time'], format='%Y-%m-%d %H:%M:%S').dt.hour
df=df.sort_values('Start_hour')

fig = px.density_mapbox(df, lat='Start_Lat', lon='Start_Lng', radius=5, zoom=7,

                        color_continuous_scale=px.colors.sequential.YlOrRd,

                        mapbox_style="stamen-terrain", animation_frame='Start_hour')

fig.update_layout(

        title = 'Mapbox Hourly Density Heatmap - Connecticut - car accidents',

)

fig.show()

df=df.sort_values('Start_month')

fig = px.density_mapbox(df, lat='Start_Lat', lon='Start_Lng', radius=5, zoom=7,

                        color_continuous_scale=px.colors.sequential.YlOrRd,

                        mapbox_style="stamen-terrain", animation_frame='Start_month')

fig.update_layout(

        title = 'Mapbox Monthly Density Heatmap - Connecticut - car accidents',

)

fig.show()
df=df.sort_values('Start_year')

fig = px.density_mapbox(df, lat='Start_Lat', lon='Start_Lng', radius=5, zoom=7,

                        color_continuous_scale=px.colors.sequential.YlOrRd,

                        mapbox_style="stamen-terrain", animation_frame='Start_year')

fig.update_layout(

        title = 'Mapbox Yearly Density Heatmap - Connecticut - car accidents',

)

fig.show()