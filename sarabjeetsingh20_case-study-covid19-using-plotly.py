# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import plotly.express as px
df=pd.read_csv("../input/casestudy-covid19-analysis/owid-covid-data.csv")
df
df.shape
print("Min and Max dates are: ", df.date.min(), df.date.max())
## Drop rows corresponding to the World
df = df[df.location != 'World']
## Sort df by date
df = df.sort_values(by=['date'])
df_latest = df[df.date == df.date.max()]
print("Date from the last date: ", df_latest.shape)
#df.shape
#df.tail()
df_latest.head()
fig = px.choropleth(df, locations="iso_code",
                    color="new_cases",
                    hover_name="location",
                    animation_frame="date",
                    title = "Daily new COVID cases",
                    color_continuous_scale=px.colors.sequential.PuRd)

fig["layout"].pop("updatemenus")
fig.show()
df['new_date'] = pd.to_datetime(df['date'])
df['Year-Week'] = df['new_date'].dt.strftime('%Y-%U')
df['Year-Week']
fig = px.choropleth(df, locations="iso_code",
                    color="total_cases",
                    hover_name="location", # column to add to hover information
                    animation_frame="Year-Week",
                    title = "Weekly total COVID cases",
                    color_continuous_scale=px.colors.sequential.PuRd)

# fig["layout"].pop("updatemenus")
fig.show()

import pandas as pd 

df_us = pd.read_csv('https://raw.githubusercontent.com/nytimes/covid-19-data/master/live/us-counties.csv')
df_us['new_date'] = pd.to_datetime(df_us['date'])
df_us['Year-Week'] = df_us['new_date'].dt.strftime('%Y-%U')
df_us
df.shape
df_us = df_us.sort_values(by=['county', 'state', 'new_date'])
df_us_week = df_us.groupby(['county', 'state', 'fips', 'Year-Week']).first().reset_index()
df_us_week
df_us_week.head(100)
df_us_week['cases'].max(), df_us_week['cases'].min()
from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)
counties["features"][100]
df_us_week = df_us_week.sort_values(by=['Year-Week'])
fig = px.choropleth(df_us_week, geojson=counties, locations='fips', color='cases',
                           color_continuous_scale=px.colors.sequential.OrRd,
                           title = "Total Weekly Cases by Counties",
                           scope="usa",
                           animation_frame="Year-Week",
                          )
fig["layout"].pop("updatemenus")
fig.show()