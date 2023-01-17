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
df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')
df['Province/State'].fillna('', inplace=True)
df.update(df[df.columns[4:]].fillna(0))
df.insert(4, 'Location', df["Country/Region"] + ' ' + df["Province/State"])
print(df.head())
print(df.loc[df['Country/Region'] == 'France'])
print(df.loc[df['Province/State'] == 'Hubei']['3/14/20'])
import plotly.express as px

days = [day for day in df.columns[5:]]
df_total = pd.melt(df, id_vars=["Province/State", "Country/Region", "Location", "Lat", "Long"], 
             value_vars=days)
print(df_total.head())
px.scatter_geo(df_total, lat='Lat', lon='Long', animation_frame="variable", animation_group="Location", size="value", projection="natural earth", size_max=55, hover_name="Location", title="Animated world map of total Covid-19 confirmed cases")
df_copy = df.copy()
df_copy[df_copy.columns[5:]] = df_copy[df_copy.columns[5:]].apply(lambda x: x - x.shift(1), axis=1)
df_new_cases = pd.concat([df[df.columns[:6]], df_copy[df_copy.columns[6:]]], axis=1)
print(df_new_cases.head())
df_new = pd.melt(df_new_cases, id_vars=["Province/State", "Country/Region", "Location", "Lat", "Long"], 
             value_vars=days)
print(df_new.head())
df_new.loc[df_new['value'] < 0]
df.loc[df['Country/Region'] == 'Guyana']
df_recov = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')
df_recov['Province/State'].fillna('', inplace=True)
df_recov.update(df_recov[df_recov.columns[4:]].fillna(0))
df_recov.insert(4, 'Location', df_recov["Country/Region"] + ' ' + df_recov["Province/State"])
df_recov.loc[df_recov['Country/Region'] == 'Guyana']
df_new['value'][df_new['value'] < 0] = 0
px.scatter_geo(df_new, lat='Lat', lon='Long', animation_frame="variable", animation_group="Location", size="value", projection="natural earth", size_max=55, hover_name="Location", title="Animated world map of new Covid-19 confirmed cases per day")