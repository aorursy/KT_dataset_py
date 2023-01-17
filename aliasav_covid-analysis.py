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
'''prepare dataset

'''

import plotly.graph_objects as go

import plotly.express as px

import plotly.io as pio

pio.templates.default = "plotly_dark"

from plotly.subplots import make_subplots



cleaned_data = pd.read_csv('../input/corona-virus-report/complete_data_new_format.csv', parse_dates=['Date'])

cleaned_data.head()



cleaned_data.rename(columns={'ObservationDate': 'date', 

                     'Province/State':'state',

                     'Country/Region':'country',

                     'Last Update':'last_updated',

                     'Confirmed': 'confirmed',

                     'Deaths':'deaths',

#                      'Recovered':'recovered'

                    }, inplace=True)



# cases 

# cases = ['confirmed', 'deaths', 'recovered', 'active']

cases = ['confirmed', 'deaths', 'active']



# Active Case = confirmed - deaths - recovered

# cleaned_data['active'] = cleaned_data['confirmed'] - cleaned_data['deaths'] - cleaned_data['recovered']

cleaned_data['active'] = cleaned_data['confirmed'] - cleaned_data['deaths']



# replacing Mainland china with just China

cleaned_data['country'] = cleaned_data['country'].replace('Mainland China', 'China')



# filling missing values 

cleaned_data[['state']] = cleaned_data[['state']].fillna('')

cleaned_data[cases] = cleaned_data[cases].fillna(0)

cleaned_data.rename(columns={'Date':'date'}, inplace=True)



data = cleaned_data



print("External Data")

print(f"Earliest Entry: {data['date'].min()}")

print(f"Last Entry:     {data['date'].max()}")

print(f"Total Days:     {data['date'].max() - data['date'].min()}")
# world wide confirmed and deaths



grouped = data.groupby('date')['date', 'confirmed', 'deaths'].sum().reset_index()



fig = px.line(grouped, x="date", y="confirmed", 

              title="Worldwide Confirmed Cases Over Time")

fig.show()
# india and neighbors analysis



from plotly.subplots import make_subplots



countries = ['India','Pakistan','Bangladesh','Sri Lanka','Afghanistan','China','US','Italy','South Korea','Nepal']

asian_countries = ['India','Pakistan','Bangladesh','Sri Lanka','Afghanistan','Nepal']

groups = {}

for c in countries:

    grouped_df = data[data['country'] == c].reset_index()

    grouped_df_date = grouped_df.groupby('date')['date', 'confirmed', 'deaths'].sum().reset_index()

    groups[c] = grouped_df_date



asian_data = data[data['country'].isin(asian_countries)]

grouped_asian = asian_data.groupby(['date', 'country'])['date', 'confirmed', 'deaths', 'country'].sum().reset_index()



fig = px.line(grouped_asian, x="date", y="confirmed", color='country',

              title="Asian Confirmed Cases Over Time")

fig.show()

fig = px.line(grouped_asian, x="date", y="deaths", color='country',

              title="Asian Deaths Cases Over Time")

fig.show()
# line and area plots



## asian data

# temp = asian_data.groupby('date')['recovered', 'deaths', 'active'].sum().reset_index()

# temp = temp.melt(id_vars="date", value_vars=['recovered', 'deaths', 'active'],

#                  var_name='case', value_name='count')

temp = asian_data.groupby('date')['deaths', 'active'].sum().reset_index()

temp = temp.melt(id_vars="date", value_vars=['deaths', 'active'],

                 var_name='case', value_name='count')

# line plot

fig = px.line(temp, x="date", y="count", color='case',

             title='Cases over time: Line Plot', color_discrete_sequence = ['cyan', 'red', 'orange'])

fig.show()



# area plot

fig = px.area(temp, x="date", y="count", color='case',

             title='Cases over time: Area Plot', color_discrete_sequence = ['cyan', 'red', 'orange'])

fig.show()



## world data

# temp = data.groupby('date')['recovered', 'deaths', 'active'].sum().reset_index()

# temp = temp.melt(id_vars="date", value_vars=['recovered', 'deaths', 'active'],

#                  var_name='case', value_name='count')

temp = data.groupby('date')['deaths', 'active'].sum().reset_index()

temp = temp.melt(id_vars="date", value_vars=['deaths', 'active'],

                 var_name='case', value_name='count')



# line plot

fig = px.line(temp, x="date", y="count", color='case',

             title='Cases over time: Line Plot', color_discrete_sequence = ['cyan', 'red', 'orange'])

fig.show()



# area plot

fig = px.area(temp, x="date", y="count", color='case',

             title='Cases over time: Area Plot', color_discrete_sequence = ['cyan', 'red', 'orange'])

fig.show()
# worldwide spread over time

formated_gdf = data.groupby(['date', 'country'])['confirmed', 'deaths'].max()

formated_gdf = formated_gdf.reset_index()

formated_gdf['date'].fillna(0, inplace=True)

formated_gdf['confirmed'].fillna(0, inplace=True)

formated_gdf['date'] = pd.to_datetime(formated_gdf['date'])

formated_gdf['date'] = formated_gdf['date'].dt.strftime('%m/%d/%Y')

formated_gdf['size'] = formated_gdf['confirmed'].pow(0.3)

formated_gdf['size'].fillna(0, inplace=True)



fig = px.scatter_geo(formated_gdf, locations="country", locationmode='country names', 

                     color="confirmed", size='size', hover_name="country", 

                     range_color= [0, 1500], 

                     projection="natural earth", animation_frame="date", 

                     title='COVID-19: Spread Over Time worldwide', color_continuous_scale="portland")

fig.show()



# asia spread over time

fig = px.scatter_geo(formated_gdf, locations="country", locationmode='country names', 

                     color="confirmed", size='size', hover_name="country", 

                     range_color= [0, 1500], 

                     projection="natural earth", animation_frame="date", scope='asia',

                     title='COVID-19: Spread Over Time Asia', color_continuous_scale="portland")



fig.show()
