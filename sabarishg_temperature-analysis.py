# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 





import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import plotly.graph_objs as go

import plotly.offline as py

import plotly.express as px

# Any results you write to the current directory are saved as output.
temp_by_country = pd.read_csv('/kaggle/input/climate-change-earth-surface-temperature-data/GlobalLandTemperaturesByCountry.csv')
temp_by_country['dt'] = pd.to_datetime(temp_by_country['dt'])

temp_by_country['year'] = temp_by_country['dt'].dt.year

temp_by_country['month'] = temp_by_country['dt'].dt.month_name()

temp_by_country['day'] = temp_by_country['dt'].dt.day
india_temp = temp_by_country[temp_by_country['Country'] == 'India']

yearly_avg_temperature = pd.DataFrame(india_temp.groupby('year')['AverageTemperature'].mean()).reset_index()

yearly_avg_temperature = yearly_avg_temperature[yearly_avg_temperature['year'] > 1900]
plt.figure(figsize=(10,10))

plt.plot(yearly_avg_temperature['year'], yearly_avg_temperature['AverageTemperature'], label = 'Average Temperature')

plt.legend()
temp_by_country
country_subset = temp_by_country[(temp_by_country['Country'] == 'India') | (temp_by_country['Country'] == 'Australia')]

country_subset = country_subset.groupby(['Country','year'], as_index = False)['AverageTemperature'].mean()

country_subset = country_subset[country_subset['year'] > 1900]

fig = px.line(country_subset, x="year", y="AverageTemperature", color='Country')

fig.show()
fig = px.scatter(country_subset, x="year", y="AverageTemperature", facet_col="Country", color="Country", trendline="ols")

fig.show()
months = ["January", "February", "March", "April", "May", "June", 

          "July", "August", "September", "October", "November", "December"]

temp_by_country['months'] = pd.Categorical(temp_by_country['month'], categories=months, ordered=True)

#temp_by_country[(temp_by_country['Country'] =='India') & (temp_by_country['year'] > 1900)].groupby(['year','months'],as_index = False)['AverageTemperature'].mean()
cities_data = pd.read_csv('/kaggle/input/climate-change-earth-surface-temperature-data/GlobalLandTemperaturesByCity.csv')

cities_data = cities_data[(cities_data['City'] =='Bombay') | 

                          (cities_data['City'] == 'Delhi') | 

                          (cities_data['City']=='Bangalore') | 

                          (cities_data['City'] == 'Pune') |

                          (cities_data['City'] == 'Madras')  |

                          (cities_data['City']== 'Calcutta')]



cities_data = cities_data[cities_data['dt'] > '1920-01-01']

cities_data.head()
cities_data['dt'] = pd.to_datetime(cities_data['dt'])

cities_data['year'] = cities_data['dt'].dt.year

cities_data['month'] = cities_data['dt'].dt.month_name()

cities_data['day'] = cities_data['dt'].dt.day



months = ["January", "February", "March", "April", "May", "June", 

          "July", "August", "September", "October", "November", "December"]

cities_data['months'] = pd.Categorical(cities_data['month'], categories=months, ordered=True)

#temp_by_country[(temp_by_country['Country'] =='India') & (temp_by_country['year'] > 1900)].groupby(['year','months'],as_index = False)['AverageTemperature'].mean()
htmap = cities_data.groupby(['City','months'], as_index = False)['AverageTemperature'].mean()

#htmap.head()



trace = go.Heatmap(z=htmap['AverageTemperature'],

                   x=htmap['months'],

                   y=htmap['City']

                  )

data=[trace]

layout = go.Layout(

    title='Average Temperature Of Major Cities By Month',

)

fig = go.Figure(data=data, layout=layout)

fig.show()
recent_temperatures = temp_by_country[(temp_by_country['year'] > 1950)]

#recent_temperatures['dt'] = pd.to_datetime(recent_temperatures['dt'])

recent_temperatures.head()



# Creating the visualization for country by country

fig = px.choropleth(recent_temperatures, 

                    locations="Country", 

                    locationmode = "country names",

                    color="AverageTemperature", 

                    hover_name="Country", 

                    animation_frame="year")

fig.update_layout(

    title_text = 'Average Temperature',

    title_x = 0.5,

    geo=dict(

        showframe = False,

        showcoastlines = False,

    ))

    

#fig.show()

#fig.write_html("kaggle/working/sampleplot.html")

py.iplot(fig, filename='/kaggle/working/sampleplot.html')