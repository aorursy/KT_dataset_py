# This Python 3 environment comes with many helpful analytics libraries installed

# Loading datasets required for analysis



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style="white", color_codes=True)

import warnings # current version of seaborn generates a bunch of warnings that we'll ignore

warnings.filterwarnings("ignore")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/"))
full_table = pd.read_csv('../input/corona-virus-report/covid_19_clean_complete.csv', 

                         parse_dates=['Date'])

full_table.head()
# Defining COVID-19 cases as per classifications 

cases = ['Confirmed', 'Deaths', 'Recovered', 'Active']



# Defining Active Case: Active Case = confirmed - deaths - recovered

full_table['Active'] = full_table['Confirmed'] - full_table['Deaths'] - full_table['Recovered']



# Renaming Mainland china as China in the data table

full_table['Country/Region'] = full_table['Country/Region'].replace('Mainland China', 'China')



# filling missing values 

full_table[['Province/State']] = full_table[['Province/State']].fillna('')

full_table[cases] = full_table[cases].fillna(0)



# cases in the ships

ship = full_table[full_table['Province/State'].str.contains('Grand Princess')|full_table['Country/Region'].str.contains('Cruise Ship')]



# china and the row

china = full_table[full_table['Country/Region']=='China']

row = full_table[full_table['Country/Region']!='China']



# latest

full_latest = full_table[full_table['Date'] == max(full_table['Date'])].reset_index()

china_latest = full_latest[full_latest['Country/Region']=='China']

row_latest = full_latest[full_latest['Country/Region']!='China']



# latest condensed

full_latest_grouped = full_latest.groupby('Country/Region')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

china_latest_grouped = china_latest.groupby('Province/State')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

row_latest_grouped = row_latest.groupby('Country/Region')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()
temp = full_table.groupby(['Country/Region', 'Province/State'])['Confirmed', 'Deaths', 'Recovered', 'Active'].max()
temp = full_table.groupby('Date')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

temp = temp[temp['Date']==max(temp['Date'])].reset_index(drop=True)

temp.style.background_gradient(cmap='Pastel1')
temp_f = full_latest_grouped.sort_values(by='Confirmed', ascending=False)

temp_f = temp_f.reset_index(drop=True)

temp_f.style.background_gradient(cmap='Reds')
temp_f.head(10)
import plotly as py

import plotly.graph_objects as go

import pandas as pd

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)    #THIS LINE IS MOST IMPORTANT AS THIS WILL DISPLAY PLOT ON 

#NOTEBOOK WHILE KERNEL IS RUNNING



#Time Series plot for knwoing the spread



fig = go.Figure()

fig.add_trace(go.Scatter(

                x=full_table.Date,

                y=full_table['Confirmed'],

                name="Confirmed",

                line_color='deepskyblue',

                opacity=0.8))



fig.add_trace(go.Scatter(

                x=full_table.Date,

                y=full_table['Recovered'],

                name="Recovered",

                line_color='dimgray',

                opacity=0.8))

fig.update_layout(title_text='Time Series with Rangeslider',

                  xaxis_rangeslider_visible=True)

py.offline.iplot(fig)
import plotly.offline as py

py.init_notebook_mode(connected=True)



# Calculating the count of confirmed cases by country



countries = np.unique(temp_f['Country/Region'])

mean_conf = []

for country in countries:

    mean_conf.append(temp_f[temp_f['Country/Region'] == country]['Confirmed'].sum())

    

# Building the dataframe



    data = [ dict(

        type = 'choropleth',

        locations = countries,

        z = mean_conf,

        locationmode = 'country names',

        text = countries,

        marker = dict(

            line = dict(color = 'rgb(0,0,0)', width = 1)),

            colorbar = dict(autotick = True, tickprefix = '', 

            title = 'Count')

            )

       ]

    

# Building the visual



    layout = dict(

    title = 'COVID-19 Confirmed Cases',

    geo = dict(

        showframe = False,

        showocean = True,

        oceancolor = 'rgb(0,255,255)',

        projection = dict(

        type = 'orthographic',

            rotation = dict(

                    lon = 60,

                    lat = 10),

        ),

        lonaxis =  dict(

                showgrid = True,

                gridcolor = 'rgb(102, 102, 102)'

            ),

        lataxis = dict(

                showgrid = True,

                gridcolor = 'rgb(102, 102, 102)'

                )

            ),

        )



fig = dict(data=data, layout=layout)

py.iplot(fig, validate=False, filename='worldmap')
import pandas as pd

global_temp_country = pd.read_csv("../input/climate-change-earth-surface-temperature-data/GlobalLandTemperaturesByCountry.csv")
global_temp_country.head()
import plotly.offline as py

py.init_notebook_mode(connected=True)



## Removing the duplicates



global_temp_country_clear = global_temp_country[~global_temp_country['Country'].isin(

    ['Denmark', 'Antarctica', 'France', 'Europe', 'Netherlands',

     'United Kingdom', 'Africa', 'South America'])]



global_temp_country_clear = global_temp_country_clear.replace(

   ['Denmark (Europe)', 'France (Europe)', 'Netherlands (Europe)', 'United Kingdom (Europe)'],

   ['Denmark', 'France', 'Netherlands', 'United Kingdom'])



#Calculating average temperature by country



countries = np.unique(global_temp_country_clear['Country'])

mean_temp = []

for country in countries:

    mean_temp.append(global_temp_country_clear[global_temp_country_clear['Country'] == 

                                               country]['AverageTemperature'].mean())



# Building the data frame

    

data = [ dict(

        type = 'choropleth',

        locations = countries,

        z = mean_temp,

        locationmode = 'country names',

        text = countries,

        marker = dict(

            line = dict(color = 'rgb(0,0,0)', width = 1)),

            colorbar = dict(autotick = True, tickprefix = '', 

            title = '# Average\nTemperature,\n°C')

            )

       ]



# Building the visual



layout = dict(

    title = 'GLOBAL AVERAGE LAND TEMPERATURES',

    geo = dict(

        showframe = False,

        showocean = True,

        oceancolor = 'rgb(0,255,255)',

        projection = dict(

        type = 'orthographic',

            rotation = dict(

                    lon = 60,

                    lat = 10),

        ),

        lonaxis =  dict(

                showgrid = True,

                gridcolor = 'rgb(102, 102, 102)'

            ),

        lataxis = dict(

                showgrid = True,

                gridcolor = 'rgb(102, 102, 102)'

                )

            ),

        )



fig = dict(data=data, layout=layout)

py.iplot(fig, validate=False, filename='worldmap')
import plotly.express as px

import plotly.offline as py

py.init_notebook_mode(connected=True)

formated_gdf = full_table.groupby(['Date', 'Country/Region'])['Confirmed', 'Deaths', 'Recovered'].max()

formated_gdf = formated_gdf.reset_index()

formated_gdf['Date'] = pd.to_datetime(formated_gdf['Date'])

formated_gdf['Date'] = formated_gdf['Date'].dt.strftime('%m/%d/%Y')

formated_gdf['size'] = formated_gdf['Confirmed'].pow(0.3)



fig = px.scatter_geo(formated_gdf, locations="Country/Region", locationmode='country names', 

                     color="Confirmed", size='size', hover_name="Country/Region", 

                     range_color= [0, max(formated_gdf['Confirmed'])+2], 

                     projection="natural earth", animation_frame="Date", 

                     title='Progression of spread of COVID-19')

fig.update(layout_coloraxis_showscale=False)

py.offline.iplot(fig)