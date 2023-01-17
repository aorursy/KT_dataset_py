#Importing Relevant Libraries

import numpy as np 

import pandas as pd 

import plotly as py

import seaborn as sns

import plotly.express as px

import plotly.graph_objs as go

from plotly.subplots import make_subplots

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)



# Input data files are available in the "../input/" directory.



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Read Data

covid_data = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv")
#Showing first rows of data

covid_data.head(3)
covid_data.info()
#let's rename columnw of Country/region and Observations for simplicity



covid_data = covid_data.rename(columns={'Country/Region':'Country'})

covid_data= covid_data.rename(columns={'ObservationDate':'Date'})
# Manipulating Dataframe

covid_countries = covid_data.groupby(['Country', 'Date']).sum().reset_index().sort_values('Date', ascending=False)

covid_countries = covid_countries.drop_duplicates(subset = ['Country'])

covid_countries = covid_countries[covid_countries['Confirmed']>0]
# Create the Choropleth ....Static

fig = go.Figure(data=go.Choropleth(

    locations = covid_countries['Country'],

    locationmode = 'country names',

    z = covid_countries['Confirmed'],

    colorscale = 'Reds',

    marker_line_color = 'black',

    marker_line_width = 0.5,

))
fig.update_layout(

    title_text = 'Confirmed Cases By April 12nd, 2020',

    title_x = 0.5,

    geo=dict(

        showframe = False,

        showcoastlines = False,

        projection_type = 'equirectangular'

    )

)
# Manipulating the original dataframe

covid_countrydate = covid_data[covid_data['Confirmed']>0]

covid_countrydate = covid_countrydate.groupby(['Date','Country']).sum().reset_index()

covid_countrydate.head(3)
# Creatinge animated visualization

fig = px.choropleth(covid_countrydate, 

                    locations="Country", 

                    locationmode = "country names",

                    color="Confirmed", 

                    hover_name="Country", 

                    animation_frame="Date"

                   )

fig.update_layout(

    title_text = 'Global Spread of Coronavirus',

    title_x = 0.5,

    geo=dict(

        showframe = False,

        showcoastlines = False,

    ))

    

fig.show()
#Plotting Deaths by date



# Manipulating the original dataframe, deaths>=0 to show all countries with Zeros cases

covid_deathsdate = covid_data[covid_data['Deaths']>=0]

covid_deathsdate = covid_deathsdate.groupby(['Date','Country']).sum().reset_index()

#covid_countrydate

covid_deathsdate.head(3)
fig_deaths = px.choropleth(covid_deathsdate, 

                    locations="Country", 

                    locationmode = "country names",

                    color="Deaths", 

                    hover_name="Country", 

                    animation_frame="Date"

                   )
fig_deaths.update_layout(

    title_text = 'Number of Deaths As Of April 12nd',

    title_x = 0.5,

    geo=dict(

        showframe = False,

        showcoastlines = False,

    ))

    

fig_deaths.show()