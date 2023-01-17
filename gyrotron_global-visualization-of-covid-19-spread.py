import numpy as np 

import pandas as pd 

import plotly as py

import plotly.express as px

import plotly.graph_objs as go

from plotly.subplots import make_subplots

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)
df = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")

df.head()
df = df.rename(columns={'Country/Region':'Country'})

df = df.rename(columns={'ObservationDate':'Date'})

df_countries = df.groupby(['Country', 'Date']).sum().reset_index().sort_values('Date', ascending=False)
df_countries.head()
df_countrydate = df[df['Confirmed']>0]

df_countrydate = df_countrydate.groupby(['Date','Country']).sum().reset_index()

df_countrydate
fig = px.choropleth(df_countrydate, 

                    locations="Country", 

                    locationmode = "country names",

                    color="Confirmed", 

                    hover_name="Country", 

                    animation_frame="Date"

                   )

fig.update_layout(

    title_text = 'Spread of Coronavirus',

    title_x = 0.5,

    geo=dict(

        showframe = False,

        showcoastlines = False,

    ))

    

fig.show()