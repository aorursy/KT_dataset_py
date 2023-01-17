import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import plotly.graph_objects as go

import plotly as px
df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',parse_dates=['Last Update'])

df.rename(columns={'ObservationDate':'Date', 'Country/Region':'Country'}, inplace=True)



conf = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")

recov = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv")

death = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv")
conf.rename(columns={'Country/Region':'Country'}, inplace=True)

recov.rename(columns={'Country/Region':'Country'}, inplace=True)

death.rename(columns={'Country/Region':'Country'}, inplace=True)
df.head()
conf.head()
death.head()
recov.head()
fig = go.Figure()

fig.add_trace(go.Bar(x=df['Date'],

                y=df['Confirmed'],

                name='Confirmed',

                marker_color='blue'

                ))



fig.update_layout(

    title='Worldwide Corona Virus Cases - Confirmed(Bar Chart)',

    xaxis_tickfont_size=14,

    yaxis=dict(

        title='Number of Cases',

        titlefont_size=16,

        tickfont_size=14,

    )

)

fig.show()
fig = go.Figure()

fig.add_trace(go.Bar(x=df['Date'],

                y=df['Deaths'],

                name='Deaths',

                marker_color='red'

                ))



fig.update_layout(

    title='Worldwide Corona Virus Cases - Deaths(Bar Chart)',

    xaxis_tickfont_size=14,

    yaxis=dict(

        title='Number of Cases',

        titlefont_size=16,

        tickfont_size=14,

    )

)

fig.show()