import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
covid_effected = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')

covid_effected.head(15)
covid_effected.shape
covid_effected.info()
covid_effected['Country/Region'].value_counts()
confirm_df= covid_effected.groupby('Country/Region').max().sort_values(by='Confirmed', ascending=False)[:10]

confirm_df
China_df = covid_effected[covid_effected['Country/Region'] == 'Mainland China'].copy()

China_df 
import plotly.graph_objects as go

def plot_aggregate_metrics(df, fig=None):

    if fig is None:

        fig = go.Figure()

    fig.update_layout(template='plotly_dark')

    fig.add_trace(go.Scatter(x=df['ObservationDate'], 

                             y=df['Confirmed'],

                             mode='lines+markers',

                             name='Confirmed',

                             line=dict(color='Yellow', width=2)

                            ))

    fig.add_trace(go.Scatter(x=df['ObservationDate'], 

                             y=df['Deaths'],

                             mode='lines+markers',

                             name='Deaths',

                             line=dict(color='Red', width=2)

                            ))

    fig.add_trace(go.Scatter(x=df['ObservationDate'], 

                             y=df['Recovered'],

                             mode='lines+markers',

                             name='Recoveries',

                             line=dict(color='Green', width=2)

                            ))

    return fig
plot_aggregate_metrics(China_df).show()
Iran_df = covid_effected[covid_effected['Country/Region'] == 'Iran'].copy()

Iran_df 
plot_aggregate_metrics(Iran_df).show()
Italy_df = covid_effected[covid_effected['Country/Region'] == 'Italy'].copy()

Italy_df 
plot_aggregate_metrics(Italy_df).show()
Spain_df = covid_effected[covid_effected['Country/Region'] == 'Spain'].copy()

Spain_df 
plot_aggregate_metrics(Spain_df).show()
us_df = covid_effected[covid_effected['Country/Region'] == 'US'].copy()

us_df 
plot_aggregate_metrics(us_df).show()
state_China_df= China_df.groupby('Province/State').max().sort_values(by='Confirmed', ascending=False)[:10]

state_China_df
Hubei_df = China_df[China_df['Province/State'] == 'Hubei'].copy()

Hubei_df 
plot_aggregate_metrics(Hubei_df).show()