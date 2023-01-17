import numpy as np

import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

import numpy as np

import plotly

import plotly.express as px

import plotly.graph_objects as go

import plotly.offline as pyo

import cufflinks as cf

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import folium

import warnings

warnings.filterwarnings('ignore')
pyo.init_notebook_mode(connected= True)

cf.go_offline()
df = pd.read_csv('/kaggle/input/covid19-in-india/covid_19_india.csv')

df.head()
df.describe()
df.drop(['Sno', 'Time', 'ConfirmedIndianNational', 'ConfirmedForeignNational'], axis= 1, inplace= True)

df.rename(columns= {'State/UnionTerritory': 'State'}, inplace= True)

df.head()
df['Active'] = df['Confirmed'] - (df['Cured'] + df['Deaths'])

df.head()
df_state = df[df['Date'] == '12/05/20']

df_state = df_state.groupby('State')[['Cured', 'Deaths', 'Confirmed', 'Active']].sum()

df_state.style.bar(color= '#de1c1c')
df_state.reset_index(inplace= True)
fig= df_state.iplot(kind= 'scatter', x= 'State', y= ['Deaths', 'Cured', 'Confirmed', 'Active'],

               mode= 'lines+markers',

               title= 'State-Wise COVID-19 Analysis',

               xTitle= 'States/Union Territories',

               yTitle= 'No. of Cases'

              )
fig= go.Figure()

fig.add_trace(go.Bar(x= df_state['State'], y= ((df_state['Cured']/df_state['Confirmed'])*100), name= 'Recovery Rate'))

fig.update_layout(title= 'State-Wise Recovery Rates', xaxis_title= 'States', yaxis_title= 'Recovery Rate (Cured Cases / Confirmed Cases)')

fig.show()
def get_cured_state_details(x, state):

    x['Date'] = pd.to_datetime(x['Date'], dayfirst= True)

    x = x[x['State'] == state].drop('State', axis= 1)

    return x

def plot_cured_state(df, state, fig):

    df = get_cured_state_details(df, state)

    fig.add_trace(go.Scatter(x= df['Date'], y= df['Confirmed'], name= 'Confirmed', mode= 'lines+markers'))

    fig.add_trace(go.Scatter(x= df['Date'], y= df['Deaths'], name= 'Deaths', mode= 'lines+markers'))

    fig.add_trace(go.Scatter(x= df['Date'], y= df['Cured'], name= 'Cured', mode= 'lines+markers'))

    fig.update_layout(title= 'Cases in '+state, xaxis_title= 'Dates', yaxis_title= 'No. of Cases')
for state in df_state['State'].unique():

    fig= go.Figure()

    plot_cured_state(df, state, fig)

    fig.show()