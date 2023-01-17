import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import seaborn as sns

import matplotlib.pyplot as plt
covid_india_cases = pd.read_csv("/kaggle/input/covid19-analysis/Covid cases in India.csv")
covid_india_cases.head()
del covid_india_cases["S. No."]
covid_india_cases["Name of State / UT"].unique()
covid_india_cases['Total cases'] = covid_india_cases['Total Confirmed cases (Indian National)'] + covid_india_cases['Total Confirmed cases ( Foreign National )'] 

covid_india_cases['Active cases'] = covid_india_cases['Total cases'] - (covid_india_cases['Cured/Discharged/Migrated'] + covid_india_cases['Deaths'])


def highlight_max(s):

    is_max = s == s.max()

    return ['background-color: blue' if v else '' for v in is_max]

#df.style.apply(highlight_max,subset=['Total Confirmed cases (Indian National)', 'Total Confirmed cases ( Foreign National )'])

covid_india_cases.style.apply(highlight_max,subset=['Cured/Discharged/Migrated', 'Deaths','Total cases','Active cases'])
covid_india_cases.groupby(["Name of State / UT"])["Active cases"].sum().sort_values(ascending=False)
covid_19_india = pd.read_csv("/kaggle/input/covid19-analysis/covid_19_india.csv")
covid_19_india.head(5)
#day by day cases

dbd_cases = covid_19_india.groupby(["Date"])["ConfirmedIndianNational"].sum()
# Rise in COVID-19 cases in India

# Visualisation libraries

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set()

import pycountry

import plotly.express as px

from plotly.offline import init_notebook_mode, iplot 

import plotly.graph_objs as go

import plotly.offline as py

from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot



fig = go.Figure()

fig.add_trace(go.Scatter(x=covid_19_india['Date'], y=covid_19_india['ConfirmedIndianNational'],

                    mode='lines+markers',name='ConfirmedIndianNational'))



fig.add_trace(go.Scatter(x=covid_19_india['Date'], y=covid_19_india['ConfirmedForeignNational'], 

                mode='lines',name='ConfirmedForeignNational'))

fig.add_trace(go.Scatter(x=covid_19_india['Date'], y=covid_19_india['Cured'], 

                mode='lines',name='Cured'))

fig.add_trace(go.Scatter(x=covid_19_india['Date'], y=covid_19_india['Deaths'], 

                mode='lines',name='Deaths'))

        

    

fig.update_layout(title_text='Trend of Coronavirus Cases in India(Cumulative cases)',plot_bgcolor='rgb(250, 242, 242)')



fig.show()
confirmed = pd.read_csv("/kaggle/input/covid19-analysis/time_series_covid_19_confirmed.csv")

recovered = pd.read_csv("/kaggle/input/covid19-analysis/time_series_covid_19_recovered.csv")

deaths = pd.read_csv("/kaggle/input/covid19-analysis/time_series_covid_19_deaths.csv")
confirmed.columns
confirmed.head(2)
india_conf = confirmed[confirmed["Country/Region"] == "India"]
#convert pandas into array of datetime value and confirmed cases

india_conf_x = india_conf.columns

india_conf_x = india_conf_x[4:]

india_conf_y = []

for i in india_conf_x:

    india_conf_y.append(india_conf[i].values[0])
fig = go.Figure()

fig.add_trace(go.Scatter(x=india_conf_x, y=india_conf_y,

                    mode='lines+markers',name='timeseries'))

fig.update_layout(title_text='Time Series of confirmed cases',plot_bgcolor='rgb(250, 242, 242)')

fig.show()
india_rec = recovered[recovered["Country/Region"] == "India"]

#convert pandas into array of datetime value and confirmed cases

india_rec_x = india_rec.columns

india_rec_x = india_rec_x[4:]

india_rec_y = []

for i in india_rec_x:

    india_rec_y.append(india_rec[i].values[0])
fig = go.Figure()

fig.add_trace(go.Scatter(x=india_rec_x, y=india_rec_y,

                    mode='lines+markers',name='timeseries'))

fig.update_layout(title_text='Time Series of recovered cases',plot_bgcolor='rgb(250, 242, 242)')

fig.show()
india_death = deaths[deaths["Country/Region"] == "India"]

#convert pandas into array of datetime value and confirmed cases

india_death_x = india_death.columns

india_death_x = india_death_x[4:]

india_death_y = []

for i in india_death_x:

    india_death_y.append(india_death[i].values[0])
fig = go.Figure()

fig.add_trace(go.Scatter(x=india_death_x, y=india_death_y,

                    mode='lines+markers',name='timeseries'))

fig.update_layout(title_text='Time Series of death cases',plot_bgcolor='rgb(250, 242, 242)')

fig.show()
fig = go.Figure()

fig.add_trace(go.Scatter(x=india_conf_x, y=india_conf_y,

                    mode='lines+markers',name='confirmed'))



fig.add_trace(go.Scatter(x=india_rec_x, y=india_rec_y,

                    mode='lines+markers',name='recovered'))



fig.add_trace(go.Scatter(x=india_death_x, y=india_death_y,

                    mode='lines+markers',name='deaths'))

fig.update_layout(title_text='Time Series of death cases',plot_bgcolor='rgb(250, 242, 242)')

fig.show()