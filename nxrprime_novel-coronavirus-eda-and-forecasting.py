import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('ggplot')

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import statsmodels

import folium

import plotly.offline as py

from plotly import tools

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.express as px
import pandas as pd

case = pd.read_csv("../input/coronavirusdataset/case.csv")

patient = pd.read_csv("../input/coronavirusdataset/patient.csv")

route = pd.read_csv("../input/coronavirusdataset/route.csv")

time = pd.read_csv("../input/coronavirusdataset/time.csv")

trend = pd.read_csv("../input/coronavirusdataset/trend.csv")
df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',parse_dates=['Last Update'])

df.rename(columns={'ObservationDate':'Date', 'Country/Region':'Country'}, inplace=True)

df_ll = pd.read_csv("../input/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv")



df_confirmed = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")

df_recovered = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv")

df_deaths = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv")



df_confirmed.rename(columns={'Country/Region':'Country'}, inplace=True)

df_recovered.rename(columns={'Country/Region':'Country'}, inplace=True)

df_deaths.rename(columns={'Country/Region':'Country'}, inplace=True)
world_map = folium.Map(location=[10, -20], zoom_start=2.3,tiles='openstreetmap')



for lat, lon, value, name in zip(df_confirmed['Lat'], df_confirmed['Long'], df['Confirmed'], df_confirmed['Country']):

    folium.CircleMarker([lat, lon],

                        radius=10,

                        popup = ('<strong>Country</strong>: ' + str(name).capitalize() + '<br>'

                                '<strong>Confirmed Cases</strong>: ' + str(value) + '<br>'),

                        color='red',

                        

                        fill_color='red',

                        fill_opacity=0.7 ).add_to(world_map)

world_map
world_map2 = folium.Map(location=[10, -20], zoom_start=2.3,tiles='openstreetmap')



for lat, lon, value, name in zip(df_deaths['Lat'], df_deaths['Long'], df['Deaths'], df_deaths['Country']):

    folium.CircleMarker([lat, lon],

                        radius=10,

                        popup = ('<strong>Country</strong>: ' + str(name).capitalize() + '<br>'

                                '<strong>Death Cases</strong>: ' + str(value) + '<br>'),

                        color='red',

                        

                        fill_color='black',

                        fill_opacity=0.7 ).add_to(world_map2)

world_map2
### Plot for number of cumulative covid cases over time

fig = px.bar(df, x="Date", y="Confirmed")

layout = go.Layout(

    title=go.layout.Title(

        text="Daily cumulative count of confirmed COVID-19 cases in the world",

        x=0.5

    ),

    font=dict(size=14),

    width=800,

    height=500,

    xaxis_title = "Date of observation",

    yaxis_title = "Number of confirmed cases"

)



fig.update_layout(layout)

fig.show()
fig = px.bar(df, x="Date", y="Deaths")

layout = go.Layout(

    title=go.layout.Title(

        text="Daily cumulative count of confirmed COVID-19 deaths in the world",

        x=0.5

    ),

    font=dict(size=14),

    width=800,

    height=500,

    xaxis_title = "Date of observation",

    yaxis_title = "Number of deaths"

)



fig.update_layout(layout)

fig.show()
from fbprophet import Prophet



df["ds"] = df["Date"]

df["y"] = df["Deaths"]

df
model = Prophet(yearly_seasonality=True) 

model.fit(df)
future = model.make_future_dataframe(periods = 5, freq = 'MS')  

# now lets make the forecasts

forecast = model.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
model.plot(forecast)
model.plot_components(forecast)