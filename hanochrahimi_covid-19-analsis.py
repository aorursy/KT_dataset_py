import pandas as pd

import numpy as np

import seaborn as sns

from matplotlib import pyplot as plt

import plotly.graph_objects as go

from plotly.subplots import make_subplots

from fbprophet import Prophet

import pycountry

import plotly.express as px
# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv',parse_dates=['Last Update'])

df.rename(columns={'ObservationDate':'Date', 'Country/Region':'Country'}, inplace=True)



df_confirmed = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")

df_recovered = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv")

df_deaths = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv")



df_confirmed.rename(columns={'Country/Region':'Country'}, inplace=True)

df_recovered.rename(columns={'Country/Region':'Country'}, inplace=True)

df_deaths.rename(columns={'Country/Region':'Country'}, inplace=True)
df2 = df.groupby(["Date", "Country"])[['Date', 'Country', 'Confirmed', 'Deaths', 'Recovered']].sum().reset_index()

# df2 = df.groupby(["Date"])[['Date', 'Confirmed', 'Deaths', 'Recovered']].sum().reset_index()

df.query('Country=="Israel"').groupby("Last Update")[['Confirmed', 'Deaths', 'Recovered']].sum().reset_index()
df.groupby("Country")[['Confirmed', 'Deaths', 'Recovered']].sum().reset_index()
confirmed = df.groupby('Date').sum()['Confirmed'].reset_index()

deaths = df.groupby('Date').sum()['Deaths'].reset_index()

recovered = df.groupby('Date').sum()['Recovered'].reset_index()



dfbydate = df[df['Country']=="Italy"].groupby('Date').sum().sort_values(by='Date')

ratio = dfbydate['Deaths'].divide(dfbydate['Confirmed'])

ratiod10 = dfbydate['Deaths'][10:].divide(dfbydate['Confirmed'][:-10])



fig = go.Figure()

fig.add_trace(go.Bar(x=dfbydate.index,

                y=ratio,

                name='ratio',

                marker_color='red'

                ))

fig.add_trace(go.Bar(x=dfbydate.index[10:],

                y=ratiod10,

                name='ratio d10',

                marker_color='blue'

                ))

countries=['Italy','Israel','US', 'Mainland China', 'Spain']

step=10



# fig = make_subplots(rows=1, cols=2)

fig1 = go.Figure()

fig1.update_layout(title="#new infections")

fig2=go.Figure()#

fig2.update_layout(title="Death Ratio(%d)"%(step))



for c in countries:

    print(c)

    tmp = df[df['Country']==c].groupby('Date').sum().sort_values(by='Date')

    # tmp = df.groupby('Date').sum().sort_values(by='Date')

    confirmed = np.array(tmp['Confirmed'])

    deaths = np.array(tmp['Deaths'])

    dc=confirmed[step:]-confirmed[:-step]

#     dd=deaths[step:]-deaths[:-step]

    fig1.add_trace(go.Scatter(x=tmp[step:].index, y=dc, name=c))

    fig2.add_trace(go.Scatter(x=tmp[step:].index, y=100*deaths[step:]/confirmed[:-step], name=c))

# fig1.add_trace(go.Scatter(x=tmp[2:].index, y=confirmed, name='confirmed',marker_color='blue'))

# fig2.add_trace(go.Scatter(x=tmp[2:].index, y=100*deaths[10:]/confirmed[:-10], name='deathratio'))

fig1.show()

fig2.show()



    

#     fig.add_trace(go.Scatter(x=tmp[2:].index, y=deaths, name='death',marker_color='red')

#                  , row=1,col=1)

# fig.add_trace(go.Scatter(x=tmp[2:].index, y=dd, name='diffdeath',marker_color='red')

#              , row=1,col=2)

fig = go.Figure()

fig.add_trace(go.Bar(x=confirmed['Date'],

                y=confirmed['Confirmed'],

                name='Confirmed',

                marker_color='blue'

                ))

fig.add_trace(go.Bar(x=deaths['Date'],

                y=deaths['Deaths'],

                name='Deaths',

                marker_color='Red'

                ))

fig.add_trace(go.Bar(x=recovered['Date'],

                y=recovered['Recovered'],

                name='Recovered',

                marker_color='Green'

                ))



fig.update_layout(

    title='Worldwide Corona Virus Cases - Confirmed, Deaths, Recovered (Bar Chart)',

    xaxis_tickfont_size=14,

    yaxis=dict(

        title='Number of Cases',

        titlefont_size=16,

        tickfont_size=14,

    ),

    legend=dict(

        x=0,

        y=1.0,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)'

    ),

    barmode='group',

    bargap=0.15, # gap between bars of adjacent location coordinates.

    bargroupgap=0.1 # gap between bars of the same location coordinate.

)

fig.show()
fig = go.Figure()

fig.add_trace(go.Scatter(x=confirmed['Date'], 

                         y=confirmed['Confirmed'],

                         mode='lines+markers',

                         name='Confirmed',

                         line=dict(color='blue', width=2)

                        ))

fig.add_trace(go.Scatter(x=deaths['Date'], 

                         y=deaths['Deaths'],

                         mode='lines+markers',

                         name='Deaths',

                         line=dict(color='Red', width=2)

                        ))

fig.add_trace(go.Scatter(x=recovered['Date'], 

                         y=recovered['Recovered'],

                         mode='lines+markers',

                         name='Recovered',

                         line=dict(color='Green', width=2)

                        ))

fig.update_layout(

    title='Worldwide Corona Virus Cases - Confirmed, Deaths, Recovered (Line Chart)',

    xaxis_tickfont_size=14,

    yaxis=dict(

        title='Number of Cases',

        titlefont_size=16,

        tickfont_size=14,

    ),

    legend=dict(

        x=0,

        y=1.0,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)'

    )

)

fig.show()
df_confirmed = df_confirmed[["Province/State","Lat","Long","Country"]]

df_latlong = pd.merge(df, df_confirmed, on=["Province/State" ,"Country"])



fig = px.density_mapbox(df_latlong, 

                        lat="Lat", 

                        lon="Long", 

                        hover_name="Province/State", 

                        hover_data=["Confirmed","Deaths","Recovered"], 

                        animation_frame="Date",

                        color_continuous_scale="Portland",

                        radius=7, 

                        zoom=0,height=700)

fig.update_layout(title='Worldwide Corona Virus Cases Time Lapse - Confirmed, Deaths, Recovered',

                  font=dict(family="Courier New, monospace",

                            size=18,

                            color="#7f7f7f")

                 )

fig.update_layout(mapbox_style="open-street-map", mapbox_center_lon=0)

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})





fig.show()
confirmed = df.groupby(['Date', 'Country']).sum()[['Confirmed']].reset_index()

deaths = df.groupby(['Date', 'Country']).sum()[['Deaths']].reset_index()

recovered = df.groupby(['Date', 'Country']).sum()[['Recovered']].reset_index()
latest_date = confirmed['Date'].max()

latest_date
confirmed = confirmed[(confirmed['Date']==latest_date)][['Country', 'Confirmed']]

deaths = deaths[(deaths['Date']==latest_date)][['Country', 'Deaths']]

recovered = recovered[(recovered['Date']==latest_date)][['Country', 'Recovered']]
all_countries = confirmed['Country'].unique()

print("Number of countries with cases: " + str(len(all_countries)))

print("Countries with cases: ")

for i in all_countries:

    print("    " + str(i))
countries = {}

for country in pycountry.countries:

    countries[country.name] = country.alpha_3

    

confirmed["iso_alpha"] = confirmed["Country"].map(countries.get)

deaths["iso_alpha"] = deaths["Country"].map(countries.get)

recovered["iso_alpha"] = recovered["Country"].map(countries.get)
plot_data_confirmed = confirmed[["iso_alpha","Confirmed", "Country"]]

plot_data_deaths = deaths[["iso_alpha","Deaths"]]

plot_data_recovered = recovered[["iso_alpha","Recovered"]]
fig = px.scatter_geo(plot_data_confirmed, locations="iso_alpha", color="Country",

                     hover_name="iso_alpha", size="Confirmed",

                     projection="natural earth", title = 'Worldwide Confirmed Cases')

fig.show()
fig = px.scatter_geo(plot_data_deaths, locations="iso_alpha", color="Deaths",

                     hover_name="iso_alpha", size="Deaths",

                     projection="natural earth", title="Worldwide Death Cases")

fig.show()
fig = px.scatter_geo(plot_data_recovered, locations="iso_alpha", color="Recovered",

                     hover_name="iso_alpha", size="Recovered",

                     projection="natural earth", title="Worldwide Recovered Cases")

fig.show()
confirmed = df.groupby('Date').sum()['Confirmed'].reset_index()

deaths = df.groupby('Date').sum()['Deaths'].reset_index()

recovered = df.groupby('Date').sum()['Recovered'].reset_index()
confirmed.columns = ['ds','y']

#confirmed['ds'] = confirmed['ds'].dt.date

confirmed['ds'] = pd.to_datetime(confirmed['ds'])
confirmed.head()
m = Prophet(interval_width=0.95)

m.fit(confirmed)

future = m.make_future_dataframe(periods=7)

future.tail()
forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
confirmed_forecast_plot = m.plot(forecast)
deaths.columns = ['ds','y']

deaths['ds'] = pd.to_datetime(deaths['ds'])
m = Prophet(interval_width=0.95)

m.fit(deaths)

future = m.make_future_dataframe(periods=7)

future.tail()
forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
deaths_forecast_plot = m.plot(forecast)
recovered.columns = ['ds','y']

recovered['ds'] = pd.to_datetime(recovered['ds'])
m = Prophet(interval_width=0.95)

m.fit(recovered)

future = m.make_future_dataframe(periods=7)

future.tail()
forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
recovered_forecast_plot = m.plot(forecast)