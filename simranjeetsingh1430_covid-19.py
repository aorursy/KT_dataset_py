# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import seaborn as sns

from matplotlib import pyplot as plt

import plotly.graph_objects as go

import pycountry

import plotly.express as px
df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',parse_dates=['Last Update'])

df.rename(columns={'ObservationDate':'Date', 'Country/Region':'Country'}, inplace=True)

df_confirmed = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")

df_recovered = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv")

df_deaths = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv")

df_confirmed.rename(columns = {'Country/Region':'Country'}, inplace=True)

df_recovered.rename(columns = {'Country/Region':'Country'}, inplace=True)

df_deaths.rename(columns ={'Country/Region':'Country'}, inplace=True)

df_confirmed.head()
#Earlier Cases

df.head()
#Latest Cases

df.tail() 
df2 = df.groupby(['Date','Country','Province/State'])[['Date','Province/State','Country','Confirmed','Deaths','Recovered']].sum().reset_index()

df2
#Cases in China

df.query('Country=="Mainland China"').groupby("Last Update")[['Confirmed', 'Deaths', 'Recovered']].sum().reset_index()
#Cases in all Countries 

df.groupby("Country")[['Confirmed', 'Deaths', 'Recovered']].sum().reset_index()
df.groupby('Date').sum()
confirmed = df.groupby('Date').sum()['Confirmed'].reset_index()

deaths = df.groupby('Date').sum()['Deaths'].reset_index()

recovered =  df.groupby('Date').sum()['Recovered'].reset_index()
fig = go.Figure()

fig.add_trace(go.Line(x=confirmed['Date'],

                    y = confirmed['Confirmed'],

                    name = 'Confirmed',

                    marker_color = 'blue'

                    ))

fig.add_trace(go.Line(x = deaths['Date'],

                    y = deaths['Deaths'],

                    name = 'Deaths',

                    marker_color = 'red'

                    ))

fig.add_trace(go.Line(x=recovered['Date'],

                    y = recovered['Recovered'],

                    name = 'Recovered',

                    marker_color = 'green'

                    ))

fig.update_layout(

title = 'Worldwide Corona Virus Cases - C - D - R',

    xaxis_tickfont_size = 14,

        yaxis = dict(

        title = 'Number of Cases',

        titlefont_size = 16,

        tickfont_size = 14,

        ),

    legend = dict(

        x = 0,

        y = 1.0,

        bgcolor = 'rgba(255,255,255,0)',

        bordercolor = 'rgba(255,255,255,0)'

    ),

    barmode = 'group',

    bargap = 0.15,

    bargroupgap = 0.1

)

fig.show()



df_confirmed = df_confirmed[["Province/State","Lat","Long","Country"]]

df_temp = df.copy()

df_temp['Country'].replace({'Mainland China': 'China'}, inplace=True)

df_latlong = pd.merge(df_temp, df_confirmed, on=["Country", "Province/State"]) 
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

                            color="#7f107f")

                 )

fig.update_layout(mapbox_style="open-street-map", mapbox_center_lon=0)

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
confirmed = df2.groupby(['Date', 'Country']).sum()[['Confirmed']].reset_index()

deaths = df2.groupby(['Date', 'Country']).sum()[['Deaths']].reset_index()

recovered = df2.groupby(['Date', 'Country']).sum()[['Recovered']].reset_index()
latest_date = confirmed['Date'].max()

latest_date
confirmed = confirmed[(confirmed['Date']==latest_date)][['Country', 'Confirmed']]

deaths = deaths[(deaths['Date']==latest_date)][['Country', 'Deaths']]

recovered = recovered[(recovered['Date']==latest_date)][['Country', 'Recovered']]
all_countries = confirmed['Country'].unique()

print("Number of countries/regions with cases: " + str(len(all_countries)))

print("Countries/Regions with cases: ")

for i in all_countries:

    print("    " + str(i))
print(list(country.name for country in pycountry.countries))
print('UK' in list(country.name for country in pycountry.countries))

print('United Kingdom' in list(country.name for country in pycountry.countries))
confirmed2 = confirmed.copy()

deaths2 = deaths.copy()

recovered2 = recovered.copy()

bubble_plot_dfs = [confirmed2, deaths2, recovered2]

for df_ in bubble_plot_dfs:

    df_["Country"].replace({'Mainland China': 'China'}, inplace=True)

    df_["Country"].replace({'UK': 'United Kingdom'}, inplace=True)

    df_["Country"].replace({'US': 'United States'}, inplace=True)
countries = {}

for country in pycountry.countries:

    countries[country.name] = country.alpha_3

    

confirmed2["iso_alpha"] = confirmed2["Country"].map(countries.get)

deaths2["iso_alpha"] = deaths2["Country"].map(countries.get)

recovered2["iso_alpha"] = recovered2["Country"].map(countries.get)
plot_data_confirmed = confirmed2[["iso_alpha","Confirmed", "Country"]]

plot_data_deaths = deaths2[["iso_alpha","Deaths"]]

plot_data_recovered = recovered2[["iso_alpha","Recovered"]]

fig = px.scatter_geo(plot_data_confirmed, locations="iso_alpha", color="Country",

                     hover_name="iso_alpha", size="Confirmed",

                     projection="natural earth", title = 'Worldwide Confirmed Cases')

fig.show()
fig = px.scatter_geo(plot_data_deaths, locations="iso_alpha", color="Deaths",

                     hover_name="iso_alpha", size="Deaths",

                     projection="natural earth", title="Worldwide Death Cases")

fig.show()