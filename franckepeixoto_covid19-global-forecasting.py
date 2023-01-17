import numpy as np 

import pandas as pd

import os

import datetime

import folium

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.offline as py

import plotly.express as px



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')

test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')

countries = pd.read_csv('/kaggle/input/countries/countries.csv')


columns =['Latitude','Longitude']

for col in columns:

  train[col]=0.

  test[col]=0.



for index, row in countries.iterrows():  

  test.loc[test.Country_Region==row['Country'],["Latitude"]] = row['Latitude'] 

  test.loc[test.Country_Region==row['Country'],["Longitude"]] = row['Longitude']

  train.loc[train.Country_Region==row['Country'],["Longitude"]] = row['Longitude']

  train.loc[train.Country_Region==row['Country'],["Latitude"]] = row['Latitude'] 



train.Date = train.Date.apply(pd.to_datetime)



train.info()
train.Date.min(),train.Date.max()
import folium

groupByConfirmed = train.groupby(["Country_Region"])["Fatalities","ConfirmedCases"].sum()

groupByConfirmed = groupByConfirmed.reset_index()

grouped = groupByConfirmed.sort_values("ConfirmedCases",ascending=False).copy()

grouped.head(30)


grpConfirmedCases = train.groupby(['Date', 'Country_Region'])['ConfirmedCases'].max()

grpConfirmedCases = grpConfirmedCases.reset_index()

grpConfirmedCases.Date = grpConfirmedCases.Date.dt.strftime('%m/%d/%Y')

grpConfirmedCases.Country  =  grpConfirmedCases.Country_Region

fig = px.choropleth(grpConfirmedCases,

                    locations="Country_Region",

                    locationmode='country names',

                    color="ConfirmedCases",

                    hover_name="Country_Region",

                    hover_data = [grpConfirmedCases.ConfirmedCases],

                    projection="mercator",

                    animation_frame="Date",

                    width=1000, 

                    height=700,

                    color_continuous_scale='Reds')



fig.update(layout_coloraxis_showscale=True)

py.offline.iplot(fig)
lat_lon = train[train.Country_Region=='Brazil'][["Latitude","Longitude"]].values[0]

map = folium.Map(location=lat_lon, tiles="Stamen Toner", zoom_start=3)



for index, row in grouped.iterrows():

  lat_lont = train[train.Country_Region==row.Country_Region][["Latitude","Longitude"]].values[0]

  folium.CircleMarker(lat_lont,

                      radius= (int((np.log(row.Fatalities+1.00001))))*1,

                      popup = ('<strong>Mortes</strong>: ' + str(row.Fatalities) + '<br><strong>Confirmado</strong>: ' + str(row.ConfirmedCases-row.Fatalities)),

                      color='#1a1a1a',

                      fill_color='#660000',

                      fill_opacity=0.9 ).add_to(map)

  folium.CircleMarker(lat_lont,

                      radius= (int((np.log((row.ConfirmedCases-row.Fatalities)+1.00001))))*1,

                      popup = ('<strong>Mortes</strong>: ' + str(row.Fatalities) + '<br><strong>Confirmado</strong>: ' + str(row.ConfirmedCases-row.Fatalities) + '<br>'),

                      color='#1a1a1a',

                      fill_color='#b38f00',

                      fill_opacity=0.1 ).add_to(map)

map
confirmedCases = train.groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities =train.groupby(['Date']).agg({'Fatalities':['sum']})

#total = confirmedCases.join(fatalities)



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17,4))

confirmedCases.plot(ax=ax1)

ax1.set_title("Global: Casos confirmados", size=13)

ax1.set_ylabel("Número de casos", size=13)

ax1.set_xlabel("Periodo", size=13)

fatalities.plot(ax=ax2, color='red')

ax2.set_title("Global: Mortos", size=13)

ax2.set_ylabel("Número de casos", size=13)

ax2.set_xlabel("Periodo", size=13)
top30 =  grouped[:30].copy()

count = 0

for index, row in top30.iterrows():

    country_Region =row.Country_Region

    confirmed = train[train.Country_Region==country_Region].groupby(['Date']).agg({'ConfirmedCases':['sum']})

    fatal = train[train.Country_Region==country_Region].groupby(['Date']).agg({'Fatalities':['sum']})

    total_date  = confirmed.join(fatal)

    plt.figure(figsize=(17,10))

    plt.subplot(2, 2,1)

    total_date.plot(ax=plt.gca(), title= country_Region)

    plt.ylabel("Casos", size=13)

plt.show()  

  
corr = train.corr()

fig, ax = plt.subplots(figsize=(10, 10))

sns.heatmap(corr,  annot=True, fmt=".3f")

plt.xticks(range(len(corr.columns)), corr.columns);

plt.yticks(range(len(corr.columns)), corr.columns)

plt.show()