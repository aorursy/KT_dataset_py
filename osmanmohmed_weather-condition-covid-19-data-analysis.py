# first import the lipraries

import pandas as pd

import numpy as np

import math

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objects as go

from plotly.subplots import make_subplots
# Import the data

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

covid_19 = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')
covid_19.head()
#Data Preprocessing

grouped = (covid_19.groupby(['Province/State','ObservationDate']).Confirmed.sum()).reset_index()



#_____________Hong Kong New Cases Data______



hong_kong = grouped.loc[grouped['Province/State']=='Hong Kong']

hong_kong.insert(3, 'New_cases', hong_kong[['Confirmed']].diff().fillna(hong_kong[['Confirmed']]), True)

hong_kong = hong_kong.iloc[1:117].reset_index()



hong_kong.head()
#_____________NSW New Cases Data_____________



nsw = grouped.loc[grouped['Province/State']=='New South Wales']

nsw.insert(3, 'New_cases', nsw[['Confirmed']].diff().fillna(nsw[['Confirmed']]), True)

nsw = nsw.iloc[0:108].reset_index()

nsw.head()
#____________Newyork State New Cases Data__________



california = grouped.loc[grouped['Province/State']=='California'] 

california.insert(3,'New_cases', california[['Confirmed']].diff().fillna(california[['Confirmed']]), True)

new_cases_ca = california.iloc[6:75,:].reset_index()

new_cases_ca.head()
newyork = grouped.loc[grouped['Province/State']=='New York'] 

newyork.insert(3,'New_cases', newyork[['Confirmed']].diff().fillna(newyork[['Confirmed']]), True)

new_cases_ny = newyork.iloc[0:69,:].reset_index()

new_cases_ny.head()
fig = go.Figure()

fig.add_trace(go.Bar(x=hong_kong['ObservationDate'],

                y=hong_kong['New_cases'],

                name='Hong Kong',

                marker_color='blue'

                ))

fig.add_trace(go.Bar(x=nsw['ObservationDate'],

                y=nsw['New_cases'],

                name='New South Wales',

                marker_color='Red'

                ))

fig.update_layout(

    title='Corona Virus New Cases Per Day (Hong Kong & NSW)',

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

fig1 = go.Figure()

fig1.add_trace(go.Bar(x=new_cases_ca['ObservationDate'],

                y=new_cases_ca['New_cases'],

                name='California',

                marker_color='Green'

                ))

fig1.add_trace(go.Bar(x=new_cases_ny['ObservationDate'],

                y=new_cases_ny['New_cases'],

                name='New York',

                marker_color='Black'

#                      'rgb(55, 83, 109)'

                ))

fig1.update_layout(

    title='Corona Virus New Cases Per Day (New York & California)',

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

fig1.show()
#____________Hong Kong Weather Data_______________



hk_weather = pd.read_excel('/kaggle/input/weather-data/covid_data/all_weather.xlsx')

hk_weather = hk_weather.rename(columns={'Temp-Mean':'TAVG','Mean Relative Humidity (%)':'Humidity',

                                       'Mean Wind Speed (km/h)':'AWND','Day':'DATE'})

hk_weather = hk_weather.iloc[13:138,:]



hk_weather.loc[:,'TAVG'] = hk_weather['TAVG'].rolling(10).mean()

hk_weather.loc[:,'Humidity'] = hk_weather['Humidity'].rolling(10).mean()

hk_weather.loc[:,'AWND'] = hk_weather['AWND'].rolling(10).mean()



hk_weather = hk_weather.iloc[9::,:].reset_index()



# # #__________Hong Kong Final Data___________



rHongKong = pd.concat([hk_weather[['DATE']],hong_kong[['New_cases']],hk_weather[['TAVG']],

                      hk_weather[['AWND']], hk_weather[['Humidity']]], axis = 1)

rHongKong.head()


fig = make_subplots(rows=2, cols=1)#, column_widths=[0.7, 0.3])



fig.add_trace(go.Bar(x=rHongKong['DATE'],

                y=rHongKong['AWND'],name='Wind Speed (km/H)',marker_color='green'),

                row=1, col=1)

fig.add_trace(go.Scatter(x=rHongKong['DATE'],

                y=rHongKong['Humidity'],name='Humidity (%)',marker_color='blue'),

                row=2, col=1)

fig.add_trace(go.Bar(x=rHongKong['DATE'],

                y=rHongKong['TAVG'],name='Temperature in (°C)',

                marker=dict(color=rHongKong['TAVG'], coloraxis="coloraxis")),

                row=2, col=1)

              



fig.update_layout(title='Hong Kong Weather')



fig.update_layout(coloraxis=dict(colorscale='Inferno'),legend=dict(x=1,

        y=1.2,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)'))

fig.show()
#____________NSW Weather Data_______________



nsw_weather = pd.read_csv('/kaggle/input/weather-data/covid_data/nswa.csv')

nswhumid = pd.read_excel('/kaggle/input/weather-data/covid_data/all_weather.xlsx', sheet_name='NSW')



nsw_temp = nsw_weather.groupby(['DATE']).TAVG.mean().reset_index()

nsw_temp.loc[:,'TAVG'] = nsw_temp['TAVG'].rolling(10).mean()

nsw_w = nsw_temp.iloc[9::,:].reset_index()

#Wind

nswwh = nswhumid.set_index(nswhumid['Date'])

nswind = nswwh['Wind'].rolling(10).mean()

rolling_nswind = nswind.iloc[9:117].reset_index()

#Humidity

nswhumid = nswhumid.set_index(nswhumid['Date'])

nswhumidh = nswwh['Humidity'].rolling(10).mean()

rolling_nswhumid = nswhumidh.iloc[9:117].reset_index()



# #NSW Final Data

rnsw = pd.concat([nsw_w[['DATE']],nsw[['New_cases']],nsw_w[['TAVG']],

                 rolling_nswind[['Wind']],rolling_nswhumid[['Humidity']]], axis = 1)

rnsw = rnsw.rename(columns={'Wind':'AWND'})

rnsw.head()


fig = make_subplots(rows=2, cols=1)



fig.add_trace(go.Bar(x=rnsw['DATE'],

                y=rnsw['AWND'],name='Wind Speed (km/H)',marker_color='green'),

                row=1, col=1)

fig.add_trace(go.Scatter(x=rnsw['DATE'],

                y=rnsw['Humidity'],name='Humidity (%)',marker_color='blue'),

                row=2, col=1)

fig.add_trace(go.Bar(x=rnsw['DATE'],

                y=rnsw['TAVG'],name='Temperature (°C)',

                marker=dict(color=rnsw['TAVG'], coloraxis="coloraxis")),

                row=2, col=1)

              



fig.update_layout(title='New South Wales Weather')



fig.update_layout(coloraxis=dict(colorscale='Inferno'),legend=dict(x=1,

        y=1.2,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)'))

fig.show()
#____________Newyork State New Weather Data__________

#Data Preprocessing

ca_weather_1 = pd.read_csv('/kaggle/input/weather-data/covid_data/cal1.csv')

ca_weather_2 = pd.read_csv('/kaggle/input/weather-data/covid_data/cal2.csv')

cal_weather_n = ca_weather_1[ca_weather_1['DATE']>='2020-03-01']



ca_weather = pd.concat([cal_weather_n,ca_weather_2], axis=0)

cahumid = pd.read_excel('/kaggle/input/weather-data/covid_data/all_weather.xlsx', sheet_name='CA')



#convert mile to km for wind speed

# ca_weather[['AWND']] = ca_weather[['AWND']].apply(lambda w: w*1.609)



ca_temp = ca_weather.groupby('DATE').TAVG.mean()

ca_rolling_temp = ca_temp.rolling(10).mean()

ca_wind = ca_weather.groupby('DATE').AWND.mean()

ca_rolling_wind = ca_wind.rolling(10).mean()

ca_rolling_temp = ca_rolling_temp.iloc[9:78].reset_index()

ca_rolling_wind = ca_rolling_wind.iloc[9:78].reset_index()

#Humidity

cahumid = cahumid.set_index(cahumid['Date'])

chumid = cahumid['Humidity'].rolling(10).mean()

rolling_cahumid = chumid.iloc[9:78].reset_index()

# # California State Final Data 



rcalifornia = pd.concat([ca_rolling_temp[['DATE']],new_cases_ca[['New_cases']],ca_rolling_temp[['TAVG']],ca_rolling_wind[['AWND']],

                        rolling_cahumid[['Humidity']]], axis = 1)

rcalifornia.head()
fig = make_subplots(rows=2, cols=1)



fig.add_trace(go.Bar(x=rcalifornia['DATE'],

                y=rcalifornia['AWND'],name='Wind Speed (km/H)',marker_color='green'),

                row=1, col=1)

fig.add_trace(go.Scatter(x=rcalifornia['DATE'],

                y=rcalifornia['Humidity'],name='Humidity (%)',marker_color='blue'),

                row=2, col=1)

fig.add_trace(go.Bar(x=rcalifornia['DATE'],

                y=rcalifornia['TAVG'],name='Temperature (°C)',

                marker=dict(color=rcalifornia['TAVG'], coloraxis="coloraxis")),

                row=2, col=1)

              



fig.update_layout(title='California State Weather')



fig.update_layout(coloraxis=dict(colorscale='Inferno'),legend=dict(x=1,

        y=1.2,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)'))

fig.show()
#____________Newyork State Weather Data____________



nweather = pd.read_csv('/kaggle/input/weather-data/covid_data/weather.csv')

nyhumid = pd.read_excel('/kaggle/input/weather-data/covid_data/all_weather.xlsx', sheet_name='NY')

#Temp & Wind



#convert mile to km for wind speed

# nweather[['AWND']] = nweather[['AWND']].apply(lambda w: w*1.609)



temp = nweather.groupby('DATE').TAVG.mean()

rolling_temp = temp.rolling(10).mean()

wind = nweather.groupby('DATE').AWND.mean()

rolling_wind = wind.rolling(10).mean()

rolling_temp = rolling_temp.iloc[9:78].reset_index()

rolling_wind = rolling_wind.iloc[9:78].reset_index()

#Humidity

nyhumid = nyhumid.set_index(nyhumid['Date'])

humid = nyhumid['Humidity'].rolling(10).mean()

rolling_humid = humid.iloc[9:78].reset_index()



# #_______Newyork State Final Data 

rnewyork = pd.concat([rolling_temp[['DATE']],new_cases_ny[['New_cases']],rolling_temp[['TAVG']],rolling_wind[['AWND']],

                     rolling_humid[['Humidity']]], axis = 1)

rnewyork.head()
fig = make_subplots(rows=2, cols=1)



fig.add_trace(go.Bar(x=rnewyork['DATE'],

                y=rnewyork['AWND'],name='Wind Speed (km/H)',marker_color='green'),

                row=1, col=1)

fig.add_trace(go.Scatter(x=rnewyork['DATE'],

                y=rnewyork['Humidity'],name='Humidity (%)',marker_color='blue'),

                row=2, col=1)

fig.add_trace(go.Bar(x=rnewyork['DATE'],

                y=rnewyork['TAVG'],name='Temperature (°C)',

                marker=dict(color=rcalifornia['TAVG'], coloraxis="coloraxis")),

                row=2, col=1)

              



fig.update_layout(title='New York State Weather')



fig.update_layout(coloraxis=dict(colorscale='Inferno'),legend=dict(x=1,

        y=1.2,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)'))

fig.show()
#Concatenate all The Data 

#Date

all_dates = pd.concat([rHongKong[['DATE']],

                      rnsw[['DATE']],

                      rcalifornia[['DATE']],

                      rnewyork[['DATE']],

                     ], axis = 0)

#New Cases

all_new_cases = pd.concat([rHongKong[['New_cases']],

                      rnsw[['New_cases']],

                      rcalifornia[['New_cases']],

                      rnewyork[['New_cases']],

                     ], axis = 0)

#Tempriture

all_temp = pd.concat([rHongKong[['TAVG']],rnsw[['TAVG']],

                      rcalifornia[['TAVG']],rnewyork[['TAVG']]

                     ], axis = 0)

#Wind

all_wind = pd.concat([rHongKong[['AWND']],rnsw[['AWND']],

                      rcalifornia[['AWND']],rnewyork[['AWND']]

                     ], axis = 0)

#Humidity

all_humid = pd.concat([rHongKong[['Humidity']],rnsw[['Humidity']],

                      rcalifornia[['Humidity']],rnewyork[['Humidity']]

                     ], axis = 0)

all_data = pd.concat([all_dates,all_new_cases, all_temp, all_wind, all_humid], axis = 1)



## Correlation Test

all_data.corr(method='spearman').style.background_gradient('viridis')