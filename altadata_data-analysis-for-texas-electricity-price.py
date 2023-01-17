import numpy as np 

import pandas as pd 

import os

import warnings

warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

import seaborn as sns

import plotly as py

import plotly.graph_objs as go 

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly.express as px

import datetime

from plotly.subplots import make_subplots
import kaggle_secrets

import requests

import json
altadata_io_api_key = kaggle_secrets.UserSecretsClient().get_secret("altadata_io_api_key")

product_code_0 = 'en_01_ercot_04'

product_code_1 = 'en_08_eiaxx_01'

product_code_2 = 'ev_01_altab_04'
def generate_hours_range(start_date, finish_date):

    diff = finish_date - start_date

    hours = int(diff.total_seconds() / (60*60))

    generated_date_list = [start_date + datetime.timedelta(hours=x) for x in range(0, (hours + 1))]

    return generated_date_list

def generate_fif_minutes_range(start_date, finish_date):

    diff = finish_date - start_date

    fif_minutes = int(diff.total_seconds() / (60*15))

    generated_date_list = [start_date + datetime.timedelta(minutes=15*x) for x in range(0, (fif_minutes + 1))]

    return generated_date_list

def generate_day_range(start_date, finish_date):

    diff = finish_date - start_date

    days = int(diff.total_seconds() / (60*60*24))

    generated_date_list = [start_date + datetime.timedelta(days=x) for x in range(0, (days+1))]

    return generated_date_list

def detect_outlier(dataframe, value_column, date_column):

    outliers = pd.DataFrame()

    threshold = 6

    new_df_index = 0

    data = dataframe[value_column].to_list()

    mean_1 = np.mean(data)

    std_1 = np.std(data)

    ind = []

    for index, datum in enumerate(data):

        z_score = (datum - mean_1) / std_1

        if np.abs(z_score) > threshold:

            outliers.at[new_df_index, value_column] = datum

            outliers.at[new_df_index, date_column] = dataframe[date_column].iloc[index]

            ind.append(index)

            new_df_index += 1

    outliers.insert(0,'ind',ind,True)

    return outliers
data_0 = pd.DataFrame()

for i in range (1,392):

    url_0 = f'https://www.altadata.io/data/api/{product_code_0}?api_key={altadata_io_api_key}&settlement_point_name=HB_SOUTH&opr_ts_gte=2015-01-01&/?format=json&page={i}'

    data_0 = data_0.append(pd.read_json(json.dumps(requests.get(url_0).json())))
data_0.info()
data_0.settlement_point_name.value_counts()
data_0.head()
data_0.tail()
data_0.sort_values(by=['settlement_point_price'],ascending=False)
data_0.describe(include='all')
missing = []

for i in data_0.opr_ts:

    d1 = datetime.datetime.strptime(i,"%Y-%m-%dT%H:%M:%S.000+0000")

    new_format = "%Y-%m-%d %H:%M:%S"

    missing.append(d1.strftime(new_format))

data_0.insert(0,'date',missing,True)
data_0.info()
x = datetime.datetime(2015, 1, 1)

y = datetime.datetime(2020, 7, 28)

generated_date_list = generate_fif_minutes_range(x,y)

check_df_date_list = pd.to_datetime(data_0['date'].values)

difference_date_list = set(generated_date_list) - set(check_df_date_list)

dif_date = list(difference_date_list)
data_0 = data_0[['date','settlement_point_price']]

data_0.info()
for i in dif_date:

    data_0 = data_0.append({'date' : i} , ignore_index=True)
data_0.tail()
data_0.date = pd.to_datetime(data_0.date)

data_0 = data_0.sort_values(by='date')
new_index = []

for i in range(0,len(data_0)):

    new_index.append(i)

data_0.insert(0,'ind',new_index,True)

data_0.index = data_0.ind

data_0.tail()
data_0.info()
trace_0 = go.Scatter(x = data_0.date,y = data_0.settlement_point_price.interpolate(method ='linear', limit_direction ='forward'), name = "Interpolation Point Price",

                       line = dict(color = '#173F5F'), opacity = 1.0)

trace_1 = go.Scatter(x = data_0.date,y = data_0.settlement_point_price, name = "Point Price",

                       line = dict(color = '#17BECF'), opacity = 1.0)

data = [trace_0,trace_1]

layout = dict( title = 'HB_SOUTH Price & Interpolation Price')

fig = dict(data=data, layout=layout)

iplot(fig, filename = 'HB_SOUTH-Price-Interpolation')
data_0.settlement_point_price = data_0.settlement_point_price.interpolate(method ='linear', limit_direction ='forward')
data_0.info()
m = 0

meann = []

for i in range(0,len(data_0),4):

    for j in range(0,4):

        m = float(m+data_0.settlement_point_price[i+j])

    meann.append(float(m/4)) # for 15. minutes

    meann.append(float(m/4)) # for 30. minutes

    meann.append(float(m/4)) # for 45. minutes

    meann.append(float(m/4)) # for last minutes

    m=0
data_0.insert(2,'meanprice',meann,True)

data_0.head(8)
data_0.describe(include='all')
data_1 = pd.DataFrame()

for j in range (1,4):

    url_1 = f'https://www.altadata.io/data/api/{product_code_1}?api_key={altadata_io_api_key}&date_gte=2015-01-01&/?format=json&page={j}'

    data_1 = data_1.append(pd.read_json(json.dumps(requests.get(url_1).json())))
data_1.info()
data_1.head()
data_1.tail()
print('Is there a missing date value?:',format(str(data_1.date.isnull().values.any())))

print('Is there a missing Henry Hub NG Spot Price ?:',format(str(data_1.henry_hub_ng_spot_price.isnull().values.any())))

print('Is there a missing NG Futures Contract 1 Price ?:',format(str(data_1.ng_futures_contract1_price.values.any())))

print('Is there a missing NG Futures Contract 2 Price ?:',format(str(data_1.ng_futures_contract2_price.values.any())))

print('Is there a missing NG Futures Contract 3 Price ?:',format(str(data_1.ng_futures_contract3_price.values.any())))

print('Is there a missing NG Futures Contract 4 Price ?:',format(str(data_1.ng_futures_contract4_price.values.any())))
data_1.describe(include='all')
fig = px.bar(data_1, x='date', y='henry_hub_ng_spot_price',title='Historical Daily Natural Gas Spot', labels={'henry_hub_ng_spot_price':'Henry Hub NG Spot Price','ng_futures_contract1_price':'Contract 1','ng_futures_contract2_price':'Contract 2','ng_futures_contract3_price':'Contract 3','ng_futures_contract4_price':'Contract 4',},color='henry_hub_ng_spot_price')

iplot(fig, filename = 'HenryHubNgSpotPrice')
generated_date_list = []

check_df_date_list = []

difference_date_list = set()

dif_date = []
x = datetime.datetime(2015, 1, 1)

y = datetime.datetime(2020, 8, 11)

generated_date_list = generate_day_range(x,y)

check_df_date_list = pd.to_datetime(data_1['date'].values)

difference_date_list = set(generated_date_list) - set(check_df_date_list)

dif_date = list(difference_date_list)
for i in dif_date:

    data_1 = data_1.append({'date' : i} , ignore_index=True)
data_1.info()
data_1.tail()
data_1['date'] =pd.to_datetime(data_1.date)

data_1 = data_1.sort_values(by='date')
new_index = []

for i in range(0,len(data_1)):

    new_index.append(i)

data_1.insert(0,'ind',new_index,True)

data_1.index = data_1.ind

data_1.tail()
data_1.head()
data_1.henry_hub_ng_spot_price.describe()
trace_1_0 = go.Scatter( x=data_1.date, y=data_1.henry_hub_ng_spot_price.interpolate(method ='linear', limit_direction ='forward'), name = "Henry Hub NG Spot Price Interpolation",

                       line = dict(color = '#D50000'), opacity = 1.0)

trace_1_1 = go.Scatter( x = data_1.date,y = data_1.henry_hub_ng_spot_price, name = "Henry Hub Missing Values",

                       line = dict(color = '#E68D20'), opacity = 1.0)

data = [trace_1_0,trace_1_1]

layout = dict( title = 'Henry Hub NG Spot Price - Interpolation')

fig = dict(data=data, layout=layout)

iplot(fig, filename = 'henry_hub_inerpolation')
data_1.ng_futures_contract1_price.describe()
trace_1_0 = go.Scatter( x=data_1.date, y=data_1.ng_futures_contract1_price.interpolate(method ='linear', limit_direction ='forward'), name = "NG Futures Contract 1 Price Interpolation",

                       line = dict(color = '#1A237E'), opacity = 1.0)

trace_1_1 = go.Scatter( x = data_1.date,y = data_1.ng_futures_contract1_price, name = "NG Futures Contract 1 Price Missing Values",

                       line = dict(color = '#00BCD4'), opacity = 1.0)

data = [trace_1_0,trace_1_1]

layout = dict( title = 'NG Futures Contract 1 Price')

fig = dict(data=data, layout=layout)

iplot(fig, filename = 'NG_Futures_Contract_1_Price')
data_1.ng_futures_contract2_price.describe()
trace_1_0 = go.Scatter( x=data_1.date, y=data_1.ng_futures_contract2_price.interpolate(method ='linear', limit_direction ='forward'), name = "NG Futures Contract 2 Price Interpolation",

                       line = dict(color = '#006400'), opacity = 1.0)

trace_1_1 = go.Scatter( x = data_1.date,y = data_1.ng_futures_contract2_price, name = "NG Futures Contract 2 Price Missing Values",

                       line = dict(color = '#00FF7F'), opacity = 1.0)

data = [trace_1_0,trace_1_1]

layout = dict( title = 'NG Futures Contract 2 Price')

fig = dict(data=data, layout=layout)

iplot(fig, filename = 'NG_Futures_Contract_2_Price')
data_1.ng_futures_contract3_price.describe()
trace_1_0 = go.Scatter( x=data_1.date, y=data_1.ng_futures_contract3_price.interpolate(method ='linear', limit_direction ='forward'), name = "NG Futures Contract 3 Price Interpolation",

                       line = dict(color = '#008080'), opacity = 1.0)

trace_1_1 = go.Scatter( x = data_1.date,y = data_1.ng_futures_contract3_price, name = "NG Futures Contract 3 Price Missing Values",

                       line = dict(color = '#66CDAA'), opacity = 1.0)

data = [trace_1_0,trace_1_1]

layout = dict( title = 'NG Futures Contract 3 Price - Interpolation')

fig = dict(data=data, layout=layout)

iplot(fig, filename = 'NG_Futures_Contract_3_Price')
data_1.ng_futures_contract4_price.describe()
trace_1_0 = go.Scatter( x=data_1.date, y=data_1.ng_futures_contract4_price.interpolate(method ='linear', limit_direction ='forward'), name = "NG Futures Contract 4 Price Interpolation",

                       line = dict(color = '#8A2BE2'), opacity = 1.0)

trace_1_1 = go.Scatter( x = data_1.date,y = data_1.ng_futures_contract4_price, name = "NG Futures Contract 4 Price Missing Values",

                       line = dict(color = '#FF00FF'), opacity = 1.0)

data = [trace_1_0,trace_1_1]

layout = dict( title = 'NG Futures Contract 4 Price')

fig = dict(data=data, layout=layout)

iplot(fig, filename = 'NG_Futures_Contract_4_Price')
data_1.henry_hub_ng_spot_price = data_1.henry_hub_ng_spot_price.interpolate(method ='linear', limit_direction ='forward')

data_1.ng_futures_contract1_price = data_1.ng_futures_contract1_price.interpolate(method ='linear', limit_direction ='forward')

data_1.ng_futures_contract2_price = data_1.ng_futures_contract2_price.interpolate(method ='linear', limit_direction ='forward')

data_1.ng_futures_contract3_price = data_1.ng_futures_contract3_price.interpolate(method ='linear', limit_direction ='forward')

data_1.ng_futures_contract4_price = data_1.ng_futures_contract4_price.interpolate(method ='linear', limit_direction ='forward')

data_1.henry_hub_ng_spot_price = data_1.henry_hub_ng_spot_price.interpolate(method ='linear', limit_direction ='backward')

data_1.ng_futures_contract1_price = data_1.ng_futures_contract1_price.interpolate(method ='linear', limit_direction ='backward')

data_1.ng_futures_contract2_price = data_1.ng_futures_contract2_price.interpolate(method ='linear', limit_direction ='backward')

data_1.ng_futures_contract3_price = data_1.ng_futures_contract3_price.interpolate(method ='linear', limit_direction ='backward')

data_1.ng_futures_contract4_price = data_1.ng_futures_contract4_price.interpolate(method ='linear', limit_direction ='backward')

data_1.info()
data_1.describe(include='all')
fig = px.bar(data_1, x='date', y='henry_hub_ng_spot_price',title='Historical Daily Natural Gas Spot',labels={'henry_hub_ng_spot_price':'Henry Hub NG Spot Price','ng_futures_contract1_price':'Contract 1','ng_futures_contract2_price':'Contract 2','ng_futures_contract3_price':'Contract 3','ng_futures_contract4_price':'Contract 4',},color='henry_hub_ng_spot_price')

iplot(fig)
fig = px.scatter(data_1, x="date", y="henry_hub_ng_spot_price", color="henry_hub_ng_spot_price", marginal_y="violin", marginal_x="box", trendline="ols", template="simple_white") #trendline = "lowess"

fig.show()
fig = px.scatter(data_1, x="henry_hub_ng_spot_price", y="ng_futures_contract1_price", color="henry_hub_ng_spot_price", marginal_y="violin", marginal_x="box", trendline="ols", template="simple_white")

fig.show()
fig = px.scatter(data_1, x="henry_hub_ng_spot_price", y="ng_futures_contract2_price", color="henry_hub_ng_spot_price", marginal_y="violin", marginal_x="box", trendline="ols", template="simple_white")

fig.show()
fig = px.scatter(data_1, x="henry_hub_ng_spot_price", y="ng_futures_contract3_price", color="henry_hub_ng_spot_price", marginal_y="violin",marginal_x="box", trendline="ols", template="simple_white")

fig.show()
fig = px.scatter(data_1, x="henry_hub_ng_spot_price", y="ng_futures_contract4_price", color="henry_hub_ng_spot_price", marginal_y="violin", marginal_x="box", trendline="ols", template="simple_white")

fig.show()
data_1.describe(include='all')
data_2 = pd.DataFrame()

for k in range(1,95):

    url_2 = f'https://www.altadata.io/data/api/{product_code_2}?api_key={altadata_io_api_key}&station_id=72243012960&observation_local_time_gte=2015-01-01&/?format=json&page={k}'

    data_2 = data_2.append(pd.read_json(json.dumps(requests.get(url_2).json())))
data_2 = data_2[['observation_local_time','wind_direction_angle','wind_direction','wind_type_code','wind_speed_ms','ceiling_height_dimension','visibility_distance_dimension','air_temp_c','dew_temp_c','relative_humidity','sea_level_pressure','precipitation_in_1_hr','sky_cond','weather_cond']]
data_2.info()
data_2.head()
data_2.tail()
data_2.describe(include='all')
date = []

for i in data_2.observation_local_time:

    i = str(i)

    d1 = datetime.datetime.strptime(i,"%Y-%m-%d %H:%M:%S+00:00")

    new_format = "%Y-%m-%d %H:%M:%S"

    date.append(d1.strftime(new_format))

data_2.insert(0,'date',date,True)
generated_date_list = []

check_df_date_list = []

difference_date_list = set()

dif_date = []
x = datetime.datetime(2015, 1, 1)

y = datetime.datetime(2020, 7, 20)

generated_date_list = generate_hours_range(x,y)

check_df_date_list = pd.to_datetime(data_2['date'].values)

difference_date_list = set(generated_date_list) - set(check_df_date_list)

dif_date = list(difference_date_list)
for i in dif_date:

    data_2 = data_2.append({'date' : i} , ignore_index=True)

data_2.date = pd.to_datetime(data_2.date)

data_2 = data_2.sort_values(by='date')
new_index = []

for i in range(0,len(data_2)):

    new_index.append(i)

data_2.insert(0,'ind',new_index,True)

data_2.index = data_2.ind
data_2.info()
datehour = []

for i in data_2.date:

    i=str(i)

    d1 = datetime.datetime.strptime(i,"%Y-%m-%d %H:%M:%S")

    new_format_date = "%m-%d %H:%M:%S"

    datehour.append(d1.strftime(new_format_date))

data_2.insert(2,'datehour',datehour,True)
missdate = pd.DataFrame()

missdate = data_2[data_2.air_temp_c.astype(str)=='nan']

val = []

for i in missdate.datehour:

    val.append(np.mean(data_2.air_temp_c[data_2.datehour == i]))
miss = pd.DataFrame()

miss.insert(0,'ind',missdate.ind,True)

miss.insert(1,'means',val,True)
data_2_1 = data_2.copy()

for i in miss.ind:

    data_2_1.air_temp_c[i] = miss.means[miss.ind == i]
trace_0 = go.Scatter(x = data_2_1.date,y = data_2_1.air_temp_c, name = "Filled Values of Air Temp C",

                       line = dict(color = '#173F5F'), opacity = 1.0)

trace_1 = go.Scatter(x = data_2.date,y = data_2.air_temp_c, name = "Air Temp C",

                       line = dict(color = '#17BECF'), opacity = 1.0)

data = [trace_0,trace_1]

layout = dict( title = 'Air Temp C')

fig = dict(data=data, layout=layout)

iplot(fig, filename = 'air.html')
missdate = pd.DataFrame()

missdate = data_2[data_2.wind_direction_angle.astype(str)=='nan']

val = []

for i in missdate.datehour:

    val.append(np.mean(data_2.wind_direction_angle[data_2.datehour == i]))
miss = pd.DataFrame()

miss.insert(0,'ind',missdate.ind,True)

miss.insert(1,'means',val,True)
for i in miss.ind:

    data_2_1.wind_direction_angle[i] = miss.means[miss.ind == i]
trace_0 = go.Scatter(x = data_2_1.date,y = data_2_1.wind_direction_angle, name = "Filled Values of Wind Direction Angle",

                       line = dict(color = '#173F5F'), opacity = 1.0)

trace_1 = go.Scatter(x = data_2.date,y = data_2.wind_direction_angle, name = "Wind Direction Angle",

                       line = dict(color = '#17BECF'), opacity = 1.0)

data = [trace_0,trace_1]

layout = dict( title = 'Wind Direction Angle')

fig = dict(data=data, layout=layout)

iplot(fig, filename = 'wind-direction-angle.html')
missdate = pd.DataFrame()

missdate = data_2[data_2.wind_speed_ms.astype(str)=='nan']

val = []

for i in missdate.datehour:

    val.append(np.mean(data_2.wind_speed_ms[data_2.datehour == i]))
miss = pd.DataFrame()

miss.insert(0,'ind',missdate.ind,True)

miss.insert(1,'means',val,True)
for i in miss.ind:

    data_2_1.wind_speed_ms[i] = miss.means[miss.ind == i]
trace_0 = go.Scatter(x = data_2_1.date,y = data_2_1.wind_speed_ms, name = "Filled Values of Wind Speed MS",

                       line = dict(color = '#173F5F'), opacity = 1.0)

trace_1 = go.Scatter(x = data_2.date,y = data_2.wind_speed_ms, name = "Wind Speed MS",

                       line = dict(color = '#17BECF'), opacity = 1.0)

data = [trace_0,trace_1]

layout = dict( title = 'Wind Speed MS')

fig = dict(data=data, layout=layout)

iplot(fig, filename = 'wind-speed-ms.html')
missdate = pd.DataFrame()

missdate = data_2[data_2.ceiling_height_dimension.astype(str)=='nan']

val = []

for i in missdate.datehour:

    val.append(np.mean(data_2.ceiling_height_dimension[data_2.datehour == i]))
miss = pd.DataFrame()

miss.insert(0,'ind',missdate.ind,True)

miss.insert(1,'means',val,True)
for i in miss.ind:

    data_2_1.ceiling_height_dimension[i] = miss.means[miss.ind == i]
trace_0 = go.Scatter(x = data_2_1.date,y = data_2_1.ceiling_height_dimension, name = "Filled Values of Ceiling Height Dimension",

                       line = dict(color = '#173F5F'), opacity = 1.0)

trace_1 = go.Scatter(x = data_2.date,y = data_2.ceiling_height_dimension, name = "Ceiling Height Dimension",

                       line = dict(color = '#17BECF'), opacity = 1.0)

data = [trace_0,trace_1]

layout = dict( title = 'Ceiling Height Dimension')

fig = dict(data=data, layout=layout)

iplot(fig, filename = 'ceiling-height-dimension.html')
missdate = pd.DataFrame()

missdate = data_2[data_2.dew_temp_c.astype(str)=='nan']

val = []

for i in missdate.datehour:

    val.append(np.mean(data_2.dew_temp_c[data_2.datehour == i]))
miss = pd.DataFrame()

miss.insert(0,'ind',missdate.ind,True)

miss.insert(1,'means',val,True)
for i in miss.ind:

    data_2_1.dew_temp_c[i] = miss.means[miss.ind == i]
trace_0 = go.Scatter(x = data_2_1.date,y = data_2_1.dew_temp_c, name = "Filled Values of Dew Temp C",

                       line = dict(color = '#173F5F'), opacity = 1.0)

trace_1 = go.Scatter(x = data_2.date,y = data_2.dew_temp_c, name = "Dew Temp C",

                       line = dict(color = '#17BECF'), opacity = 1.0)

data = [trace_0,trace_1]

layout = dict( title = 'Dew Temp C')

fig = dict(data=data, layout=layout)

iplot(fig, filename = 'dew-temp-c.html')
missdate = pd.DataFrame()

missdate = data_2[data_2.relative_humidity.astype(str)=='nan']

val = []

for i in missdate.datehour:

    val.append(np.mean(data_2.relative_humidity[data_2.datehour == i]))
miss = pd.DataFrame()

miss.insert(0,'ind',missdate.ind,True)

miss.insert(1,'means',val,True)
for i in miss.ind:

    data_2_1.relative_humidity[i] = miss.means[miss.ind == i]
trace_0 = go.Scatter(x = data_2_1.date,y = data_2_1.relative_humidity, name = "Filled Values of Relative Humidity",

                       line = dict(color = '#173F5F'), opacity = 1.0)

trace_1 = go.Scatter(x = data_2.date,y = data_2.relative_humidity, name = "Relative Humidity",

                       line = dict(color = '#17BECF'), opacity = 1.0)

data = [trace_0,trace_1]

layout = dict( title = 'Relative Humidity')

fig = dict(data=data, layout=layout)

iplot(fig, filename = 'relative-humidity.html')
missdate = pd.DataFrame

missdate = data_2[data_2.sea_level_pressure.astype(str)=='nan']

val = []

for i in missdate.datehour:

    val.append(np.mean(data_2.sea_level_pressure[data_2.datehour == i]))
miss = pd.DataFrame()

miss.insert(0,'ind',missdate.ind,True)

miss.insert(1,'means',val,True)
for i in miss.ind:

    data_2_1.sea_level_pressure[i] = miss.means[miss.ind == i]
trace_0 = go.Scatter(x = data_2_1.date,y = data_2_1.sea_level_pressure, name = "Filled Values of Sea Level Pressure",

                       line = dict(color = '#173F5F'), opacity = 1.0)

trace_1 = go.Scatter(x = data_2.date,y = data_2.sea_level_pressure, name = "Sea Level Pressure",

                       line = dict(color = '#17BECF'), opacity = 1.0)

data = [trace_0,trace_1]

layout = dict( title = 'Sea Level Pressure')

fig = dict(data=data, layout=layout)

iplot(fig, filename = 'sea-level-pressure.html')
missdate = pd.DataFrame()

missdate = data_2[data_2.precipitation_in_1_hr.astype(str)=='nan']

val = []

for i in missdate.datehour:

    val.append(np.mean(data_2.precipitation_in_1_hr[data_2.datehour == i]))
miss = pd.DataFrame()

miss.insert(0,'ind',missdate.ind,True)

miss.insert(1,'means',val,True)
for i in miss.ind:

    data_2_1.precipitation_in_1_hr[i] = miss.means[miss.ind == i]
trace_0 = go.Scatter(x = data_2_1.date,y = data_2_1.precipitation_in_1_hr, name = "Filling Precipitation in 1 Hour",

                       line = dict(color = '#173F5F'), opacity = 1.0)

trace_1 = go.Scatter(x = data_2.date,y = data_2.precipitation_in_1_hr, name = "Precipitation in 1 Hour",

                       line = dict(color = '#17BECF'), opacity = 1.0)

data = [trace_0,trace_1]

layout = dict( title = 'Precipitation in 1 Hour')

fig = dict(data=data, layout=layout)

iplot(fig, filename = 'precipitation-in-1-hour.html')
del data_2

data_2 = data_2_1.copy()

del data_2_1
data_2.info()
trace1 = go.Scatter(x = data_2.date, y = data_2.air_temp_c, mode = "lines", name = "air temp c", 

                    marker = dict(color = 'rgba(16, 112, 2, 1.0)'), text= data_2.wind_type_code)

trace2 = go.Scatter(x = data_2.date, y = data_2.dew_temp_c, mode = "lines+markers", name = "dew temp c",

                    marker = dict(color = 'rgba(80, 26, 80, 0.5)'), text= data_2.wind_type_code)

data = [trace1, trace2]

layout = dict(title = 'Air Temp C & Dew Temp C', xaxis= dict(title= 'Local Time',ticklen= 5,zeroline= False))

fig = dict(data = data, layout = layout)

iplot(fig,filename='airvsdew.html')
labels_wd = data_2.wind_direction.value_counts().index

values_wd = data_2.wind_direction.value_counts().values

fig = go.Figure(data=[go.Pie(labels=labels_wd, values=values_wd, textinfo='label+percent',

                             insidetextorientation='radial')])

iplot(fig,filename='wd.html')
labels_wtc = data_2.wind_type_code.value_counts().index

values_wtc = data_2.wind_type_code.value_counts().values

fig = go.Figure(data=[go.Pie(labels=labels_wtc, values=values_wtc,

                           textinfo='label+percent',insidetextorientation='radial')])

iplot(fig,filename='wtc.html')
labels_sky = data_2.sky_cond.value_counts().index

values_sky = data_2.sky_cond.value_counts().values

fig = go.Figure([go.Bar(x=labels_sky, y=values_sky)])

iplot(fig,filename='sky-cond.html')
labels_weather = data_2.weather_cond.value_counts().index

values_weather = data_2.weather_cond.value_counts().values

trace1 = go.Bar(

                x = labels_weather,

                y = values_weather,

                name = "Weather Condition",

                marker = dict(color = 'rgba(255, 255, 128, 0.5)',

                              line=dict(color='rgb(0,0,0)',width=1.5)),

                text = data_2.weather_cond)

data = [trace1]

layout = go.Layout(title='Weather Condition',barmode = "group")

fig = go.Figure(data = data, layout = layout)

iplot(fig,filename='barchart-sky-whet-2.html')
data_2 = data_2[['date','datehour','wind_direction_angle','wind_speed_ms','ceiling_height_dimension','air_temp_c','dew_temp_c','relative_humidity','sea_level_pressure','precipitation_in_1_hr',]]

data_2.head()
data_0 = data_0.loc[0:194591]

data_1 = data_1.loc[0:2026]

data_2 = data_2.loc[0:48648]
hour = []

mean_price = []

m = 0

for i in range(0,len(data_0),4):

    hour.append(data_0.date[i])

    mean_price.append(data_0.meanprice[i])
henry = []

con1 = []

con2 = []

con3 = []

con4 = []

date = []

for i in range(0,len(data_1)):

    for j in range(0,24):

        henry.append(data_1.henry_hub_ng_spot_price[i])

        con1.append(data_1.ng_futures_contract1_price[i])

        con2.append(data_1.ng_futures_contract2_price[i])

        con3.append(data_1.ng_futures_contract3_price[i])

        con4.append(data_1.ng_futures_contract4_price[i])

        date.append(data_1.date[i])
df = pd.DataFrame()

df.insert(0,'hour',hour,True)

df.insert(1,'mean_price',mean_price,True)

df.insert(2,"henry",henry,True)

df.insert(3,'ng_futures_contract1_price',con1,True)

df.insert(4,'ng_futures_contract2_price',con2,True)

df.insert(5,'ng_futures_contract3_price',con3,True)

df.insert(6,'ng_futures_contract4_price',con4,True)

df.insert(7,'wind_speed_ms',data_2.wind_speed_ms,True)

df.insert(8,'ceiling_height_dimension',data_2.ceiling_height_dimension,True)

df.insert(9,'air_temp_c',data_2.air_temp_c,True)

df.insert(10,'dew_temp_c',data_2.dew_temp_c,True)

df.insert(11,'relative_humidity',data_2.relative_humidity,True)

df.insert(12,'sea_level_pressure',data_2.sea_level_pressure,True)

df.insert(13,'precipitation_in_1_hr',data_2.precipitation_in_1_hr,True)

df.info()
df.info()
df.head()
df.tail()
df.describe(include='all')
hourpeak = []

dateday = []

datemonth = []

dayyear = []

for i in df.hour:

    i = str(i)

    d1 = datetime.datetime.strptime(i,"%Y-%m-%d %H:%M:%S")

    new_format_date_day = "%m-%d"

    new_format_hour_peak = "%H:00:00"

    new_format_date_month = "%m"

    new_format_day_year = "%Y-%m"

    hourpeak.append(d1.strftime(new_format_hour_peak))

    dateday.append(d1.strftime(new_format_date_day))

    datemonth.append(d1.strftime(new_format_date_month))

    dayyear.append(d1.strftime(new_format_day_year))
df.insert(2,'peakhour',hourpeak,True)

df.insert(3,'dateday',dateday,True)

df.insert(4,'month',datemonth,True)

df.insert(5,'dayyear',dayyear,True)
df.head()
outlier = detect_outlier(df, 'mean_price', 'hour')

outlier.head(len(outlier))
fig = px.histogram(df, x="mean_price",marginal="box")

iplot(fig, filename='first_histogram.html')
df = df.drop(df.index[c] for c in outlier.ind)

trace_0 = go.Scatter(x = df.hour,y = df.mean_price, name = "Point Price", line = dict(color = '#17BECF'), opacity = 1.0)

data = [trace_0]

layout = dict( title = 'HB_SOUTH Price')

fig = dict(data=data, layout=layout)

iplot(fig, filename = 'notoutlier.html')
fig = px.histogram(df, x="mean_price",marginal="rug")

iplot(fig, filename='non-outlier_histogram.html')
df.hour = pd.to_datetime(df.hour)

hour = df['hour'].dt.hour

dayofweek = df['hour'].dt.dayofweek

quarter = df['hour'].dt.quarter

month = df['hour'].dt.month

year = df['hour'].dt.year

dayofyear = df['hour'].dt.dayofyear

dayofmonth = df['hour'].dt.day

weekofyear = df['hour'].dt.weekofyear
df.insert(0,'hours',hour,True)

df.insert(1,'dayofweek',dayofweek,True)

df.insert(2,'quarter',quarter,True)

df.insert(3,'month',month,True)

df.insert(4,'year',year,True)

df.insert(5,'dayofyear',dayofyear,True)

df.insert(6,'dayofmonth',dayofmonth,True)

df.insert(7,'weekofyear',weekofyear,True)
new_index = []

for i in range(0,len(df)):

    new_index.append(i)

df.insert(0,'ind',new_index,True)

df.info()
ft = ['hour','mean_price','henry','wind_speed_ms','ceiling_height_dimension','air_temp_c','dew_temp_c','relative_humidity','sea_level_pressure','precipitation_in_1_hr']

data = df.copy()

data = data[ft]

data.hour = pd.to_datetime(data.hour)

hour = data['hour'].dt.hour

dayofweek = data['hour'].dt.dayofweek

quarter = data['hour'].dt.quarter

month = data['hour'].dt.month

year = data['hour'].dt.year

dayofyear = data['hour'].dt.dayofyear

dayofmonth = data['hour'].dt.day

weekofyear = data['hour'].dt.weekofyear

data.insert(0,'hours',hour,True)

data.insert(1,'dayofweek',dayofweek,True)

data.insert(2,'quarter',quarter,True)

data.insert(3,'month',month,True)

data.insert(4,'year',year,True)

data.insert(5,'dayofyear',dayofyear,True)

data.insert(6,'dayofmonth',dayofmonth,True)

data.insert(7,'weekofyear',weekofyear,True)
jan = data[data.month == 1]

day = []

x = []

for i in range(0,24):

    x.append(np.mean(jan.mean_price[jan.hours == i]))

    day.append(i)

first = pd.DataFrame()

first.insert(0,'hours',day,True)

first.insert(1,'january',x,True)

feb = data[data.month == 2]

x = []

for i in range(0,24):

    x.append(np.mean(feb.mean_price[feb.hours == i]))

first.insert(2,'february',x,True)

feb = data[data.month == 3]

x = []

for i in range(0,24):

    x.append(np.mean(feb.mean_price[feb.hours == i]))

first.insert(3,'march',x,True)

feb = data[data.month == 4]

x = []

for i in range(0,24):

    x.append(np.mean(feb.mean_price[feb.hours == i]))

first.insert(4,'april',x,True)

feb = data[data.month == 5]

x = []

for i in range(0,24):

    x.append(np.mean(feb.mean_price[feb.hours == i]))

first.insert(5,'may',x,True)

feb = data[data.month == 6]

x = []

for i in range(0,24):

    x.append(np.mean(feb.mean_price[feb.hours == i]))

first.insert(6,'june',x,True)

feb = data[data.month == 7]

x = []

for i in range(0,24):

    x.append(np.mean(feb.mean_price[feb.hours == i]))

first.insert(7,'july',x,True)

feb = data[data.month == 8]

x = []

for i in range(0,24):

    x.append(np.mean(feb.mean_price[feb.hours == i]))

first.insert(8,'august',x,True)

feb = data[data.month == 9]

x = []

for i in range(0,24):

    x.append(np.mean(feb.mean_price[feb.hours == i]))

first.insert(9,'september',x,True)

feb = data[data.month == 10]

x = []

for i in range(0,24):

    x.append(np.mean(feb.mean_price[feb.hours == i]))

first.insert(10,'october',x,True)

feb = data[data.month == 11]

x = []

for i in range(0,24):

    x.append(np.mean(feb.mean_price[feb.hours == i]))

first.insert(11,'november',x,True)

feb = data[data.month == 12]

x = []

for i in range(0,24):

    x.append(np.mean(feb.mean_price[feb.hours == i]))

first.insert(12,'december',x,True)

ft = ['january','february','march','april','may','june','july','august','september','october','november','december']

plt.figure(figsize=(20,15))

plt.title('Electricity price for the average hourly with monthly view for years')

ax = sns.heatmap(first[ft],linewidth=.5, annot=True, fmt="f")

plt.show()
jan = data[data.year == 2015]

jan = jan[jan.month == 1]

day = []

x = []

for i in range(0,24):

    x.append(np.mean(jan.mean_price[jan.hours == i]))

    day.append(i)

first = pd.DataFrame()

first.insert(0,'hours',day,True)

first.insert(1,'january',x,True)

feb = data[data.year == 2015]

feb = feb[feb.month == 2]

x = []

for i in range(0,24):

    x.append(np.mean(feb.mean_price[feb.hours == i]))

first.insert(2,'february',x,True)

feb = data[data.year == 2015]

feb = feb[feb.month == 3]

x = []

for i in range(0,24):

    x.append(np.mean(feb.mean_price[feb.hours == i]))

first.insert(3,'march',x,True)

feb = data[data.year == 2015]

feb = feb[feb.month == 4]

x = []

for i in range(0,24):

    x.append(np.mean(feb.mean_price[feb.hours == i]))

first.insert(4,'april',x,True)

feb = data[data.year == 2015]

feb = feb[feb.month == 5]

x = []

for i in range(0,24):

    x.append(np.mean(feb.mean_price[feb.hours == i]))

first.insert(5,'may',x,True)

feb = data[data.year == 2015]

feb = feb[feb.month == 6]

x = []

for i in range(0,24):

    x.append(np.mean(feb.mean_price[feb.hours == i]))

first.insert(6,'june',x,True)

feb = data[data.year == 2015]

feb = feb[feb.month == 7]

x = []

for i in range(0,24):

    x.append(np.mean(feb.mean_price[feb.hours == i]))

first.insert(7,'july',x,True)

feb = data[data.year == 2015]

feb = feb[feb.month == 8]

x = []

for i in range(0,24):

    x.append(np.mean(feb.mean_price[feb.hours == i]))

first.insert(8,'august',x,True)

feb = data[data.year == 2015]

feb = feb[feb.month == 9]

x = []

for i in range(0,24):

    x.append(np.mean(feb.mean_price[feb.hours == i]))

first.insert(9,'september',x,True)

feb = data[data.year == 2015]

feb = feb[feb.month == 10]

x = []

for i in range(0,24):

    x.append(np.mean(feb.mean_price[feb.hours == i]))

first.insert(10,'october',x,True)

feb = data[data.year == 2015]

feb = feb[feb.month == 11]

x = []

for i in range(0,24):

    x.append(np.mean(feb.mean_price[feb.hours == i]))

first.insert(11,'november',x,True)

feb = data[data.year == 2015]

feb = feb[feb.month == 12]

x = []

for i in range(0,24):

    x.append(np.mean(feb.mean_price[feb.hours == i]))

first.insert(12,'december',x,True)

ft = ['january','february','march','april','may','june','july','august','september','october','november','december']

plt.figure(figsize=(20,15))

plt.title('Electricity price for the average hourly with a monthly view for 2015')

ax = sns.heatmap(first[ft],linewidth=.5, annot=True, fmt="f")

plt.show()
jan = data[data.year == 2016]

jan = jan[jan.month == 1]

day = []

x = []

for i in range(0,24):

    x.append(np.mean(jan.mean_price[jan.hours == i]))

    day.append(i)

second = pd.DataFrame()

second.insert(0,'hours',day,True)

second.insert(1,'january',x,True)

feb = data[data.year == 2016]

feb = feb[feb.month == 2]

x = []

for i in range(0,24):

    x.append(np.mean(feb.mean_price[feb.hours == i]))

second.insert(2,'february',x,True)

feb = data[data.year == 2016]

feb = feb[feb.month == 3]

x = []

for i in range(0,24):

    x.append(np.mean(feb.mean_price[feb.hours == i]))

second.insert(3,'march',x,True)

feb = data[data.year == 2016]

feb = feb[feb.month == 4]

x = []

for i in range(0,24):

    x.append(np.mean(feb.mean_price[feb.hours == i]))

second.insert(4,'april',x,True)

feb = data[data.year == 2016]

feb = feb[feb.month == 5]

x = []

for i in range(0,24):

    x.append(np.mean(feb.mean_price[feb.hours == i]))

second.insert(5,'may',x,True)

feb = data[data.year == 2016]

feb = feb[feb.month == 6]

x = []

for i in range(0,24):

    x.append(np.mean(feb.mean_price[feb.hours == i]))

second.insert(6,'june',x,True)

feb = data[data.year == 2016]

feb = feb[feb.month == 7]

x = []

for i in range(0,24):

    x.append(np.mean(feb.mean_price[feb.hours == i]))

second.insert(7,'july',x,True)

feb = data[data.year == 2016]

feb = feb[feb.month == 8]

x = []

for i in range(0,24):

    x.append(np.mean(feb.mean_price[feb.hours == i]))

second.insert(8,'august',x,True)

feb = data[data.year == 2016]

feb = feb[feb.month == 9]

x = []

for i in range(0,24):

    x.append(np.mean(feb.mean_price[feb.hours == i]))

second.insert(9,'september',x,True)

feb = data[data.year == 2016]

feb = feb[feb.month == 10]

x = []

for i in range(0,24):

    x.append(np.mean(feb.mean_price[feb.hours == i]))

second.insert(10,'october',x,True)

feb = data[data.year == 2016]

feb = feb[feb.month == 11]

x = []

for i in range(0,24):

    x.append(np.mean(feb.mean_price[feb.hours == i]))

second.insert(11,'november',x,True)

feb = data[data.year == 2016]

feb = feb[feb.month == 12]

x = []

for i in range(0,24):

    x.append(np.mean(feb.mean_price[feb.hours == i]))

second.insert(12,'december',x,True)

ft = ['january','february','march','april','may','june','july','august','september','october','november','december']

plt.figure(figsize=(20,15))

plt.title('Electricity price for the average hourly with a monthly view for 2016')

ax = sns.heatmap(second[ft],linewidth=.5, annot=True, fmt="f")

plt.show()
jan = data[data.year == 2017]

jan = jan[jan.month == 1]

day = []

x = []

for i in range(0,24):

    x.append(np.mean(jan.mean_price[jan.hours == i]))

    day.append(i)

second = pd.DataFrame()

second.insert(0,'hours',day,True)

second.insert(1,'january',x,True)

feb = data[data.year == 2017]

feb = feb[feb.month == 2]

x = []

for i in range(0,24):

    x.append(np.mean(feb.mean_price[feb.hours == i]))

second.insert(2,'february',x,True)

feb = data[data.year == 2017]

feb = feb[feb.month == 3]

x = []

for i in range(0,24):

    x.append(np.mean(feb.mean_price[feb.hours == i]))

second.insert(3,'march',x,True)

feb = data[data.year == 2017]

feb = feb[feb.month == 4]

x = []

for i in range(0,24):

    x.append(np.mean(feb.mean_price[feb.hours == i]))

second.insert(4,'april',x,True)

feb = data[data.year == 2017]

feb = feb[feb.month == 5]

x = []

for i in range(0,24):

    x.append(np.mean(feb.mean_price[feb.hours == i]))

second.insert(5,'may',x,True)

feb = data[data.year == 2017]

feb = feb[feb.month == 6]

x = []

for i in range(0,24):

    x.append(np.mean(feb.mean_price[feb.hours == i]))

second.insert(6,'june',x,True)

feb = data[data.year == 2017]

feb = feb[feb.month == 7]

x = []

for i in range(0,24):

    x.append(np.mean(feb.mean_price[feb.hours == i]))

second.insert(7,'july',x,True)

feb = data[data.year == 2017]

feb = feb[feb.month == 8]

x = []

for i in range(0,24):

    x.append(np.mean(feb.mean_price[feb.hours == i]))

second.insert(8,'august',x,True)

feb = data[data.year == 2017]

feb = feb[feb.month == 9]

x = []

for i in range(0,24):

    x.append(np.mean(feb.mean_price[feb.hours == i]))

second.insert(9,'september',x,True)

feb = data[data.year == 2017]

feb = feb[feb.month == 10]

x = []

for i in range(0,24):

    x.append(np.mean(feb.mean_price[feb.hours == i]))

second.insert(10,'october',x,True)

feb = data[data.year == 2017]

feb = feb[feb.month == 11]

x = []

for i in range(0,24):

    x.append(np.mean(feb.mean_price[feb.hours == i]))

second.insert(11,'november',x,True)

feb = data[data.year == 2017]

feb = feb[feb.month == 12]

x = []

for i in range(0,24):

    x.append(np.mean(feb.mean_price[feb.hours == i]))

second.insert(12,'december',x,True)

ft = ['january','february','march','april','may','june','july','august','september','october','november','december']

plt.figure(figsize=(20,15))

plt.title('Electricity price for the average hourly with a monthly view for 2017')

ax = sns.heatmap(second[ft],linewidth=.5, annot=True, fmt="f")

plt.show()
jan = data[data.year == 2018]

jan = jan[jan.month == 1]

day = []

x = []

for i in range(0,24):

    x.append(np.mean(jan.mean_price[jan.hours == i]))

    day.append(i)

second = pd.DataFrame()

second.insert(0,'hours',day,True)

second.insert(1,'january',x,True)

feb = data[data.year == 2018]

feb = feb[feb.month == 2]

x = []

for i in range(0,24):

    x.append(np.mean(feb.mean_price[feb.hours == i]))

second.insert(2,'february',x,True)

feb = data[data.year == 2018]

feb = feb[feb.month == 3]

x = []

for i in range(0,24):

    x.append(np.mean(feb.mean_price[feb.hours == i]))

second.insert(3,'march',x,True)

feb = data[data.year == 2018]

feb = feb[feb.month == 4]

x = []

for i in range(0,24):

    x.append(np.mean(feb.mean_price[feb.hours == i]))

second.insert(4,'april',x,True)

feb = data[data.year == 2018]

feb = feb[feb.month == 5]

x = []

for i in range(0,24):

    x.append(np.mean(feb.mean_price[feb.hours == i]))

second.insert(5,'may',x,True)

feb = data[data.year == 2018]

feb = feb[feb.month == 6]

x = []

for i in range(0,24):

    x.append(np.mean(feb.mean_price[feb.hours == i]))

second.insert(6,'june',x,True)

feb = data[data.year == 2018]

feb = feb[feb.month == 7]

x = []

for i in range(0,24):

    x.append(np.mean(feb.mean_price[feb.hours == i]))

second.insert(7,'july',x,True)

feb = data[data.year == 2018]

feb = feb[feb.month == 8]

x = []

for i in range(0,24):

    x.append(np.mean(feb.mean_price[feb.hours == i]))

second.insert(8,'august',x,True)

feb = data[data.year == 2018]

feb = feb[feb.month == 9]

x = []

for i in range(0,24):

    x.append(np.mean(feb.mean_price[feb.hours == i]))

second.insert(9,'september',x,True)

feb = data[data.year == 2018]

feb = feb[feb.month == 10]

x = []

for i in range(0,24):

    x.append(np.mean(feb.mean_price[feb.hours == i]))

second.insert(10,'october',x,True)

feb = data[data.year == 2018]

feb = feb[feb.month == 11]

x = []

for i in range(0,24):

    x.append(np.mean(feb.mean_price[feb.hours == i]))

second.insert(11,'november',x,True)

feb = data[data.year == 2018]

feb = feb[feb.month == 12]

x = []

for i in range(0,24):

    x.append(np.mean(feb.mean_price[feb.hours == i]))

second.insert(12,'december',x,True)

ft = ['january','february','march','april','may','june','july','august','september','october','november','december']

plt.figure(figsize=(20,15))

plt.title('Electricity price for the average hourly with a monthly view for 2018')

ax = sns.heatmap(second[ft],linewidth=.5, annot=True, fmt="f")

plt.show()
jan = data[data.year == 2019]

jan = jan[jan.month == 1]

day = []

x = []

for i in range(0,24):

    x.append(np.mean(jan.mean_price[jan.hours == i]))

    day.append(i)

second = pd.DataFrame()

second.insert(0,'hours',day,True)

second.insert(1,'january',x,True)

feb = data[data.year == 2019]

feb = feb[feb.month == 2]

x = []

for i in range(0,24):

    x.append(np.mean(feb.mean_price[feb.hours == i]))

second.insert(2,'february',x,True)

feb = data[data.year == 2019]

feb = feb[feb.month == 3]

x = []

for i in range(0,24):

    x.append(np.mean(feb.mean_price[feb.hours == i]))

second.insert(3,'march',x,True)

feb = data[data.year == 2019]

feb = feb[feb.month == 4]

x = []

for i in range(0,24):

    x.append(np.mean(feb.mean_price[feb.hours == i]))

second.insert(4,'april',x,True)

feb = data[data.year == 2019]

feb = feb[feb.month == 5]

x = []

for i in range(0,24):

    x.append(np.mean(feb.mean_price[feb.hours == i]))

second.insert(5,'may',x,True)

feb = data[data.year == 2019]

feb = feb[feb.month == 6]

x = []

for i in range(0,24):

    x.append(np.mean(feb.mean_price[feb.hours == i]))

second.insert(6,'june',x,True)

feb = data[data.year == 2019]

feb = feb[feb.month == 7]

x = []

for i in range(0,24):

    x.append(np.mean(feb.mean_price[feb.hours == i]))

second.insert(7,'july',x,True)

feb = data[data.year == 2019]

feb = feb[feb.month == 8]

x = []

for i in range(0,24):

    x.append(np.mean(feb.mean_price[feb.hours == i]))

second.insert(8,'august',x,True)

feb = data[data.year == 2019]

feb = feb[feb.month == 9]

x = []

for i in range(0,24):

    x.append(np.mean(feb.mean_price[feb.hours == i]))

second.insert(9,'september',x,True)

feb = data[data.year == 2019]

feb = feb[feb.month == 10]

x = []

for i in range(0,24):

    x.append(np.mean(feb.mean_price[feb.hours == i]))

second.insert(10,'october',x,True)

feb = data[data.year == 2019]

feb = feb[feb.month == 11]

x = []

for i in range(0,24):

    x.append(np.mean(feb.mean_price[feb.hours == i]))

second.insert(11,'november',x,True)

feb = data[data.year == 2019]

feb = feb[feb.month == 12]

x = []

for i in range(0,24):

    x.append(np.mean(feb.mean_price[feb.hours == i]))

second.insert(12,'december',x,True)

ft = ['january','february','march','april','may','june','july','august','september','october','november','december']

plt.figure(figsize=(20,15))

plt.title('Electricity price for the average hourly with a monthly view for 2019')

ax = sns.heatmap(second[ft],linewidth=.5, annot=True, fmt="f")

plt.show()
jan = data[data.year == 2020]

jan = jan[jan.month == 1]

day = []

x = []

for i in range(0,24):

    x.append(np.mean(jan.mean_price[jan.hours == i]))

    day.append(i)

second = pd.DataFrame()

second.insert(0,'hours',day,True)

second.insert(1,'january',x,True)

feb = data[data.year == 2020]

feb = feb[feb.month == 2]

x = []

for i in range(0,24):

    x.append(np.mean(feb.mean_price[feb.hours == i]))

second.insert(2,'february',x,True)

feb = data[data.year == 2020]

feb = feb[feb.month == 3]

x = []

for i in range(0,24):

    x.append(np.mean(feb.mean_price[feb.hours == i]))

second.insert(3,'march',x,True)

feb = data[data.year == 2020]

feb = feb[feb.month == 4]

x = []

for i in range(0,24):

    x.append(np.mean(feb.mean_price[feb.hours == i]))

second.insert(4,'april',x,True)

feb = data[data.year == 2020]

feb = feb[feb.month == 5]

x = []

for i in range(0,24):

    x.append(np.mean(feb.mean_price[feb.hours == i]))

second.insert(5,'may',x,True)

feb = data[data.year == 2020]

feb = feb[feb.month == 6]

x = []

for i in range(0,24):

    x.append(np.mean(feb.mean_price[feb.hours == i]))

second.insert(6,'june',x,True)

feb = data[data.year == 2020]

feb = feb[feb.month == 7]

x = []

for i in range(0,24):

    x.append(np.mean(feb.mean_price[feb.hours == i]))

second.insert(7,'july',x,True)

ft = ['january','february','march','april','may','june','july']

plt.figure(figsize=(15,8))

plt.title('Electricity price for the average hourly with a monthly view for 2020')

ax = sns.heatmap(second[ft],linewidth=.5, annot=True, fmt="f")

plt.show()
jan = data[data.year == 2015]

jan = jan[jan.month == 1]

day = []

x = []

for i in range(0,24):

    x.append(np.mean(jan.mean_price[jan.hours == i]))

    day.append(i)

first = pd.DataFrame()

first.insert(0,'hours',day,True)

first.insert(1,'2015',x,True)

jan = data[data.year == 2016]

jan = jan[jan.month == 1]

day = []

x = []

for i in range(0,24):

    x.append(np.mean(jan.mean_price[jan.hours == i]))

    day.append(i)

first.insert(2,'2016',x,True)

jan = data[data.year == 2017]

jan = jan[jan.month == 1]

day = []

x = []

for i in range(0,24):

    x.append(np.mean(jan.mean_price[jan.hours == i]))

    day.append(i)

first.insert(3,'2017',x,True)

jan = data[data.year == 2018]

jan = jan[jan.month == 1]

day = []

x = []

for i in range(0,24):

    x.append(np.mean(jan.mean_price[jan.hours == i]))

    day.append(i)

first.insert(4,'2018',x,True)

jan = data[data.year == 2019]

jan = jan[jan.month == 1]

day = []

x = []

for i in range(0,24):

    x.append(np.mean(jan.mean_price[jan.hours == i]))

    day.append(i)

first.insert(5,'2019',x,True)

jan = data[data.year == 2020]

jan = jan[jan.month == 1]

day = []

x = []

for i in range(0,24):

    x.append(np.mean(jan.mean_price[jan.hours == i]))

    day.append(i)

first.insert(6,'2020',x,True)

ft = ['2015','2016','2017','2018','2019','2020']

plt.figure(figsize=(15,8))

plt.title('Electricity price for the average hourly with a January view for years')

ax = sns.heatmap(first[ft],linewidth=.5, annot=True, fmt="f")

plt.show()
jan = data[data.year == 2015]

jan = jan[jan.month == 2]

day = []

x = []

for i in range(0,24):

    x.append(np.mean(jan.mean_price[jan.hours == i]))

    day.append(i)

first = pd.DataFrame()

first.insert(0,'hours',day,True)

first.insert(1,'2015',x,True)

jan = data[data.year == 2016]

jan = jan[jan.month == 2]

day = []

x = []

for i in range(0,24):

    x.append(np.mean(jan.mean_price[jan.hours == i]))

    day.append(i)

first.insert(2,'2016',x,True)

jan = data[data.year == 2017]

jan = jan[jan.month == 2]

day = []

x = []

for i in range(0,24):

    x.append(np.mean(jan.mean_price[jan.hours == i]))

    day.append(i)

first.insert(3,'2017',x,True)

jan = data[data.year == 2018]

jan = jan[jan.month == 2]

day = []

x = []

for i in range(0,24):

    x.append(np.mean(jan.mean_price[jan.hours == i]))

    day.append(i)

first.insert(4,'2018',x,True)

jan = data[data.year == 2019]

jan = jan[jan.month == 2]

day = []

x = []

for i in range(0,24):

    x.append(np.mean(jan.mean_price[jan.hours == i]))

    day.append(i)

first.insert(5,'2019',x,True)

jan = data[data.year == 2020]

jan = jan[jan.month == 2]

day = []

x = []

for i in range(0,24):

    x.append(np.mean(jan.mean_price[jan.hours == i]))

    day.append(i)

first.insert(6,'2020',x,True)

ft = ['2015','2016','2017','2018','2019','2020']

plt.figure(figsize=(15,8))

plt.title('Electricity price for the average hourly with a February view for years')

ax = sns.heatmap(first[ft],linewidth=.5, annot=True, fmt="f")

plt.show()
jan = data[data.year == 2015]

jan = jan[jan.month == 3]

day = []

x = []

for i in range(0,24):

    x.append(np.mean(jan.mean_price[jan.hours == i]))

    day.append(i)

first = pd.DataFrame()

first.insert(0,'hours',day,True)

first.insert(1,'2015',x,True)

jan = data[data.year == 2016]

jan = jan[jan.month == 3]

day = []

x = []

for i in range(0,24):

    x.append(np.mean(jan.mean_price[jan.hours == i]))

    day.append(i)

first.insert(2,'2016',x,True)

jan = data[data.year == 2017]

jan = jan[jan.month == 3]

day = []

x = []

for i in range(0,24):

    x.append(np.mean(jan.mean_price[jan.hours == i]))

    day.append(i)

first.insert(3,'2017',x,True)

jan = data[data.year == 2018]

jan = jan[jan.month == 3]

day = []

x = []

for i in range(0,24):

    x.append(np.mean(jan.mean_price[jan.hours == i]))

    day.append(i)

first.insert(4,'2018',x,True)

jan = data[data.year == 2019]

jan = jan[jan.month == 3]

day = []

x = []

for i in range(0,24):

    x.append(np.mean(jan.mean_price[jan.hours == i]))

    day.append(i)

first.insert(5,'2019',x,True)

jan = data[data.year == 3020]

jan = jan[jan.month == 3]

day = []

x = []

for i in range(0,24):

    x.append(np.mean(jan.mean_price[jan.hours == i]))

    day.append(i)

first.insert(6,'2020',x,True)

ft = ['2015','2016','2017','2018','2019','2020']

plt.figure(figsize=(15,8))

plt.title('Electricity price for the average hourly with a March view for years')

ax = sns.heatmap(first[ft],linewidth=.5, annot=True, fmt="f")

plt.show()
jan = data[data.year == 2015]

jan = jan[jan.month == 4]

day = []

x = []

for i in range(0,24):

    x.append(np.mean(jan.mean_price[jan.hours == i]))

    day.append(i)

first = pd.DataFrame()

first.insert(0,'hours',day,True)

first.insert(1,'2015',x,True)

jan = data[data.year == 2016]

jan = jan[jan.month == 4]

day = []

x = []

for i in range(0,24):

    x.append(np.mean(jan.mean_price[jan.hours == i]))

    day.append(i)

first.insert(2,'2016',x,True)

jan = data[data.year == 2017]

jan = jan[jan.month == 4]

day = []

x = []

for i in range(0,24):

    x.append(np.mean(jan.mean_price[jan.hours == i]))

    day.append(i)

first.insert(3,'2017',x,True)

jan = data[data.year == 2018]

jan = jan[jan.month == 4]

day = []

x = []

for i in range(0,24):

    x.append(np.mean(jan.mean_price[jan.hours == i]))

    day.append(i)

first.insert(4,'2018',x,True)

jan = data[data.year == 2019]

jan = jan[jan.month == 4]

day = []

x = []

for i in range(0,24):

    x.append(np.mean(jan.mean_price[jan.hours == i]))

    day.append(i)

first.insert(5,'2019',x,True)

jan = data[data.year == 4020]

jan = jan[jan.month == 4]

day = []

x = []

for i in range(0,24):

    x.append(np.mean(jan.mean_price[jan.hours == i]))

    day.append(i)

first.insert(6,'2020',x,True)

ft = ['2015','2016','2017','2018','2019','2020']

plt.figure(figsize=(15,8))

plt.title('Electricity price for the average hourly with a April view for years')

ax = sns.heatmap(first[ft],linewidth=.5, annot=True, fmt="f")

plt.show()
jan = data[data.year == 2015]

jan = jan[jan.month == 6]

day = []

x = []

for i in range(0,24):

    x.append(np.mean(jan.mean_price[jan.hours == i]))

    day.append(i)

first = pd.DataFrame()

first.insert(0,'hours',day,True)

first.insert(1,'2015',x,True)

jan = data[data.year == 2016]

jan = jan[jan.month == 6]

day = []

x = []

for i in range(0,24):

    x.append(np.mean(jan.mean_price[jan.hours == i]))

    day.append(i)

first.insert(2,'2016',x,True)

jan = data[data.year == 2017]

jan = jan[jan.month == 6]

day = []

x = []

for i in range(0,24):

    x.append(np.mean(jan.mean_price[jan.hours == i]))

    day.append(i)

first.insert(3,'2017',x,True)

jan = data[data.year == 2018]

jan = jan[jan.month == 6]

day = []

x = []

for i in range(0,24):

    x.append(np.mean(jan.mean_price[jan.hours == i]))

    day.append(i)

first.insert(4,'2018',x,True)

jan = data[data.year == 2019]

jan = jan[jan.month == 6]

day = []

x = []

for i in range(0,24):

    x.append(np.mean(jan.mean_price[jan.hours == i]))

    day.append(i)

first.insert(5,'2019',x,True)

jan = data[data.year == 6020]

jan = jan[jan.month == 6]

day = []

x = []

for i in range(0,24):

    x.append(np.mean(jan.mean_price[jan.hours == i]))

    day.append(i)

first.insert(6,'2020',x,True)

ft = ['2015','2016','2017','2018','2019','2020']

plt.figure(figsize=(15,8))

plt.title('Electricity price for the average hourly with a June view for years')

ax = sns.heatmap(first[ft],linewidth=.5, annot=True, fmt="f")

plt.show()
jan = data[data.year == 2015]

jan = jan[jan.month == 7]

day = []

x = []

for i in range(0,24):

    x.append(np.mean(jan.mean_price[jan.hours == i]))

    day.append(i)

first = pd.DataFrame()

first.insert(0,'hours',day,True)

first.insert(1,'2015',x,True)

jan = data[data.year == 2016]

jan = jan[jan.month == 7]

day = []

x = []

for i in range(0,24):

    x.append(np.mean(jan.mean_price[jan.hours == i]))

    day.append(i)

first.insert(2,'2016',x,True)

jan = data[data.year == 2017]

jan = jan[jan.month == 7]

day = []

x = []

for i in range(0,24):

    x.append(np.mean(jan.mean_price[jan.hours == i]))

    day.append(i)

first.insert(3,'2017',x,True)

jan = data[data.year == 2018]

jan = jan[jan.month == 7]

day = []

x = []

for i in range(0,24):

    x.append(np.mean(jan.mean_price[jan.hours == i]))

    day.append(i)

first.insert(4,'2018',x,True)

jan = data[data.year == 2019]

jan = jan[jan.month == 7]

day = []

x = []

for i in range(0,24):

    x.append(np.mean(jan.mean_price[jan.hours == i]))

    day.append(i)

first.insert(5,'2019',x,True)

jan = data[data.year == 7020]

jan = jan[jan.month == 7]

day = []

x = []

for i in range(0,24):

    x.append(np.mean(jan.mean_price[jan.hours == i]))

    day.append(i)

first.insert(6,'2020',x,True)

ft = ['2015','2016','2017','2018','2019','2020']

plt.figure(figsize=(15,8))

plt.title('Electricity price for the average hourly with a July view for years')

ax = sns.heatmap(first[ft],linewidth=.5, annot=True, fmt="f")

plt.show()
jan = data[data.year == 2015]

jan = jan[jan.month == 8]

day = []

x = []

for i in range(0,24):

    x.append(np.mean(jan.mean_price[jan.hours == i]))

    day.append(i)

first = pd.DataFrame()

first.insert(0,'hours',day,True)

first.insert(1,'2015',x,True)

jan = data[data.year == 2016]

jan = jan[jan.month == 8]

day = []

x = []

for i in range(0,24):

    x.append(np.mean(jan.mean_price[jan.hours == i]))

    day.append(i)

first.insert(2,'2016',x,True)

jan = data[data.year == 2017]

jan = jan[jan.month == 8]

day = []

x = []

for i in range(0,24):

    x.append(np.mean(jan.mean_price[jan.hours == i]))

    day.append(i)

first.insert(3,'2017',x,True)

jan = data[data.year == 2018]

jan = jan[jan.month == 8]

day = []

x = []

for i in range(0,24):

    x.append(np.mean(jan.mean_price[jan.hours == i]))

    day.append(i)

first.insert(4,'2018',x,True)

jan = data[data.year == 2019]

jan = jan[jan.month == 8]

day = []

x = []

for i in range(0,24):

    x.append(np.mean(jan.mean_price[jan.hours == i]))

    day.append(i)

first.insert(5,'2019',x,True)

jan = data[data.year == 8020]

jan = jan[jan.month == 8]

day = []

x = []

for i in range(0,24):

    x.append(np.mean(jan.mean_price[jan.hours == i]))

    day.append(i)

first.insert(6,'2020',x,True)

ft = ['2015','2016','2017','2018','2019','2020']

plt.figure(figsize=(15,8))

plt.title('Electricity price for the average hourly with a August view for years')

ax = sns.heatmap(first[ft],linewidth=.5, annot=True, fmt="f")

plt.show()
jan = data[data.year == 2015]

jan = jan[jan.month == 9]

day = []

x = []

for i in range(0,24):

    x.append(np.mean(jan.mean_price[jan.hours == i]))

    day.append(i)

first = pd.DataFrame()

first.insert(0,'hours',day,True)

first.insert(1,'2015',x,True)

jan = data[data.year == 2016]

jan = jan[jan.month == 9]

day = []

x = []

for i in range(0,24):

    x.append(np.mean(jan.mean_price[jan.hours == i]))

    day.append(i)

first.insert(2,'2016',x,True)

jan = data[data.year == 2017]

jan = jan[jan.month == 9]

day = []

x = []

for i in range(0,24):

    x.append(np.mean(jan.mean_price[jan.hours == i]))

    day.append(i)

first.insert(3,'2017',x,True)

jan = data[data.year == 2018]

jan = jan[jan.month == 9]

day = []

x = []

for i in range(0,24):

    x.append(np.mean(jan.mean_price[jan.hours == i]))

    day.append(i)

first.insert(4,'2018',x,True)

jan = data[data.year == 2019]

jan = jan[jan.month == 9]

day = []

x = []

for i in range(0,24):

    x.append(np.mean(jan.mean_price[jan.hours == i]))

    day.append(i)

first.insert(5,'2019',x,True)

jan = data[data.year == 9020]

jan = jan[jan.month == 9]

day = []

x = []

for i in range(0,24):

    x.append(np.mean(jan.mean_price[jan.hours == i]))

    day.append(i)

first.insert(6,'2020',x,True)

ft = ['2015','2016','2017','2018','2019','2020']

plt.figure(figsize=(15,8))

plt.title('Electricity price for the average hourly with a September view for years')

ax = sns.heatmap(first[ft],linewidth=.5, annot=True, fmt="f")

plt.show()
jan = data[data.year == 2015]

jan = jan[jan.month == 10]

day = []

x = []

for i in range(0,24):

    x.append(np.mean(jan.mean_price[jan.hours == i]))

    day.append(i)

first = pd.DataFrame()

first.insert(0,'hours',day,True)

first.insert(1,'2015',x,True)

jan = data[data.year == 2016]

jan = jan[jan.month == 10]

day = []

x = []

for i in range(0,24):

    x.append(np.mean(jan.mean_price[jan.hours == i]))

    day.append(i)

first.insert(2,'2016',x,True)

jan = data[data.year == 2017]

jan = jan[jan.month == 10]

day = []

x = []

for i in range(0,24):

    x.append(np.mean(jan.mean_price[jan.hours == i]))

    day.append(i)

first.insert(3,'2017',x,True)

jan = data[data.year == 2018]

jan = jan[jan.month == 10]

day = []

x = []

for i in range(0,24):

    x.append(np.mean(jan.mean_price[jan.hours == i]))

    day.append(i)

first.insert(4,'2018',x,True)

jan = data[data.year == 2019]

jan = jan[jan.month == 10]

day = []

x = []

for i in range(0,24):

    x.append(np.mean(jan.mean_price[jan.hours == i]))

    day.append(i)

first.insert(5,'2019',x,True)

jan = data[data.year == 2020]

jan = jan[jan.month == 10]

day = []

x = []

for i in range(0,24):

    x.append(np.mean(jan.mean_price[jan.hours == i]))

    day.append(i)

first.insert(6,'2020',x,True)

ft = ['2015','2016','2017','2018','2019','2020']

plt.figure(figsize=(15,8))

plt.title('Electricity price for the average hourly with a October view for years')

ax = sns.heatmap(first[ft],linewidth=.5, annot=True, fmt="f")

plt.show()
jan = data[data.year == 2015]

jan = jan[jan.month == 11]

day = []

x = []

for i in range(0,24):

    x.append(np.mean(jan.mean_price[jan.hours == i]))

    day.append(i)

first = pd.DataFrame()

first.insert(0,'hours',day,True)

first.insert(1,'2015',x,True)

jan = data[data.year == 2016]

jan = jan[jan.month == 11]

day = []

x = []

for i in range(0,24):

    x.append(np.mean(jan.mean_price[jan.hours == i]))

    day.append(i)

first.insert(2,'2016',x,True)

jan = data[data.year == 2017]

jan = jan[jan.month == 11]

day = []

x = []

for i in range(0,24):

    x.append(np.mean(jan.mean_price[jan.hours == i]))

    day.append(i)

first.insert(3,'2017',x,True)

jan = data[data.year == 2018]

jan = jan[jan.month == 11]

day = []

x = []

for i in range(0,24):

    x.append(np.mean(jan.mean_price[jan.hours == i]))

    day.append(i)

first.insert(4,'2018',x,True)

jan = data[data.year == 2019]

jan = jan[jan.month == 11]

day = []

x = []

for i in range(0,24):

    x.append(np.mean(jan.mean_price[jan.hours == i]))

    day.append(i)

first.insert(5,'2019',x,True)

jan = data[data.year == 2020]

jan = jan[jan.month == 11]

day = []

x = []

for i in range(0,24):

    x.append(np.mean(jan.mean_price[jan.hours == i]))

    day.append(i)

first.insert(6,'2020',x,True)

ft = ['2015','2016','2017','2018','2019','2020']

plt.figure(figsize=(15,8))

plt.title('Electricity price for the average hourly with a November view for years')

ax = sns.heatmap(first[ft],linewidth=.5, annot=True, fmt="f")

plt.show()
jan = data[data.year == 2015]

jan = jan[jan.month == 12]

day = []

x = []

for i in range(0,24):

    x.append(np.mean(jan.mean_price[jan.hours == i]))

    day.append(i)

first = pd.DataFrame()

first.insert(0,'hours',day,True)

first.insert(1,'2015',x,True)

jan = data[data.year == 2016]

jan = jan[jan.month == 12]

day = []

x = []

for i in range(0,24):

    x.append(np.mean(jan.mean_price[jan.hours == i]))

    day.append(i)

first.insert(2,'2016',x,True)

jan = data[data.year == 2017]

jan = jan[jan.month == 12]

day = []

x = []

for i in range(0,24):

    x.append(np.mean(jan.mean_price[jan.hours == i]))

    day.append(i)

first.insert(3,'2017',x,True)

jan = data[data.year == 2018]

jan = jan[jan.month == 12]

day = []

x = []

for i in range(0,24):

    x.append(np.mean(jan.mean_price[jan.hours == i]))

    day.append(i)

first.insert(4,'2018',x,True)

jan = data[data.year == 2019]

jan = jan[jan.month == 12]

day = []

x = []

for i in range(0,24):

    x.append(np.mean(jan.mean_price[jan.hours == i]))

    day.append(i)

first.insert(5,'2019',x,True)

jan = data[data.year == 2020]

jan = jan[jan.month == 12]

day = []

x = []

for i in range(0,24):

    x.append(np.mean(jan.mean_price[jan.hours == i]))

    day.append(i)

first.insert(6,'2020',x,True)

ft = ['2015','2016','2017','2018','2019','2020']

plt.figure(figsize=(15,8))

plt.title('Electricity price for the average hourly with a December view for years')

ax = sns.heatmap(first[ft],linewidth=.5, annot=True, fmt="f")

plt.show()
f,ax = plt.subplots(figsize=(10, 6))

sns.heatmap(df.corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.1f',ax=ax)

plt.title('Heatmap')

plt.show()
fig = go.Figure()

fig.add_trace(go.Box(y=df.mean_price,name="Mean Price"))

fig.add_trace(go.Box(y=df.henry,name='Henry Hub NG Spot Price'))

fig.add_trace(go.Box(y=df.wind_speed_ms,name='Wind Speed MS'))

fig.add_trace(go.Box(y=df.ceiling_height_dimension,name='Ceiling Height Dimension'))

fig.add_trace(go.Box(y=df.air_temp_c,name='Air Temp C'))

fig.add_trace(go.Box(y=df.dew_temp_c,name='Dew Temp C'))

fig.add_trace(go.Box(y=df.relative_humidity,name='Relative Humidity'))

fig.add_trace(go.Box(y=df.sea_level_pressure,name="Sea Level Pressure"))

fig.add_trace(go.Box(y=df.precipitation_in_1_hr,name='Precipitation in 1 Hour'))

iplot(fig,filename='hist-features.html')
onpeak = df[df.hours > 6]

onpeak = onpeak[onpeak.hours < 23]

fig = px.histogram(onpeak, x="mean_price",marginal="violin")

iplot(fig, filename='onpeak_meanprice_histogram.html')
f,ax = plt.subplots(figsize=(10, 6))

sns.heatmap(onpeak.corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.1f',ax=ax)

plt.title('Heatmap')

plt.show()
fig = px.scatter(onpeak, x="mean_price", y="henry", color="henry", marginal_y="violin", marginal_x="box", trendline="ols", template="simple_white") #trendline = "lowess"

iplot(fig, filename = 'HenryHub-mean')
f,ax = plt.subplots(figsize=(10, 6))

sns.heatmap(df[['mean_price','henry','wind_speed_ms','ceiling_height_dimension','air_temp_c','dew_temp_c','relative_humidity','sea_level_pressure','precipitation_in_1_hr']].corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.1f',ax=ax)

plt.title('Heatmap')

plt.show()
date = []

for i in df.hour:

    i = str(i)

    d1 = datetime.datetime.strptime(i,"%Y-%m-%d %H:%M:%S")

    new_format = "%m"

    date.append(d1.strftime(new_format))

df.insert(0,'months',date,True)
dff=df[['months','hours','hour','mean_price','henry','wind_speed_ms','ceiling_height_dimension','air_temp_c','dew_temp_c','relative_humidity','sea_level_pressure','precipitation_in_1_hr']]

f,ax = plt.subplots(figsize=(10, 6))

sns.heatmap(dff[dff.months == "01"].corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.1f',ax=ax)

plt.title('Heatmap - January')

plt.show()
f,ax = plt.subplots(figsize=(10, 6))

sns.heatmap(dff[dff.months == '02'].corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.1f',ax=ax)

plt.title('Heatmap - February')

plt.show()
f,ax = plt.subplots(figsize=(10, 6))

sns.heatmap(dff[dff.months == '03'].corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.1f',ax=ax)

plt.title('Heatmap - March')

plt.show()
f,ax = plt.subplots(figsize=(10, 6))

sns.heatmap(dff[dff.months == '04'].corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.1f',ax=ax)

plt.title('Heatmap - April')

plt.show()
f,ax = plt.subplots(figsize=(10, 6))

sns.heatmap(dff[dff.months == '05'].corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.1f',ax=ax)

plt.title('Heatmap - May')

plt.show()
f,ax = plt.subplots(figsize=(10, 6))

sns.heatmap(dff[dff.months == '06'].corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.1f',ax=ax)

plt.title('Heatmap - June')

plt.show()
f,ax = plt.subplots(figsize=(10, 6))

sns.heatmap(dff[dff.months == '07'].corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.1f',ax=ax)

plt.title('Heatmap - July')

plt.show()
f,ax = plt.subplots(figsize=(10, 6))

sns.heatmap(dff[dff.months == '08'].corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.1f',ax=ax)

plt.title('Heatmap - August')

plt.show()
f,ax = plt.subplots(figsize=(10, 6))

sns.heatmap(dff[dff.months == '09'].corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.1f',ax=ax)

plt.title('Heatmap - September')

plt.show()
f,ax = plt.subplots(figsize=(10, 6))

sns.heatmap(dff[dff.months == '10'].corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.1f',ax=ax)

plt.title('Heatmap - October')

plt.show()
f,ax = plt.subplots(figsize=(10, 6))

sns.heatmap(dff[dff.months == '11'].corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.1f',ax=ax)

plt.title('Heatmap - November')

plt.show()
f,ax = plt.subplots(figsize=(10, 6))

sns.heatmap(dff[dff.months == '12'].corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.1f',ax=ax)

plt.title('Heatmap - December')

plt.show()
del dff
new_index = []

for i in range(0,len(df)):

    new_index.append(i)

df.index = new_index
measureheatmap = []

for i in range(0,len(df)):

    measureheatmap.append(float(df.mean_price[i]*df.henry[i]))

df.insert(9,'measureheatmap',measureheatmap,True)

df.info()
df.head()
print('Measuring Heat Map - Describe:')

print(df.measureheatmap.describe())
trace_1_0 = go.Scatter( x = df.hour, y=df.measureheatmap, name = "Measure Heat Map", line = dict(color = '#332CA8'), opacity = 1.0)

data = [trace_1_0]

layout = dict( title = 'Measure Heat Map = Settlement Point Price X Henry Hub NG Spot Price')

fig = dict(data=data, layout=layout)

iplot(fig, filename = 'measure-heatmap.html')
fig = px.histogram(df, x="measureheatmap",marginal="box")

iplot(fig, filename='measure_heatmap_histogram.html')
ft = ['hour','measureheatmap','henry','wind_speed_ms','ceiling_height_dimension','air_temp_c','dew_temp_c','relative_humidity','sea_level_pressure','precipitation_in_1_hr']

data = df.copy()

data = data[ft]

data.hour = pd.to_datetime(data.hour)

hour = data['hour'].dt.hour

dayofweek = data['hour'].dt.dayofweek

quarter = data['hour'].dt.quarter

month = data['hour'].dt.month

year = data['hour'].dt.year

dayofyear = data['hour'].dt.dayofyear

dayofmonth = data['hour'].dt.day

weekofyear = data['hour'].dt.weekofyear

data.insert(0,'hours',hour,True)

data.insert(1,'dayofweek',dayofweek,True)

data.insert(2,'quarter',quarter,True)

data.insert(3,'month',month,True)

data.insert(4,'year',year,True)

data.insert(5,'dayofyear',dayofyear,True)

data.insert(6,'dayofmonth',dayofmonth,True)

data.insert(7,'weekofyear',weekofyear,True)

jan = data[data.year == 2015]

jan = jan[jan.month == 1]

day = []

x = []

for i in range(0,24):

    x.append(np.mean(jan.measureheatmap[jan.hours == i]))

    day.append(i)

first = pd.DataFrame()

first.insert(0,'hours',day,True)

first.insert(1,'january',x,True)

feb = data[data.year == 2015]

feb = feb[feb.month == 2]

x = []

for i in range(0,24):

    x.append(np.mean(feb.measureheatmap[feb.hours == i]))

first.insert(2,'february',x,True)

feb = data[data.year == 2015]

feb = feb[feb.month == 3]

x = []

for i in range(0,24):

    x.append(np.mean(feb.measureheatmap[feb.hours == i]))

first.insert(3,'march',x,True)

feb = data[data.year == 2015]

feb = feb[feb.month == 4]

x = []

for i in range(0,24):

    x.append(np.mean(feb.measureheatmap[feb.hours == i]))

first.insert(4,'april',x,True)

feb = data[data.year == 2015]

feb = feb[feb.month == 5]

x = []

for i in range(0,24):

    x.append(np.mean(feb.measureheatmap[feb.hours == i]))

first.insert(5,'may',x,True)

feb = data[data.year == 2015]

feb = feb[feb.month == 6]

x = []

for i in range(0,24):

    x.append(np.mean(feb.measureheatmap[feb.hours == i]))

first.insert(6,'june',x,True)

feb = data[data.year == 2015]

feb = feb[feb.month == 7]

x = []

for i in range(0,24):

    x.append(np.mean(feb.measureheatmap[feb.hours == i]))

first.insert(7,'july',x,True)

feb = data[data.year == 2015]

feb = feb[feb.month == 8]

x = []

for i in range(0,24):

    x.append(np.mean(feb.measureheatmap[feb.hours == i]))

first.insert(8,'august',x,True)

feb = data[data.year == 2015]

feb = feb[feb.month == 9]

x = []

for i in range(0,24):

    x.append(np.mean(feb.measureheatmap[feb.hours == i]))

first.insert(9,'september',x,True)

feb = data[data.year == 2015]

feb = feb[feb.month == 10]

x = []

for i in range(0,24):

    x.append(np.mean(feb.measureheatmap[feb.hours == i]))

first.insert(10,'october',x,True)

feb = data[data.year == 2015]

feb = feb[feb.month == 11]

x = []

for i in range(0,24):

    x.append(np.mean(feb.measureheatmap[feb.hours == i]))

first.insert(11,'november',x,True)

feb = data[data.year == 2015]

feb = feb[feb.month == 12]

x = []

for i in range(0,24):

    x.append(np.mean(feb.measureheatmap[feb.hours == i]))

first.insert(12,'december',x,True)

ft = ['january','february','march','april','may','june','july','august','september','october','november','december']

plt.figure(figsize=(20,15))

plt.title('Measuring heatmap for the average hourly price with a monthly view for 2015')

ax = sns.heatmap(first[ft],linewidth=.5, annot=True, fmt="f")

plt.show()
ft = ['hour','measureheatmap','henry','wind_speed_ms','ceiling_height_dimension','air_temp_c','dew_temp_c','relative_humidity','sea_level_pressure','precipitation_in_1_hr']

data = df.copy()

data = data[ft]

data.hour = pd.to_datetime(data.hour)

hour = data['hour'].dt.hour

dayofweek = data['hour'].dt.dayofweek

quarter = data['hour'].dt.quarter

month = data['hour'].dt.month

year = data['hour'].dt.year

dayofyear = data['hour'].dt.dayofyear

dayofmonth = data['hour'].dt.day

weekofyear = data['hour'].dt.weekofyear

data.insert(0,'hours',hour,True)

data.insert(1,'dayofweek',dayofweek,True)

data.insert(2,'quarter',quarter,True)

data.insert(3,'month',month,True)

data.insert(4,'year',year,True)

data.insert(5,'dayofyear',dayofyear,True)

data.insert(6,'dayofmonth',dayofmonth,True)

data.insert(7,'weekofyear',weekofyear,True)

jan = data[data.year == 2016]

jan = jan[jan.month == 1]

day = []

x = []

for i in range(0,24):

    x.append(np.mean(jan.measureheatmap[jan.hours == i]))

    day.append(i)

first = pd.DataFrame()

first.insert(0,'hours',day,True)

first.insert(1,'january',x,True)

feb = data[data.year == 2016]

feb = feb[feb.month == 2]

x = []

for i in range(0,24):

    x.append(np.mean(feb.measureheatmap[feb.hours == i]))

first.insert(2,'february',x,True)

feb = data[data.year == 2016]

feb = feb[feb.month == 3]

x = []

for i in range(0,24):

    x.append(np.mean(feb.measureheatmap[feb.hours == i]))

first.insert(3,'march',x,True)

feb = data[data.year == 2016]

feb = feb[feb.month == 4]

x = []

for i in range(0,24):

    x.append(np.mean(feb.measureheatmap[feb.hours == i]))

first.insert(4,'april',x,True)

feb = data[data.year == 2016]

feb = feb[feb.month == 5]

x = []

for i in range(0,24):

    x.append(np.mean(feb.measureheatmap[feb.hours == i]))

first.insert(5,'may',x,True)

feb = data[data.year == 2016]

feb = feb[feb.month == 6]

x = []

for i in range(0,24):

    x.append(np.mean(feb.measureheatmap[feb.hours == i]))

first.insert(6,'june',x,True)

feb = data[data.year == 2016]

feb = feb[feb.month == 7]

x = []

for i in range(0,24):

    x.append(np.mean(feb.measureheatmap[feb.hours == i]))

first.insert(7,'july',x,True)

feb = data[data.year == 2016]

feb = feb[feb.month == 8]

x = []

for i in range(0,24):

    x.append(np.mean(feb.measureheatmap[feb.hours == i]))

first.insert(8,'august',x,True)

feb = data[data.year == 2016]

feb = feb[feb.month == 9]

x = []

for i in range(0,24):

    x.append(np.mean(feb.measureheatmap[feb.hours == i]))

first.insert(9,'september',x,True)

feb = data[data.year == 2016]

feb = feb[feb.month == 10]

x = []

for i in range(0,24):

    x.append(np.mean(feb.measureheatmap[feb.hours == i]))

first.insert(10,'october',x,True)

feb = data[data.year == 2016]

feb = feb[feb.month == 11]

x = []

for i in range(0,24):

    x.append(np.mean(feb.measureheatmap[feb.hours == i]))

first.insert(11,'november',x,True)

feb = data[data.year == 2016]

feb = feb[feb.month == 12]

x = []

for i in range(0,24):

    x.append(np.mean(feb.measureheatmap[feb.hours == i]))

first.insert(12,'december',x,True)

ft = ['january','february','march','april','may','june','july','august','september','october','november','december']

plt.figure(figsize=(20,15))

plt.title('Measuring heatmap for the average hourly price with a monthly view for 2016')

ax = sns.heatmap(first[ft],linewidth=.5, annot=True, fmt="f")

plt.show()
ft = ['hour','measureheatmap','henry','wind_speed_ms','ceiling_height_dimension','air_temp_c','dew_temp_c','relative_humidity','sea_level_pressure','precipitation_in_1_hr']

data = df.copy()

data = data[ft]

data.hour = pd.to_datetime(data.hour)

hour = data['hour'].dt.hour

dayofweek = data['hour'].dt.dayofweek

quarter = data['hour'].dt.quarter

month = data['hour'].dt.month

year = data['hour'].dt.year

dayofyear = data['hour'].dt.dayofyear

dayofmonth = data['hour'].dt.day

weekofyear = data['hour'].dt.weekofyear

data.insert(0,'hours',hour,True)

data.insert(1,'dayofweek',dayofweek,True)

data.insert(2,'quarter',quarter,True)

data.insert(3,'month',month,True)

data.insert(4,'year',year,True)

data.insert(5,'dayofyear',dayofyear,True)

data.insert(6,'dayofmonth',dayofmonth,True)

data.insert(7,'weekofyear',weekofyear,True)

jan = data[data.year == 2017]

jan = jan[jan.month == 1]

day = []

x = []

for i in range(0,24):

    x.append(np.mean(jan.measureheatmap[jan.hours == i]))

    day.append(i)

first = pd.DataFrame()

first.insert(0,'hours',day,True)

first.insert(1,'january',x,True)

feb = data[data.year == 2017]

feb = feb[feb.month == 2]

x = []

for i in range(0,24):

    x.append(np.mean(feb.measureheatmap[feb.hours == i]))

first.insert(2,'february',x,True)

feb = data[data.year == 2017]

feb = feb[feb.month == 3]

x = []

for i in range(0,24):

    x.append(np.mean(feb.measureheatmap[feb.hours == i]))

first.insert(3,'march',x,True)

feb = data[data.year == 2017]

feb = feb[feb.month == 4]

x = []

for i in range(0,24):

    x.append(np.mean(feb.measureheatmap[feb.hours == i]))

first.insert(4,'april',x,True)

feb = data[data.year == 2017]

feb = feb[feb.month == 5]

x = []

for i in range(0,24):

    x.append(np.mean(feb.measureheatmap[feb.hours == i]))

first.insert(5,'may',x,True)

feb = data[data.year == 2017]

feb = feb[feb.month == 6]

x = []

for i in range(0,24):

    x.append(np.mean(feb.measureheatmap[feb.hours == i]))

first.insert(6,'june',x,True)

feb = data[data.year == 2017]

feb = feb[feb.month == 7]

x = []

for i in range(0,24):

    x.append(np.mean(feb.measureheatmap[feb.hours == i]))

first.insert(7,'july',x,True)

feb = data[data.year == 2017]

feb = feb[feb.month == 8]

x = []

for i in range(0,24):

    x.append(np.mean(feb.measureheatmap[feb.hours == i]))

first.insert(8,'august',x,True)

feb = data[data.year == 2017]

feb = feb[feb.month == 9]

x = []

for i in range(0,24):

    x.append(np.mean(feb.measureheatmap[feb.hours == i]))

first.insert(9,'september',x,True)

feb = data[data.year == 2017]

feb = feb[feb.month == 10]

x = []

for i in range(0,24):

    x.append(np.mean(feb.measureheatmap[feb.hours == i]))

first.insert(10,'october',x,True)

feb = data[data.year == 2017]

feb = feb[feb.month == 11]

x = []

for i in range(0,24):

    x.append(np.mean(feb.measureheatmap[feb.hours == i]))

first.insert(11,'november',x,True)

feb = data[data.year == 2017]

feb = feb[feb.month == 12]

x = []

for i in range(0,24):

    x.append(np.mean(feb.measureheatmap[feb.hours == i]))

first.insert(12,'december',x,True)

ft = ['january','february','march','april','may','june','july','august','september','october','november','december']

plt.figure(figsize=(20,15))

plt.title('Measuring heatmap for the average hourly price with a monthly view for 2017')

ax = sns.heatmap(first[ft],linewidth=.5, annot=True, fmt="f")

plt.show()
ft = ['hour','measureheatmap','henry','wind_speed_ms','ceiling_height_dimension','air_temp_c','dew_temp_c','relative_humidity','sea_level_pressure','precipitation_in_1_hr']

data = df.copy()

data = data[ft]

data.hour = pd.to_datetime(data.hour)

hour = data['hour'].dt.hour

dayofweek = data['hour'].dt.dayofweek

quarter = data['hour'].dt.quarter

month = data['hour'].dt.month

year = data['hour'].dt.year

dayofyear = data['hour'].dt.dayofyear

dayofmonth = data['hour'].dt.day

weekofyear = data['hour'].dt.weekofyear

data.insert(0,'hours',hour,True)

data.insert(1,'dayofweek',dayofweek,True)

data.insert(2,'quarter',quarter,True)

data.insert(3,'month',month,True)

data.insert(4,'year',year,True)

data.insert(5,'dayofyear',dayofyear,True)

data.insert(6,'dayofmonth',dayofmonth,True)

data.insert(7,'weekofyear',weekofyear,True)

jan = data[data.year == 2018]

jan = jan[jan.month == 1]

day = []

x = []

for i in range(0,24):

    x.append(np.mean(jan.measureheatmap[jan.hours == i]))

    day.append(i)

first = pd.DataFrame()

first.insert(0,'hours',day,True)

first.insert(1,'january',x,True)

feb = data[data.year == 2018]

feb = feb[feb.month == 2]

x = []

for i in range(0,24):

    x.append(np.mean(feb.measureheatmap[feb.hours == i]))

first.insert(2,'february',x,True)

feb = data[data.year == 2018]

feb = feb[feb.month == 3]

x = []

for i in range(0,24):

    x.append(np.mean(feb.measureheatmap[feb.hours == i]))

first.insert(3,'march',x,True)

feb = data[data.year == 2018]

feb = feb[feb.month == 4]

x = []

for i in range(0,24):

    x.append(np.mean(feb.measureheatmap[feb.hours == i]))

first.insert(4,'april',x,True)

feb = data[data.year == 2018]

feb = feb[feb.month == 5]

x = []

for i in range(0,24):

    x.append(np.mean(feb.measureheatmap[feb.hours == i]))

first.insert(5,'may',x,True)

feb = data[data.year == 2018]

feb = feb[feb.month == 6]

x = []

for i in range(0,24):

    x.append(np.mean(feb.measureheatmap[feb.hours == i]))

first.insert(6,'june',x,True)

feb = data[data.year == 2018]

feb = feb[feb.month == 7]

x = []

for i in range(0,24):

    x.append(np.mean(feb.measureheatmap[feb.hours == i]))

first.insert(7,'july',x,True)

feb = data[data.year == 2018]

feb = feb[feb.month == 8]

x = []

for i in range(0,24):

    x.append(np.mean(feb.measureheatmap[feb.hours == i]))

first.insert(8,'august',x,True)

feb = data[data.year == 2018]

feb = feb[feb.month == 9]

x = []

for i in range(0,24):

    x.append(np.mean(feb.measureheatmap[feb.hours == i]))

first.insert(9,'september',x,True)

feb = data[data.year == 2018]

feb = feb[feb.month == 10]

x = []

for i in range(0,24):

    x.append(np.mean(feb.measureheatmap[feb.hours == i]))

first.insert(10,'october',x,True)

feb = data[data.year == 2018]

feb = feb[feb.month == 11]

x = []

for i in range(0,24):

    x.append(np.mean(feb.measureheatmap[feb.hours == i]))

first.insert(11,'november',x,True)

feb = data[data.year == 2018]

feb = feb[feb.month == 12]

x = []

for i in range(0,24):

    x.append(np.mean(feb.measureheatmap[feb.hours == i]))

first.insert(12,'december',x,True)

ft = ['january','february','march','april','may','june','july','august','september','october','november','december']

plt.figure(figsize=(20,15))

plt.title('Measuring heatmap for the average hourly price with a monthly view for 2018')

ax = sns.heatmap(first[ft],linewidth=.5, annot=True, fmt="f")

plt.show()
ft = ['hour','measureheatmap','henry','wind_speed_ms','ceiling_height_dimension','air_temp_c','dew_temp_c','relative_humidity','sea_level_pressure','precipitation_in_1_hr']

data = df.copy()

data = data[ft]

data.hour = pd.to_datetime(data.hour)

hour = data['hour'].dt.hour

dayofweek = data['hour'].dt.dayofweek

quarter = data['hour'].dt.quarter

month = data['hour'].dt.month

year = data['hour'].dt.year

dayofyear = data['hour'].dt.dayofyear

dayofmonth = data['hour'].dt.day

weekofyear = data['hour'].dt.weekofyear

data.insert(0,'hours',hour,True)

data.insert(1,'dayofweek',dayofweek,True)

data.insert(2,'quarter',quarter,True)

data.insert(3,'month',month,True)

data.insert(4,'year',year,True)

data.insert(5,'dayofyear',dayofyear,True)

data.insert(6,'dayofmonth',dayofmonth,True)

data.insert(7,'weekofyear',weekofyear,True)

jan = data[data.year == 2019]

jan = jan[jan.month == 1]

day = []

x = []

for i in range(0,24):

    x.append(np.mean(jan.measureheatmap[jan.hours == i]))

    day.append(i)

first = pd.DataFrame()

first.insert(0,'hours',day,True)

first.insert(1,'january',x,True)

feb = data[data.year == 2019]

feb = feb[feb.month == 2]

x = []

for i in range(0,24):

    x.append(np.mean(feb.measureheatmap[feb.hours == i]))

first.insert(2,'february',x,True)

feb = data[data.year == 2019]

feb = feb[feb.month == 3]

x = []

for i in range(0,24):

    x.append(np.mean(feb.measureheatmap[feb.hours == i]))

first.insert(3,'march',x,True)

feb = data[data.year == 2019]

feb = feb[feb.month == 4]

x = []

for i in range(0,24):

    x.append(np.mean(feb.measureheatmap[feb.hours == i]))

first.insert(4,'april',x,True)

feb = data[data.year == 2019]

feb = feb[feb.month == 5]

x = []

for i in range(0,24):

    x.append(np.mean(feb.measureheatmap[feb.hours == i]))

first.insert(5,'may',x,True)

feb = data[data.year == 2019]

feb = feb[feb.month == 6]

x = []

for i in range(0,24):

    x.append(np.mean(feb.measureheatmap[feb.hours == i]))

first.insert(6,'june',x,True)

feb = data[data.year == 2019]

feb = feb[feb.month == 7]

x = []

for i in range(0,24):

    x.append(np.mean(feb.measureheatmap[feb.hours == i]))

first.insert(7,'july',x,True)

feb = data[data.year == 2019]

feb = feb[feb.month == 8]

x = []

for i in range(0,24):

    x.append(np.mean(feb.measureheatmap[feb.hours == i]))

first.insert(8,'august',x,True)

feb = data[data.year == 2019]

feb = feb[feb.month == 9]

x = []

for i in range(0,24):

    x.append(np.mean(feb.measureheatmap[feb.hours == i]))

first.insert(9,'september',x,True)

feb = data[data.year == 2019]

feb = feb[feb.month == 10]

x = []

for i in range(0,24):

    x.append(np.mean(feb.measureheatmap[feb.hours == i]))

first.insert(10,'october',x,True)

feb = data[data.year == 2019]

feb = feb[feb.month == 11]

x = []

for i in range(0,24):

    x.append(np.mean(feb.measureheatmap[feb.hours == i]))

first.insert(11,'november',x,True)

feb = data[data.year == 2019]

feb = feb[feb.month == 12]

x = []

for i in range(0,24):

    x.append(np.mean(feb.measureheatmap[feb.hours == i]))

first.insert(12,'december',x,True)

ft = ['january','february','march','april','may','june','july','august','september','october','november','december']

plt.figure(figsize=(20,15))

plt.title('Measuring heatmap for the average hourly price with a monthly view for 2019')

ax = sns.heatmap(first[ft],linewidth=.5, annot=True, fmt="f")

plt.show()
jan = data[data.year == 2020]

jan = jan[jan.month == 1]

day = []

x = []

for i in range(0,24):

    x.append(np.mean(jan.measureheatmap[jan.hours == i]))

    day.append(i)

second = pd.DataFrame()

second.insert(0,'hours',day,True)

second.insert(1,'january',x,True)

feb = data[data.year == 2020]

feb = feb[feb.month == 2]

x = []

for i in range(0,24):

    x.append(np.mean(feb.measureheatmap[feb.hours == i]))

second.insert(2,'february',x,True)

feb = data[data.year == 2020]

feb = feb[feb.month == 3]

x = []

for i in range(0,24):

    x.append(np.mean(feb.measureheatmap[feb.hours == i]))

second.insert(3,'march',x,True)

feb = data[data.year == 2020]

feb = feb[feb.month == 4]

x = []

for i in range(0,24):

    x.append(np.mean(feb.measureheatmap[feb.hours == i]))

second.insert(4,'april',x,True)

feb = data[data.year == 2020]

feb = feb[feb.month == 5]

x = []

for i in range(0,24):

    x.append(np.mean(feb.measureheatmap[feb.hours == i]))

second.insert(5,'may',x,True)

feb = data[data.year == 2020]

feb = feb[feb.month == 6]

x = []

for i in range(0,24):

    x.append(np.mean(feb.measureheatmap[feb.hours == i]))

second.insert(6,'june',x,True)

feb = data[data.year == 2020]

feb = feb[feb.month == 7]

x = []

for i in range(0,24):

    x.append(np.mean(feb.measureheatmap[feb.hours == i]))

second.insert(7,'july',x,True)

ft = ['january','february','march','april','may','june','july']

plt.figure(figsize=(15,8))

plt.title('Measuring heatmap for the average hourly price with a monthly view for 2020')

ax = sns.heatmap(second[ft],linewidth=.5, annot=True, fmt="f")

plt.show()
onpeak = data[data.hours > 6]

onpeak = onpeak[onpeak.hours < 23]

fig = px.histogram(onpeak, x="measureheatmap",marginal="violin")

iplot(fig, filename='onpeak_measureheatmap.html')
f,ax = plt.subplots(figsize=(10, 6))

sns.heatmap(onpeak.corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.1f',ax=ax)

plt.title('Heatmap')

plt.show()