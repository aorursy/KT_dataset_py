import os
import sys
import math

import time

import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from fbprophet import Prophet

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error

import folium
from folium import plugins

import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode()

%matplotlib inline
stations = pd.read_csv('../input/stations.csv')

present = defaultdict(list)
dfs = []
for year in range(2001, 2018):
    df = pd.read_csv('../input/csvs_per_year/csvs_per_year/madrid_{}.csv'.format(year))
    for col in df:
        present[col].append(year)
    dfs.append(df)
df = pd.concat(dfs, sort=False)

df.date = pd.to_datetime(df.date)

del dfs
df.head()
stations.head()
width = len(present.keys())-2
height = len(present["date"])

p = np.zeros(shape = (height, width), dtype = np.float32)
i = 0
x_ticks = []
y_ticks = list(range(2001, 2018))
for col in present.keys():
    if col in ["date", "station"] : continue
    
    x_ticks.append(col)
    for year in present[col]:
        p[year-2001, i] = df[df.date.dt.year == year][col].notnull().mean()
    i+=1
t = pd.DataFrame(p, columns = x_ticks, index=y_ticks)

plt.figure(figsize=(20,12))
sns.heatmap(t, annot=True)
plt.title("Average of present data per year and features")
plt.show()
center = [40.41694727138085, -3.7035298347473145]

def get_dist(lon_a,lat_a, lon_b, lat_b):
    x = (lon_b - lon_a)*math.cos( (lat_a+lat_b)/2 )
    y = lat_b - lat_a
    d = 6371 * math.sqrt( x**2 + y**2 )
    return d

kept_id = []
center_lat = math.radians(center[0])
center_lon = math.radians(center[1])
for i in range(len(stations)):
    lon = math.radians(stations.loc[i, "lon"])
    lat = math.radians(stations.loc[i, "lat"])

    distance = get_dist(lon, lat, center_lon, center_lat)
    if distance < 7.5:
        kept_id.append(stations.loc[i, "id"])
df = df[df["station"].isin(kept_id)]
city = folium.Map(location=center, zoom_start=15)

folium.Marker(location=center, icon=folium.Icon(color='blue'), popup='Center').add_to(city)

folium.Circle(location=center, radius=7500,
                        popup='Considered Area', #line_color='#3186cc',
                        fill_color='#3186cc').add_to(city)

for i in range(len(stations)):
    lon = stations.loc[i, "lon"]
    lat = stations.loc[i, "lat"]
    name = stations.loc[i, "name"]
    color = "green" if stations.loc[i, "id"] in kept_id else "red"
    folium.Marker(location=[lat, lon], icon=folium.Icon(color=color), popup=name).add_to(city)
    
city
df_grouped = df.groupby(["date"]).mean()
df_grouped.drop("station", axis=1, inplace = True)
to_delete = []
for col in df_grouped:
    if df_grouped[col].isnull().mean() > 0.001:
        to_delete.append(col)
df_grouped.drop(to_delete, axis=1, inplace = True)

df_grouped = df_grouped.dropna()
df_grouped.info()
df_grouped.head()
final_df = df_grouped.reset_index()[["date", "CO"]]
final_df.columns = ["ds", "y"]
final_df.head()
df_train = final_df[final_df["ds"].dt.year < 2016]
df_test = final_df[final_df["ds"].dt.year >= 2016]
print(df_train.info())
print(df_test.info())
print(df_test.head())
print(df_test.tail())
model = Prophet(changepoint_prior_scale=2.5, daily_seasonality=True)

start = time.time()
model.fit(df_train)
print("Fitting duration : {:.3f}s".format(time.time() - start) )
future_data = df_test.drop("y", axis=1)

start = time.time()
forecast_data = model.predict(future_data)
print("Predict duration : {:.3f}s".format(time.time() - start) )

forecast_data["y"] = df_test["y"].values
forecast_data[['ds', 'y', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
py.iplot([
    go.Scatter(x=df_test['ds'], y=df_test['y'], name='y'),
    go.Scatter(x=forecast_data['ds'], y=forecast_data['yhat'], name='yhat'),
    go.Scatter(x=forecast_data['ds'], y=forecast_data['yhat_upper'], fill='tonexty', mode='none', name='upper'),
    go.Scatter(x=forecast_data['ds'], y=forecast_data['yhat_lower'], fill='tonexty', mode='none', name='lower'),
    go.Scatter(x=forecast_data['ds'], y=forecast_data['trend'], name='Trend')
])
mse = []
for i in range(0, len(forecast_data), 48):
    mse.append(mean_squared_error(
                    forecast_data.loc[i:i+48, "y"],
                    forecast_data.loc[i:i+48, "yhat"]
                ))

plt.figure(figsize=(20,12))
plt.plot(mse) # mse per day during 2 years
plt.title("Evolution of MSE during year 2016 - 2017")
plt.show()
model.plot_components(forecast_data)
plt.show()
from sklearn.preprocessing import StandardScaler
df_train2 = df_grouped[ df_grouped.index.year < 2016 ]
df_test2 = df_grouped[ df_grouped.index.year >= 2016 ]
scaler = StandardScaler()
X_train = scaler.fit_transform(df_train2)
X_test = scaler.transform(df_test2)

df_train2["sum"] = X_train.sum(axis=1)
df_test2["sum"] = X_test.sum(axis=1)
df_train2.head()
final_df_train = df_train2.reset_index()[["date", "sum"]]
final_df_train.columns = ["ds", "y"]

final_df_test = df_test2.reset_index()[["date", "sum"]]
final_df_test.columns = ["ds", "y"]
model = Prophet(changepoint_prior_scale=2.5, daily_seasonality=True)

start = time.time()
model.fit(final_df_train)
print("Fitting duration : {:.3f}s".format(time.time() - start) )
future_data = final_df_test.drop("y", axis=1)

start = time.time()
forecast_data = model.predict(future_data)
print("Predict duration : {:.3f}s".format(time.time() - start) )

forecast_data["y"] = final_df_test["y"].values
forecast_data[['ds', 'y', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
mse = []
for i in range(0, len(forecast_data), 48):
    mse.append(mean_squared_error(
                    forecast_data.loc[i:i+48, "y"],
                    forecast_data.loc[i:i+48, "yhat"]
                ))

plt.figure(figsize=(20,12))
plt.plot(mse) # mse per day during 2 years
plt.title("Evolution of MSE during year 2016 - 2017")
plt.show()
py.iplot([
    go.Scatter(x=df_test['ds'], y=final_df_test['y'], name='y'),
    go.Scatter(x=forecast_data['ds'], y=forecast_data['yhat'], name='yhat'),
    go.Scatter(x=forecast_data['ds'], y=forecast_data['yhat_upper'], fill='tonexty', mode='none', name='upper'),
    go.Scatter(x=forecast_data['ds'], y=forecast_data['yhat_lower'], fill='tonexty', mode='none', name='lower'),
    go.Scatter(x=forecast_data['ds'], y=forecast_data['trend'], name='Trend')
])
model.plot_components(forecast_data)
plt.show()
