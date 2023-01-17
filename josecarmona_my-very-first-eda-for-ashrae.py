import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.graph_objects as go

from plotly.subplots import make_subplots

import datetime

import os



%matplotlib inline
def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
building_metadata = pd.read_csv("../input/ashrae-energy-prediction/building_metadata.csv")

train = pd.read_csv("../input/ashrae-energy-prediction/train.csv",parse_dates=['timestamp'])

test = pd.read_csv("../input/ashrae-energy-prediction/test.csv",parse_dates=['timestamp'])

weather_train = pd.read_csv("../input/ashrae-energy-prediction/weather_train.csv",parse_dates=['timestamp'])

weather_test = pd.read_csv("../input/ashrae-energy-prediction/weather_test.csv",parse_dates=['timestamp'])
building_metadata = reduce_mem_usage(building_metadata)

train = reduce_mem_usage(train)

test = reduce_mem_usage(test)

weather_train = reduce_mem_usage(weather_train)

weather_test = reduce_mem_usage(weather_test)
building_metadata.groupby(['primary_use','site_id']).size().unstack().fillna(0).style.background_gradient(axis=None)
by_site_id = building_metadata.groupby('site_id')



ind = ['total','square_feet', 'year_built', 'floor_count']

df = pd.DataFrame([by_site_id.building_id.count(),by_site_id.square_feet.count(),by_site_id.year_built.count(),by_site_id.floor_count.count()], index=ind)



fig, axes = plt.subplots(1, 1, figsize=(14, 6), dpi=100)



sns.heatmap(df, cmap='Blues', linewidths=1.5, annot=True, fmt="d", ax=axes)

      
by_site_id = weather_train.groupby('site_id')



ind = ['total',

       'air_temperature', 

       'dew_temperature', 

       'wind_speed',

       'wind_direction', 

       'sea_level_pressure', 

       'precip_depth_1_hr',

       'cloud_coverage'

      ]

df = pd.DataFrame([by_site_id.timestamp.count(),

                   by_site_id.air_temperature.count(),

                   by_site_id.dew_temperature.count(),

                   by_site_id.wind_speed.count(),

                   by_site_id.wind_direction.count(),

                   by_site_id.sea_level_pressure.count(),

                   by_site_id.precip_depth_1_hr.count(),

                   by_site_id.cloud_coverage.count()

                  ],

                  index=ind)



fig, axes = plt.subplots(1, 1, figsize=(14, 6), dpi=100)



sns.heatmap(df, cmap='Blues', linewidths=1.5, annot=True, fmt="d", ax=axes)



by_site_id = weather_test.groupby('site_id')



ind = ['total',

       'air_temperature', 

       'dew_temperature', 

       'wind_speed',

       'wind_direction', 

       'sea_level_pressure', 

       'precip_depth_1_hr',

       'cloud_coverage'

      ]

df = pd.DataFrame([by_site_id.timestamp.count(),

                   by_site_id.air_temperature.count(),

                   by_site_id.dew_temperature.count(),

                   by_site_id.wind_speed.count(),

                   by_site_id.wind_direction.count(),

                   by_site_id.sea_level_pressure.count(),

                   by_site_id.precip_depth_1_hr.count(),

                   by_site_id.cloud_coverage.count()

                  ],

                  index=ind)



fig, axes = plt.subplots(1, 1, figsize=(14, 6), dpi=100)



sns.heatmap(df, cmap='Blues', linewidths=1.5, annot=True, fmt="d", ax=axes)





w = pd.concat([weather_train,weather_test])[["site_id","timestamp","air_temperature","dew_temperature"]]
def plot_weather_site(weather, site_id):

    

    lw = weather.query(f"site_id == {site_id}").copy()

    lw["day"] = lw["timestamp"].dt.ceil("1d")

    lw["hour"] = lw["timestamp"].dt.hour

    lw['air_temperature'] = lw['air_temperature'].astype(np.float32)

    lw['dew_temperature'] = lw['dew_temperature'].astype(np.float32)



    p1 = lw[['day','hour','air_temperature']].pivot_table(values='air_temperature', index=['day'], columns=['hour']).copy()

    p2 = lw[['day','hour','dew_temperature']].pivot_table(values='dew_temperature', index=['day'], columns=['hour']).copy()





    fig = go.Figure(data=[

        go.Surface(z=p1, colorscale='YlOrRd', opacity=0.9, showscale=False),

        go.Surface(z=p2, colorscale='RdBu', opacity=0.2, showscale=False)

    ])



    fig.update_layout(title_text=f'site_id {site_id}',

                      height=1000,

                      width=1000)

    fig.show()
plot_weather_site(w,1)
plot_weather_site(w,13)
t = train.merge(building_metadata, on='building_id', how='left')
def plot_building_meter_reading(train, building_id):

    fig = make_subplots(rows=2, 

                        cols=2,

                        specs=[[{'type': 'surface'}, {'type': 'surface'}],[{'type': 'surface'}, {'type': 'surface'}]],

                        subplot_titles=("meter = 0", "meter = 2", "meter = 1", "meter = 3"))

    t = train.query(f'building_id == {building_id}').copy()

    t["day"] = t["timestamp"].dt.ceil("1d")

    t["hour"] = t["timestamp"].dt.hour

    for m in range(4):

        p = t.query(f'meter == {m}')[['day','hour','meter_reading']].pivot_table(values='meter_reading', index=['day'], columns=['hour']).copy()

        fig.add_trace( go.Surface(z=p, colorscale='YlOrRd', showscale=False), row=1+m%2, col=1+m//2)

    

    fig.update_layout(title_text=f'Building {building_id}',

                      height=1000,

                      width=1000)

    fig.show()
plot_building_meter_reading(t,801)
plot_building_meter_reading(t,1230)
w = pd.concat([weather_train,weather_test])

w["hour"] = w["timestamp"].dt.hour

w["dmy"] = w["timestamp"].dt.floor('D')

w = w.loc[w.groupby(["site_id", "dmy"])["air_temperature"].idxmax()] 

w = w.groupby(["site_id"])

w.hour.apply(lambda x: x.mode())



# Hour of T max by site_id

htmax = [19,14,0,19,0,12,20,0,19,21,0,0,14,0,20,20]
# Hour of T max by site_id

w = pd.concat([weather_train,weather_test])

w["hour"] = w["timestamp"].dt.hour



htmax = [19,14,0,19,0,12,20,0,19,21,0,0,14,0,20,20]

w["htmax"] = w.site_id.apply (lambda x: htmax[x])

w["w_htmax"] = w.hour.sub(w.htmax).abs()

w["w_htmax"] = w.w_htmax.apply(lambda x: (12 - x) if x<12 else (x%12))

del w["htmax"], w["hour"]

w.head(25)