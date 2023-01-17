import pandas as pd

import numpy as np

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot

from datetime import datetime, timezone

import folium

from folium import plugins

from tqdm import tqdm



tqdm.pandas()

init_notebook_mode(connected=True)

# %load_ext nb_black
taxi = pd.read_csv(

    "../input/taxi-trajectory-data-extended/train_extended.csv.zip",

    sep=",",

    compression="zip",

    low_memory=False,

)
taxi.head()
taxi.info()
taxi.CALL_TYPE.describe()
call_type_count = taxi.CALL_TYPE.value_counts(sort=False).sort_index()

call_type_count.index = ["CENTRAL", "STAND", "OTHER"]

print(call_type_count)
data = [go.Bar(x=call_type_count.index, y=call_type_count.values)]



iplot(data)
taxi.ORIGIN_CALL = (

    taxi.ORIGIN_CALL.fillna(-1)

    .astype("int64")

    .astype(str)

    .replace("-1", np.nan)

)

origin_call_cat = taxi.ORIGIN_CALL.astype("category")

origin_call_cat.describe()
origin_call_count = origin_call_cat.value_counts()

pd.cut(origin_call_count, bins=[0, 1, 2, 3, 4, 6, 10, 100, 10000]).value_counts(

    sort=False

)
taxi.ORIGIN_STAND = (

    taxi.ORIGIN_STAND.fillna(-1)

    .astype("int64")

    .astype(str)

    .replace("-1", np.nan)

)

origin_stand_cat = taxi.ORIGIN_STAND.astype("category")

origin_stand_cat.describe()
origin_stand_count = origin_stand_cat.value_counts(sort=True)

data = [go.Bar(x=origin_stand_count.index, y=origin_stand_count.values)]

layout = go.Layout(xaxis=dict(type="category"))

fig = go.Figure(data=data, layout=layout)

iplot(fig)
taxi_id_cat = taxi.TAXI_ID.astype("category")

taxi_id_cat.describe()
taxi_id_count = taxi_id_cat.value_counts(sort=True)

data = [go.Violin(y=taxi_id_count.values, name="Taxi IDs")]

layout = go.Layout(yaxis=dict(rangemode="nonnegative"))

fig = go.Figure(data=data, layout=layout)

iplot(fig)
taxi.TIMESTAMP.count()
datetime.fromtimestamp(taxi.TIMESTAMP.min(), timezone.utc).strftime(

    "%Y-%m-%d %H:%M:%S"

)
datetime.fromtimestamp(taxi.TIMESTAMP.max(), timezone.utc).strftime(

    "%Y-%m-%d %H:%M:%S"

)
taxi.DAY_TYPE.describe()
taxi.MISSING_DATA.describe()
taxi.POLYLINE.describe()
trip_distance_cleaned = taxi.TRIP_DISTANCE[

    (taxi.TRIP_DISTANCE < taxi.TRIP_DISTANCE.quantile(0.99))

]



data = [go.Violin(y=trip_distance_cleaned, name="", points=False)]

layout = go.Layout(yaxis=dict(rangemode="nonnegative"))

fig = go.Figure(data=data, layout=layout)

iplot(fig)
trip_time_cleaned = taxi.TRIP_TIME[

    (taxi.TRIP_TIME < taxi.TRIP_TIME.quantile(0.99))

]



data = [go.Violin(y=trip_time_cleaned, name="", points=False)]

layout = go.Layout(yaxis=dict(rangemode="nonnegative"))

fig = go.Figure(data=data, layout=layout)

iplot(fig)
average_speed_cleaned = taxi.AVERAGE_SPEED[

    (taxi.AVERAGE_SPEED < taxi.AVERAGE_SPEED.quantile(0.99))

]



data = [go.Violin(y=average_speed_cleaned, name="", points=False)]

layout = go.Layout(yaxis=dict(rangemode="nonnegative"))

fig = go.Figure(data=data, layout=layout)

iplot(fig)
top_speed_cleaned = taxi.TOP_SPEED[

    (taxi.TOP_SPEED < taxi.TOP_SPEED.quantile(0.99))

]



data = [go.Violin(y=top_speed_cleaned, name="", points=False)]

layout = go.Layout(yaxis=dict(rangemode="nonnegative"))

fig = go.Figure(data=data, layout=layout)

iplot(fig)
top_speed_cleaned2 = taxi.TOP_SPEED[(taxi.TOP_SPEED < 120)]



data = [go.Violin(y=top_speed_cleaned2, name="", points=False)]

layout = go.Layout(yaxis=dict(rangemode="nonnegative"))

fig = go.Figure(data=data, layout=layout)

iplot(fig)
data = [go.Histogram2dContour(x=taxi.TRIP_DISTANCE, y=taxi.TRIP_TIME)]

layout = go.Layout(

    xaxis=dict(range=[0, 10], title="Trip distance [km]"),

    yaxis=dict(range=[0, 20], title="Trip time [min]"),

)

fig = go.Figure(data=data, layout=layout)



iplot(fig)
taxi_start = taxi.TRIP_START.progress_apply(lambda x: eval(x)[::-1])
trip_start_map = folium.Map(location=[41.1579605, -8.629241], zoom_start=12)

plugins.HeatMap(taxi_start, radius=10).add_to(trip_start_map)

trip_start_map