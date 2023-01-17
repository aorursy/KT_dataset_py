import pandas as pd

import numpy as np

from datetime import datetime, timezone

import folium

from folium import plugins

from tqdm import tqdm

import seaborn as sns



tqdm.pandas()

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
sns.set(rc={"figure.figsize": (16, 6)})

ax = sns.barplot(x=call_type_count.index, y=call_type_count.values)
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

ax = sns.barplot(

    x=origin_stand_count.index,

    y=origin_stand_count.values,

    order=origin_stand_count.index,

)
taxi_id_cat = taxi.TAXI_ID.astype("category")

taxi_id_cat.describe()
taxi_id_count = taxi_id_cat.value_counts(sort=True)

ax = sns.violinplot(y=taxi_id_count.values, cut=0)
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

trip_distance_cleaned.rename("Trip distance", inplace=True)

ax = sns.violinplot(y=trip_distance_cleaned, cut=0)
trip_time_cleaned = taxi.TRIP_TIME[

    (taxi.TRIP_TIME < taxi.TRIP_TIME.quantile(0.99))

]



trip_time_cleaned.rename("Trip time", inplace=True)

ax = sns.violinplot(y=trip_time_cleaned, cut=0)
average_speed_cleaned = taxi.AVERAGE_SPEED[

    (taxi.AVERAGE_SPEED < taxi.AVERAGE_SPEED.quantile(0.99))

]



average_speed_cleaned.rename("Average speed", inplace=True)

ax = sns.violinplot(y=average_speed_cleaned, cut=0)
top_speed_cleaned = taxi.TOP_SPEED[

    (taxi.TOP_SPEED < taxi.TOP_SPEED.quantile(0.99))

]



top_speed_cleaned.rename("Top speed", inplace=True)

ax = sns.violinplot(y=top_speed_cleaned, cut=0)
top_speed_cleaned2 = taxi.TOP_SPEED[(taxi.TOP_SPEED < 120)]



top_speed_cleaned2.rename("Top speed", inplace=True)

ax = sns.violinplot(y=top_speed_cleaned2, cut=0)
trip_distance_time = taxi[["TRIP_DISTANCE", "TRIP_TIME"]]

trip_distance_time = trip_distance_time[trip_distance_time.TRIP_DISTANCE < 10]

trip_distance_time = trip_distance_time[trip_distance_time.TRIP_TIME < 20]

trip_distance_time.TRIP_DISTANCE.rename("Trip distance [km]", inplace=True)

trip_distance_time.TRIP_TIME.rename("Top Trip time [min]", inplace=True)

ax = sns.jointplot(

    x="TRIP_DISTANCE", y="TRIP_TIME", data=trip_distance_time, kind="kde"

)
taxi_start = taxi.TRIP_START.progress_apply(lambda x: eval(x)[::-1])
trip_start_map = folium.Map(location=[41.1579605, -8.629241], zoom_start=12)

plugins.HeatMap(taxi_start, radius=10).add_to(trip_start_map)

trip_start_map