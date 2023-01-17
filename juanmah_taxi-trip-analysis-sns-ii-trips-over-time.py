import pandas as pd

import numpy as np

from datetime import datetime, timezone

import folium

from folium import plugins

from tqdm import tqdm

import seaborn as sns

import matplotlib.pyplot as plt



tqdm.pandas()

# %load_ext nb_black
taxi = pd.read_csv(

    "../input/taxi-trajectory-data-extended/train_extended.csv.zip",

    sep=",",

    compression="zip",

    low_memory=False,

)
sns.set(rc={"figure.figsize": (16, 6)})

data = taxi.pivot_table(

    index="HOUR", columns="WEEKDAY", values="TRIP_ID", aggfunc="count"

)

data.columns = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]



ax = sns.heatmap(data, cmap="coolwarm")
def draw_heatmap(*args, **kwargs):

    data = kwargs.pop("data")

    d = data.pivot_table(

        index="HOUR", columns="WEEKDAY", values="TRIP_ID", aggfunc="count"

    )

    d.columns = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    sns.heatmap(d, **kwargs)





data = taxi[["TRIP_ID", "CALL_TYPE", "WEEKDAY", "HOUR"]].copy()

data["CALL_TYPE"] = data.CALL_TYPE.astype("category")

data.CALL_TYPE.cat.rename_categories(

    {"A": "CENTRAL", "B": "STAND", "C": "OTHER"}, inplace=True

)



g = sns.FacetGrid(data, col="CALL_TYPE", height=6, aspect=1)

g = g.map_dataframe(draw_heatmap, "WEEKDAY", "HOUR", cmap="coolwarm")
# Drop last day (30.06.14) as it is overlapping (week 27 / weekday 0)

# with the first day previous year (01.07.13) and trips of both days are aggregated

taxi = taxi[(taxi.YEAR != 2014) | (taxi.MONTH != 6) | (taxi.DAY != 30)]



taxi["WEEK_NUMBER"] = taxi.TIMESTAMP.apply(

    lambda x: datetime.fromtimestamp(x).isocalendar()[1]

)



data = taxi.pivot_table(

    index="WEEKDAY", columns="WEEK_NUMBER", values="TRIP_ID", aggfunc="count"

)



data.set_index(

    pd.Series(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]), inplace=True

)



# Reverse weekdays

data = data.iloc[::-1]



sns.set(rc={"figure.figsize": (16, 2)})

ax = sns.heatmap(data, cmap="coolwarm")
sns.set(rc={"figure.figsize": (16, 6)})



# Drop extreme values as the could spoil averages

taxi = taxi[(taxi.TRIP_DISTANCE < taxi.TRIP_DISTANCE.quantile(0.99))]



distance = taxi.pivot_table(

    index="HOUR", columns="WEEKDAY", values=["TRIP_DISTANCE"], aggfunc=np.mean

)

distance.columns = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]



# Drop extreme values as the could spoil averages

taxi = taxi[(taxi.TRIP_TIME < taxi.TRIP_TIME.quantile(0.99))]



time = taxi.pivot_table(

    index="HOUR", columns="WEEKDAY", values=["TRIP_TIME"], aggfunc=np.mean

)

time.columns = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]



fig, axes = plt.subplots(1, 2)



sns.heatmap(distance, cmap="coolwarm", ax=axes[0])

sns.heatmap(time, cmap="coolwarm", ax=axes[1])
def draw_heatmap(*args, **kwargs):

    data = kwargs.pop("data")

    d = data.pivot_table(

        index=args[0], columns=args[1], values=args[2], aggfunc=np.mean

    )

    d.columns = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    sns.heatmap(d, **kwargs)

#     print(d.values.min())

#     print(d.values.max())





data = taxi[

    ["CALL_TYPE", "WEEKDAY", "HOUR", "TRIP_DISTANCE", "TRIP_TIME"]

].copy()

data["CALL_TYPE"] = data.CALL_TYPE.astype("category")

data.CALL_TYPE.cat.rename_categories(

    {"A": "CENTRAL", "B": "STAND", "C": "OTHER"}, inplace=True

)



g = sns.FacetGrid(data, col="CALL_TYPE", height=6, aspect=1)

g = g.map_dataframe(

    draw_heatmap,

    "HOUR",

    "WEEKDAY",

    "TRIP_DISTANCE",

    vmin=5.2,

    vmax=12.3,

    cmap="coolwarm",

)
g = sns.FacetGrid(data, col="CALL_TYPE", height=6, aspect=1)

g = g.map_dataframe(

    draw_heatmap,

    "HOUR",

    "WEEKDAY",

    "TRIP_TIME",

    vmin=8.6,

    vmax=15.6,

    cmap="coolwarm",

)
## Trips distance throught the year
# Drop last day (30.06.14) as it is overlapping (week 27 / weekday 0)

# with the first day previous year (01.07.13) and trips of both days are aggregated

taxi = taxi[(taxi.YEAR != 2014) | (taxi.MONTH != 6) | (taxi.DAY != 30)]



taxi["WEEK_NUMBER"] = taxi.TIMESTAMP.apply(

    lambda x: datetime.fromtimestamp(x).isocalendar()[1]

)



data = taxi.pivot_table(

    index="WEEKDAY",

    columns="WEEK_NUMBER",

    values="TRIP_DISTANCE",

    aggfunc=np.mean,

)



data.set_index(

    pd.Series(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]), inplace=True

)



# Reverse weekdays

data = data.iloc[::-1]



sns.set(rc={"figure.figsize": (16, 2)})

ax = sns.heatmap(data, cmap="coolwarm")
# Drop last day (30.06.14) as it is overlapping (week 27 / weekday 0)

# with the first day previous year (01.07.13) and trips of both days are aggregated

taxi = taxi[(taxi.YEAR != 2014) | (taxi.MONTH != 6) | (taxi.DAY != 30)]



taxi["WEEK_NUMBER"] = taxi.TIMESTAMP.apply(

    lambda x: datetime.fromtimestamp(x).isocalendar()[1]

)



data = taxi.pivot_table(

    index="WEEKDAY", columns="WEEK_NUMBER", values="TRIP_TIME", aggfunc=np.mean

)



data.set_index(

    pd.Series(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]), inplace=True

)



# Reverse weekdays

data = data.iloc[::-1]



sns.set(rc={"figure.figsize": (16, 2)})

ax = sns.heatmap(data, cmap="coolwarm")