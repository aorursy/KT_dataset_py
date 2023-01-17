# Basic library

import pandas as pd

import numpy as np

from datetime import datetime as dt

from pathlib import Path



# Graph drawing

import matplotlib

import matplotlib.pyplot as plt



plt.rcParams["patch.force_edgecolor"] = True

#rc('text', usetex=True)

from IPython.display import display # Allows the use of display() for DataFrames

import seaborn as sns

sns.set(style="whitegrid", palette="muted", color_codes=True)

sns.set_style("whitegrid", {'grid.linestyle': '--'})

red = sns.xkcd_rgb["light red"]

green = sns.xkcd_rgb["medium green"]

blue = sns.xkcd_rgb["denim blue"]



%matplotlib inline

#%config InlineBackend.figure_format='retina'
train = pd.read_csv("../input/ashrae-energy-prediction/train.csv", dtype={"meter":"category"}, parse_dates=True)

test = pd.read_csv("../input/ashrae-energy-prediction/test.csv", dtype={"meter":"category"}, parse_dates=True)



weather_train = pd.read_csv("../input/ashrae-energy-prediction/weather_train.csv", parse_dates=True)

weather_test = pd.read_csv("../input/ashrae-energy-prediction/weather_test.csv", parse_dates=True)



building = pd.read_csv("../input/ashrae-energy-prediction/building_metadata.csv", dtype={"primary_use":"category"})
meter_type = {i: v for i, v in enumerate(["electricity", "chilledwater", "steam", " hotwater"])}

meter_type
for i in range(20):

    print(i, train[train.building_id==i].timestamp.shape)

    build_s = building[building.building_id==i]

    display(build_s)

    df_s = train[train.building_id==i].copy()

    df_s.timestamp = pd.to_datetime(df_s.timestamp)

    site_id = build_s.iloc[0].site_id

    weather_train_s = weather_train[weather_train.site_id==site_id]

    meters = np.sort(df_s.meter.unique())

    for m in meters:

        df_s_m = df_s[df_s.meter==m].set_index("timestamp")

        plt.figure(figsize=(25, 5))

        ax=plt.subplot(111)

        df_s_m.meter_reading.plot(ax=ax, color="r")

        plt.title(f"[TARGET] building_id: {i}, {build_s.iloc[0].primary_use}, meter:{m} {meter_type[int(m)]}, site_id: {site_id}")

        plt.xticks(rotation=60)

        plt.xticks(df_s_m.index[::24*30], [str(d).split(" ")[0] for d in df_s_m.index[::24*30]], rotation=60)

        plt.xlim(df_s_m.index[0],df_s_m.index[-1])

        plt.show()

        

        weather_train_s.set_index("timestamp", inplace=True)

        for c in ['air_temperature', 'cloud_coverage','dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure','wind_direction', 'wind_speed']:

            plt.figure(figsize=(25, 2))

            ax=plt.subplot(111)

            plt.plot(weather_train_s.index, weather_train_s[c])

            

            plt.title(f"{c}")

            plt.xticks(weather_train_s.index[::24*30], [d.split(" ")[0] for d in weather_train_s.index[::24*30]], rotation=60)

            plt.xlim(weather_train_s.index[0],weather_train_s.index[-1])

            

            plt.show()

        

    print("="*100)