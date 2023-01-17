import gc

import os

import random

import csv

import sys



import lightgbm as lgb

import numpy as np

import pandas as pd

import seaborn as sns



from matplotlib import pyplot as plt



from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import KFold, train_test_split

from sklearn.ensemble import BaggingRegressor



from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import IterativeImputer



import time

from datetime import datetime, timedelta

import pickle
def preparationDonnees(data, building, weather, encoder=None, imputer=None, seuils=None):

    t0 = time.time()

    building = reduce_mem_usage(pd.read_csv(building))

    #Traitement des outliers

    seuil_min, seuil_max = seuil_min_max(building["square_feet"], 2)

    building["square_feet"] = replaceOutliers(building["square_feet"], seuil_min, seuil_max)

    building.drop(["year_built", "floor_count"], axis=1, inplace=True)

    #arrondi de la colonne square_feet

    building["square_feet"] = building["square_feet"].apply(lambda x: int(x / 10) * 10)

    building["square_feet"] = np.log1p(building["square_feet"])

    

    col_weather = ["air_temperature", "sea_level_pressure", "wind_direction", "wind_speed"]

    weather = reduce_mem_usage(pd.read_csv(weather))

    weather.drop(["precip_depth_1_hr", "cloud_coverage", "dew_temperature"], axis=1, inplace=True)

    weather["timestamp"] = weather["timestamp"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))

    

    if encoder is None and imputer is None:

        encoder = LabelEncoder()

        encoder.fit(building["primary_use"])

        building["primary_use"] = encoder.transform(building["primary_use"])

        imputer = IterativeImputer()

        imputer.fit(weather[col_weather])

        seuils={}

        seuil_min_air, seuil_max_air = seuil_min_max(weather["air_temperature"])

        seuil_min_pressure, seuil_max_pressure = seuil_min_max(weather["sea_level_pressure"])

        seuil_min_wind, seuil_max_wind = seuil_min_max(weather["wind_speed"])

        seuils["seuil_min_air"] = seuil_min_air

        seuils["seuil_max_air"] = seuil_max_air

        seuils["seuil_min_pressure"] = seuil_min_pressure

        seuils["seuil_max_pressure"] = seuil_max_pressure

        seuils["seuil_min_wind"] = seuil_min_wind

        seuils["seuil_max_wind"] = seuil_max_wind      

    else:

        building["primary_use"] = encoder.transform(building["primary_use"])

        seuil_min_air = seuils["seuil_min_air"]

        seuil_max_air = seuils["seuil_max_air"]

        seuil_min_pressure = seuils["seuil_min_pressure"]

        seuil_max_pressure = seuils["seuil_max_pressure"]

        seuil_min_wind = seuils["seuil_min_wind"]

        seuil_max_wind = seuils["seuil_max_wind"]

        

    #### DATA ####

    data = reduce_mem_usage(pd.read_csv(data))

    data["timestamp"] = data["timestamp"].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))

    ### FUSION TRAIN ET BUILDING --> TRAIN ####

    print("Fusion de 'data' et 'building'")

    data = pd.merge(data, building, on="building_id", how="left")

    #### FUSION TRAIN ET WEATHER --> TRAIN ####

    print("Fusion de data et weather")

    data = pd.merge(data, weather, on=["timestamp", "site_id"], how="left")

    

    filtre = np.logical_or(data["air_temperature"].isnull(), data["sea_level_pressure"].isnull())

    filtre2 = np.logical_or(data["wind_direction"].isnull(), data["wind_speed"].isnull())

    filtre = np.logical_or(filtre, filtre2)

    

    d2 = data[filtre][col_weather]

    d = imputer.transform(d2)

    d2 = pd.DataFrame(data=d, index=d2.index, columns=col_weather)

    data.update(d2)

    

    d2 = None

    d = None

    filtre = None

    filtre2 = None

    weather = None

    nuilding = None

    

    #Ecrasement des outliers

    print("Remplacements des outliers")

    data["air_temperature"] = replaceOutliers(data["air_temperature"], seuil_min_air, seuil_max_air)

    data["sea_level_pressure"] = replaceOutliers(data["sea_level_pressure"], seuil_min_pressure, seuil_max_pressure)

    data["wind_speed"] = replaceOutliers(data["wind_speed"], seuil_min_wind, seuil_max_wind)

    #arrondi de la temperature

    data["air_temperature"] = np.round(data["air_temperature"], decimals=1)

    #Arrondir la direction du vent par dizaine

    data["wind_direction"] = data["wind_direction"].apply(lambda x: int(round(x / 10, 0) * 10))

    #remplacer 360 par 0

    data["wind_direction"] = data["wind_direction"].apply(lambda x: 0 if x == 360 else x)



    data = reduce_mem_usage(data)

    

    if "meter_reading" in list(data.columns):

        data = data[['timestamp', 'site_id', 'building_id', 'meter', 'meter_reading',

                 'primary_use', 'square_feet', 'air_temperature',

                 'sea_level_pressure', 'wind_direction', 'wind_speed']]

    else:

        data = data[['timestamp', 'site_id', 'building_id', 'meter',

                 'primary_use', 'square_feet', 'air_temperature',

                 'sea_level_pressure', 'wind_direction', 'wind_speed']]

    

    

    #Création des colonnes MONTH, DAY et HOUR

    print("Création de la colonne Month")

    data["MONTH"] = data["timestamp"].apply(lambda x: x.month)

    print("Création de la colonne Day")

    data["DAY"] = data["timestamp"].apply(lambda x: x.day)

    print("Création de Hour")

    data["HOUR"] = data["timestamp"].apply(lambda x: x.hour)

    print("Création du jour de la semaine")

    data["DAYOFWEEK"] = data["timestamp"].apply(lambda x: x.dayofweek)

    

    #Suppression de la colonne TIMESTAMP

    #data.drop("timestamp", axis=1, inplace=True)

    

    #Suppression de la colonne building_id

    #data.drop("building_id", axis=1, inplace=True)

    

    try:

        data.drop("row_id", axis=1, inplace=True)

    except:

        pass

    

    #Suppression des lignes avec 0 en meter_reading

    #if "meter_reading" in data.columns:

        #print("Nombre de lignes avec meter_reading à 0: {}".format(len(data[data["meter_reading"] <= 0])))

        #data = data[data["meter_reading"] > 0]

    

    print("Réduction de la place mémoire")

    data = reduce_mem_usage(data)

    

    #suppression des données en double (data leakage)

    if "meter_reading" in data.columns:

        target = data["meter_reading"]

        data.drop("meter_reading", axis=1, inplace=True)

        data.drop_duplicates(inplace=True)

        filtre = data.duplicated()

        print("Nombre de ligne en double dans le train: {}".format(filtre.sum()))

        data = data[np.logical_not(filtre)]

        target = target[data.index]

        target = np.log1p(target)

    else:

        target = None

        

    print("Durée exécution: {:0.2f} secondes".format(time.time() - t0))

        

    return data, target, encoder, imputer, seuils
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
def seuil_min_max(data_series, coef=1.5):

    seuil_max = data_series.quantile(0.75) + (data_series.quantile(0.75) - data_series.quantile(0.25)) * coef

    seuil_min = data_series.quantile(0.25) - (data_series.quantile(0.75) - data_series.quantile(0.25)) * coef

    print("Seuil min {}: {}".format(data_series.name, seuil_min))

    print("Seuil max {}: {}".format(data_series.name, seuil_max))

    #Proportion des outliers

    filtre = np.logical_or(data_series >= seuil_max, data_series <= seuil_min)

    print("Proportion outliers: {:0.2f}%".format(len(data_series[filtre]) / len(data_series) * 100))

    return seuil_min, seuil_max





def replaceOutliers(data_series, seuil_min, seuil_max):

    #Remplace des extremes par les seuils

    data_series = data_series.apply(lambda x: seuil_min if x <= seuil_min else x)

    data_series = data_series.apply(lambda x: seuil_max if x >= seuil_max else x)

    return data_series
def prepare_data(X, building_data, weather_data, test=False):

    """

    Preparing final dataset with all features.

    """

    

    X = X.merge(building_data, on="building_id", how="left")

    X = X.merge(weather_data, on=["site_id", "timestamp"], how="left")

    

    X.timestamp = pd.to_datetime(X.timestamp, format="%Y-%m-%d %H:%M:%S")

    X.square_feet = np.log1p(X.square_feet)

    

    if not test:

        X.sort_values("timestamp", inplace=True)

        X.reset_index(drop=True, inplace=True)

    

    gc.collect()

    

    holidays = ["2016-01-01", "2016-01-18", "2016-02-15", "2016-05-30", "2016-07-04",

                "2016-09-05", "2016-10-10", "2016-11-11", "2016-11-24", "2016-12-26",

                "2017-01-01", "2017-01-16", "2017-02-20", "2017-05-29", "2017-07-04",

                "2017-09-04", "2017-10-09", "2017-11-10", "2017-11-23", "2017-12-25",

                "2018-01-01", "2018-01-15", "2018-02-19", "2018-05-28", "2018-07-04",

                "2018-09-03", "2018-10-08", "2018-11-12", "2018-11-22", "2018-12-25",

                "2019-01-01"]

    

    X["month"] = X.timestamp.dt.month #Rajout

    X["day"] = X.timestamp.dt.day #Rajout

    X["hour"] = X.timestamp.dt.hour

    X["weekday"] = X.timestamp.dt.weekday

    #X["is_holiday"] = (X.timestamp.dt.date.astype("str").isin(holidays)).astype(int)

    

    #suppression en plus de la caratéristique "dew_temperature"

    drop_features = [ "dew_temperature", "sea_level_pressure", "wind_direction", "wind_speed","year_built","floor_count","cloud_coverage", "precip_depth_1_hr"]



    X.drop(drop_features, axis=1, inplace=True)



    if test:

        row_ids = X.row_id

        X.drop("row_id", axis=1, inplace=True)

        return X, row_ids

    else:

        y = np.log1p(X.meter_reading)

        X.drop("meter_reading", axis=1, inplace=True)

        return X, y
plt.style.use("seaborn")

sns.set(font_scale=1)
path_data = "/kaggle/input/ashrae-energy-prediction/"

path_train = path_data + "train.csv"

path_test = path_data + "test.csv"

path_building = path_data + "building_metadata.csv"

path_weather_train = path_data + "weather_train.csv"

path_weather_test = path_data + "weather_test.csv"
building = reduce_mem_usage(pd.read_csv(path_building))
building.info()
col = list(building.columns)

pct_missing = [building[c].isnull().sum() / len(building) * 100 for c in building.columns]

df_missing = pd.DataFrame({"Name feature": col, "pct_missing": pct_missing})

df_missing.set_index("Name feature", inplace=True, drop=True)

plt.figure(figsize=(10, 6))

plt.bar(df_missing.index, df_missing.pct_missing)

for i, n in enumerate(list(df_missing.index)):

    plt.text(i, df_missing.loc[n]['pct_missing'], s="{:0.2f}%".format(df_missing.loc[n]['pct_missing']), horizontalalignment="center")

plt.xticks(rotation=45, horizontalalignment="right")

plt.ylabel("Percentage of missing data")

plt.title("Proportion of missing data by feature")
labelEncoder = LabelEncoder()

labelEncoder.fit(building["primary_use"])

building["primary_use"] = labelEncoder.transform(building["primary_use"])

building.set_index("building_id", drop=True, inplace=True)
fig, ax = plt.subplots(figsize=(10, 10))

ax = sns.heatmap(building.corr(), fmt=".2f", annot=True, ax=ax, cmap="RdBu_r", vmin=-1, vmax=1)
df = building[["site_id", "primary_use", "year_built", "floor_count"]].groupby("site_id").count()

df["%_missing_data_year_built"] = df[["primary_use", "year_built"]].apply(lambda x: round((x["primary_use"] - x["year_built"]) / x["primary_use"] * 100, 2), axis=1)

df["%_missing_data_floor_count"] = df[["primary_use", "floor_count"]].apply(lambda x: round((x["primary_use"] - x["floor_count"]) / x["primary_use"] * 100, 2), axis=1)





plt.figure(figsize=(12, 6))

plt.bar(df.index - 0.2, df["%_missing_data_year_built"], width=0.4, label="year_built")

plt.bar(df.index +  0.2, df["%_missing_data_floor_count"], width=0.4, label="floor_count")

plt.xticks(range(len(df)))

plt.xlabel("site_id")

plt.ylabel("%")

plt.legend()

plt.title("Proportion of missing data for the year_built and floor_count columns for each site")
"Number of types of building: {}".format(len(set(building["primary_use"])))
print("The different types of building:")

for t in set(building["primary_use"]):

    print("- " + labelEncoder.classes_[t])
df = building[["site_id", "primary_use"]].groupby("site_id").agg({"primary_use":["nunique", "count"]})



fig, ax1 = plt.subplots(figsize=(15, 8))



ax1.bar(df.index - 0.2, df["primary_use"]["nunique"], width=0.4, color="orange", label="Number of building types")

for i, row in df.iterrows():

    ax1.text(i - 0.2, row["primary_use"]["nunique"] + 0.2, s=row["primary_use"]["nunique"], horizontalalignment="center")



ax2 = ax1.twinx()



ax2.bar(df.index + 0.2, df["primary_use"]["count"], width=0.4, label="Number of buildings")

for i, row in df.iterrows():

    plt.text(i + 0.2, row["primary_use"]["count"], s=row["primary_use"]["count"], horizontalalignment="center")



ax1.set_xlabel("site_id")

ax1.set_ylabel("Number of building types")

ax2.set_ylabel("Number of buildings per site")

plt.xticks(range(16))

ax1.legend()

ax2.legend()

ax1.grid(visible=False)

ax2.grid(visible=False)

plt.title("Number of building types per site")
df = building[["primary_use", "site_id"]].groupby("primary_use").count()

df.sort_values("site_id", ascending=False, inplace=True)

df.reset_index(inplace=True, drop=False)

df["percentage"] = df["site_id"].apply(lambda x: round((x / df["site_id"].sum()) * 100, 2))

plt.figure(figsize=(14, 5))

plt.bar(df.index, df["site_id"])

for i, n in enumerate(list(df.index)):

    plt.text(i, df.iloc[i]["site_id"], s="{}%".format(df.iloc[i]["percentage"]), horizontalalignment="center")

plt.xticks(range(len(df)), df["primary_use"])

plt.ylabel("Number of sites")

plt.title("Number of buildings per site")
building[["square_feet"]].boxplot()
#Remplacement des outliers par le seuil

seuil_min, seuil_max = seuil_min_max(building["square_feet"], 2)

building["square_feet"] = replaceOutliers(building["square_feet"], seuil_min, seuil_max)
building[["square_feet"]].describe()
building.drop(["year_built", "floor_count"], axis=1, inplace=True)
weather_train = reduce_mem_usage(pd.read_csv(path_weather_train))
weather_train.info()
weather_train["timestamp"] = pd.to_datetime(weather_train["timestamp"], format="%Y-%m-%d %H:%M:%S")
print("First Date: {}".format(weather_train["timestamp"].min()))

print("Last Date: {}".format(weather_train["timestamp"].max()))
col = []

pct = []

for c in weather_train.columns:

    col.append(c)

    pct.append(weather_train[c].isnull().sum() / len(weather_train) * 100)



a = pd.DataFrame({"colonne": col, "pct": pct}).set_index("colonne")
plt.figure(figsize=(10, 6))

plt.bar(a.index, a["pct"])

for i, c, in enumerate(a.index):

    y = a.iloc[i]["pct"]

    plt.text(i, y=y, s="{:0.2f}%".format(y), horizontalalignment="center")



plt.xticks(rotation =45, horizontalalignment="right")



plt.ylabel("Percentage")

plt.title("Percentage of missing data by data type")
weather_train[["precip_depth_1_hr"]].dropna().describe()
#Replacement -1 by 0

weather_train['precip_depth_1_hr'] = np.where(weather_train["precip_depth_1_hr"] == -1, 0, weather_train["precip_depth_1_hr"])
#I want to know how many observation with rain in the entire database Weather_train

weather_train["rain"] = weather_train["precip_depth_1_hr"].dropna().apply(lambda x: 1 if x > 0 else 0)

plt.figure(figsize=(6, 6))

plt.pie(weather_train[["rain", "site_id"]].dropna().groupby("rain").count(), 

        explode=(0, 0.2), 

        labels=["without rain", "rain"], 

        shadow=True,

        autopct='%1.1f%%')
weather_train.drop(["precip_depth_1_hr", "cloud_coverage", "rain"], axis=1, inplace=True)
iterativeImputer = IterativeImputer()
iterativeImputer.fit(weather_train[['site_id', 'dew_temperature', 'air_temperature', 'sea_level_pressure',

       'wind_direction', 'wind_speed']])

weather_train[['site_id', 'dew_temperature', 'air_temperature', 'sea_level_pressure',

       'wind_direction', 'wind_speed']] = iterativeImputer.transform(weather_train[['site_id', 'dew_temperature', 'air_temperature', 'sea_level_pressure',

       'wind_direction', 'wind_speed']])
fig, ax = plt.subplots(figsize=(8, 8))

sns.heatmap(weather_train.corr(), fmt=".2f", annot=True, ax=ax, cmap="RdBu_r", vmin=-1, vmax=1)
weather_train.drop("dew_temperature", axis=1, inplace=True)
weather_train[["air_temperature"]].boxplot()
#Remplace des extremes par les seuils

seuil_min_air_temp, seuil_max_air_temp = seuil_min_max(weather_train["air_temperature"], 1)

weather_train["air_temperature"] = replaceOutliers(weather_train["air_temperature"], seuil_min_air_temp, seuil_max_air_temp)
weather_train[["sea_level_pressure"]].boxplot()
seuil_min_sea_level_pressure, seuil_max_sea_level_pressure = seuil_min_max(weather_train["sea_level_pressure"])

weather_train["sea_level_pressure"] = replaceOutliers(weather_train["sea_level_pressure"], seuil_min_sea_level_pressure, seuil_max_sea_level_pressure)
weather_train[["wind_speed"]].boxplot()
seuil_min_wind_speed, seuil_max_wind_speed = seuil_min_max(weather_train["wind_speed"])

weather_train["wind_speed"] = replaceOutliers(weather_train["wind_speed"], seuil_min_wind_speed, seuil_max_wind_speed)
#Arrondir la direction du vent par dizaine

weather_train["wind_direction"] = weather_train["wind_direction"].apply(lambda x: int(round(x / 10, 0) * 10))

#remplacer 360 par 0

weather_train["wind_direction"] = weather_train["wind_direction"].apply(lambda x: 0 if x == 360 else x)
plt.hist(weather_train["wind_direction"], bins=35)
train, target, encoder, imputer, seuils = preparationDonnees(path_train, path_building, path_weather_train)
train.info()
train["meter_reading"] = target
features = ["building", "meter", "count", "mean", "std", "min", "25%", "50%", "75%", "maxi"]

df_building = pd.DataFrame(columns=features)



compteur = 0

for i in range(1500):

    for j in range(4):

        train_building = train[np.logical_and(train["building_id"] == i,  train["meter"] == j)]

        df = train_building[["meter_reading", "MONTH"]].groupby("MONTH").sum()

        df_building.loc[compteur] = [i] + [j] + list(df.describe()["meter_reading"])

        compteur += 1
df_building.info()
df_building["std / mean"] = df_building["std"] / df_building["mean"]
#Building_id without data

building_id_without_data = []

for i in range(1500):

    df_b = df_building[df_building["building"] == i]

    if df_b["std"].isnull().sum() == 4:

        building_id_without_data.append(i)
"Number buiding without data: {} ({:0.2f}%)".format(len(building_id_without_data), len(building_id_without_data) / 1500 * 100)
index_to_eliminate = []

for b in building_id_without_data:

    df = train[train["building_id"] == b]

    index_to_eliminate.extend(list(df.index))
index_to_keep = set(train.index).difference(set(index_to_eliminate))
train = train[train.index.isin(index_to_keep)]
#Data by meter

df_building[["meter", "std"]].groupby("meter").count().plot(kind="bar")

plt.title("Number building with data by meter")

meter = {0: "electricity", 1: "chilledwater", 2: "steam", 3: "hotwater"}

plt.xticks(range(4), ["electricity", "chilledwater", "steam", "hotwater"], rotation=45, horizontalalignment="right")

plt.ylabel("Number of building")
#outliers 

df_building[["std / mean"]].boxplot()
threshold_building = (df_building[["std / mean"]].quantile(0.75) - df_building[["std / mean"]].quantile(0.25)) + df_building[["std / mean"]].quantile(0.75)

threshold_building = threshold_building[0]
df_outliers = df_building[np.logical_or(df_building["std"].isnull(), df_building["std / mean"] > threshold_building)]
#Building id outliers

number_graph = len(df_outliers[df_outliers["std / mean"].notnull()])

fig = plt.figure(figsize=(17, 100))

j = 0

for k in range(4):

    for i, row in df_outliers[np.logical_and(df_outliers["std / mean"].notnull(), df_outliers["meter"] == k)].iterrows():

        train_building_meter = train[np.logical_and(train["building_id"] == row["building"], train["meter"] == row["meter"])]

        fig.add_subplot(int(number_graph / 3) + 1, 3, j + 1)

        plt.plot(train_building_meter["timestamp"], train_building_meter["meter_reading"])

        plt.title("Building: {} - Meter: {}".format(int(row["building"]), int(row["meter"])))

        j += 1

df_traces = pd.DataFrame(columns=["building", "meter", "start date", "end date", "duration"])

for i, row in df_outliers[df_outliers["std / mean"].notnull()].iterrows():

    df = train[np.logical_and(train["building_id"] == row["building"], train["meter"] == row["meter"])]

    df_0 = df[df["meter_reading"] == 0][["timestamp"]]

    if len(df_0) > 0:

        df_0["delta"] = df_0["timestamp"].diff()

        periode = 0

        date1 = df_0.iloc[0]["timestamp"]

        for i in range(1, len(df_0)):

            if df_0.iloc[i]["delta"] == timedelta(hours=1):

                periode += 1

            elif periode > 0:

                date2 = df_0.iloc[i-1]["timestamp"]

                len_df_traces = len(df_traces)

                df_traces.loc[len_df_traces + 1] = [int(row["building"]), int(row["meter"]), date1, date2, periode]

                date1 = df_0.iloc[i]["timestamp"]

                periode = 0

            else:

                date1 = df_0.iloc[i]["timestamp"]
pickle.dump(df_traces, open("df_traces.pickle", "wb"))
df_traces = df_traces.sort_values("duration", ascending=False)



plt.plot(range(len(df_traces)), df_traces["duration"])

plt.xlim(-10, 1000)

plt.ylabel("number of consecutive hours with zero energy consumption")

plt.xlabel("Observation")

plt.title("Decreasing ranking of periods with zero energy consumption on Outliers buildings")
df_traces_to_eliminate = df_traces[df_traces["duration"] >= 117]

df_traces_to_eliminate
index_to_eliminate = []

for i, row in df_traces_to_eliminate.iterrows():

    date1 = row["start date"]

    date2 = row["end date"]

    df_temp = train[np.logical_and(train["building_id"] == row["building"], train["meter"] == row["meter"])]

    df_temp = df_temp[np.logical_and(df_temp["timestamp"] >= date1, df_temp["timestamp"] <= date2)]

    index_to_eliminate.extend(list(df_temp.index))
index_to_keep = set(train.index).difference(set(index_to_eliminate))
train = train[train.index.isin(index_to_keep)]
def dataPreparation(data, building, weather, encoder=None, imputer=None, seuils=None):

    t0 = time.time()

    

    #Preparation of the BUILDING table

    print("Preparation of the BUILDING table")

    building = reduce_mem_usage(pd.read_csv(building))

    #Treatment of the outliers

    seuil_min, seuil_max = seuil_min_max(building["square_feet"], 2)

    building["square_feet"] = replaceOutliers(building["square_feet"], seuil_min, seuil_max)

    building.drop(["year_built", "floor_count"], axis=1, inplace=True)

    #value rounding of the column square_feet

    building["square_feet"] = building["square_feet"].apply(lambda x: int(x / 10) * 10)

    building["square_feet"] = np.log1p(building["square_feet"])

    

    #Preparation of the Weather table

    print("Preparation of the Weather table")

    col_weather = ["air_temperature", "sea_level_pressure", "wind_direction", "wind_speed"]

    weather = reduce_mem_usage(pd.read_csv(weather))

    weather.drop(["precip_depth_1_hr", "cloud_coverage", "dew_temperature"], axis=1, inplace=True)

    weather["timestamp"] = weather["timestamp"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))

    if encoder is None and imputer is None:

        encoder = LabelEncoder()

        encoder.fit(building["primary_use"])

        building["primary_use"] = encoder.transform(building["primary_use"])

        imputer = IterativeImputer()

        imputer.fit(weather[col_weather])

        seuils={}

        seuil_min_air, seuil_max_air = seuil_min_max(weather["air_temperature"])

        seuil_min_pressure, seuil_max_pressure = seuil_min_max(weather["sea_level_pressure"])

        seuil_min_wind, seuil_max_wind = seuil_min_max(weather["wind_speed"])

        seuils["seuil_min_air"] = seuil_min_air

        seuils["seuil_max_air"] = seuil_max_air

        seuils["seuil_min_pressure"] = seuil_min_pressure

        seuils["seuil_max_pressure"] = seuil_max_pressure

        seuils["seuil_min_wind"] = seuil_min_wind

        seuils["seuil_max_wind"] = seuil_max_wind    

    else:

        building["primary_use"] = encoder.transform(building["primary_use"])

        seuil_min_air = seuils["seuil_min_air"]

        seuil_max_air = seuils["seuil_max_air"]

        seuil_min_pressure = seuils["seuil_min_pressure"]

        seuil_max_pressure = seuils["seuil_max_pressure"]

        seuil_min_wind = seuils["seuil_min_wind"]

        seuil_max_wind = seuils["seuil_max_wind"]

    

    #### DATA ####

    print("MERGE TABLES DATA, BUILDING AND WEATHER")

    data = reduce_mem_usage(pd.read_csv(data))

    data["timestamp"] = data["timestamp"].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))

    ### FUSION TRAIN ET BUILDING --> TRAIN ####

    print("Fusion de 'data' et 'building'")

    data = pd.merge(data, building, on="building_id", how="left")

    #### FUSION TRAIN ET WEATHER --> TRAIN ####

    print("Fusion de data et weather")

    data = pd.merge(data, weather, on=["timestamp", "site_id"], how="left")

    

    #Création des colonnes MONTH, DAY et HOUR

    print("Création de la colonne Month")

    data["MONTH"] = data["timestamp"].apply(lambda x: x.month)

    print("Création de la colonne Day")

    data["DAY"] = data["timestamp"].apply(lambda x: x.day)

    print("Création de Hour")

    data["HOUR"] = data["timestamp"].apply(lambda x: x.hour)

    print("Création du jour de la semaine")

    data["DAYOFWEEK"] = data["timestamp"].apply(lambda x: x.dayofweek)

    

    #### detection of buildings with abnormal data ####

    

    if "meter_reading" in list(data.columns):

        print("detection of buildings with abnormal data")

        features = ["building", "meter", "count", "mean", "std", "min", "25%", "50%", "75%", "maxi"]

        df_building = pd.DataFrame(columns=features)



        compteur = 0

        for i in range(1500):

            for j in range(4):

                train_building = data[np.logical_and(data["building_id"] == i,  data["meter"] == j)]

                df = train_building[["meter_reading", "MONTH"]].groupby("MONTH").sum()

                df_building.loc[compteur] = [i] + [j] + list(df.describe()["meter_reading"])

                compteur += 1

    

        #Building_id without data

        print("Building_id without data")

        building_id_without_data = []

        for i in range(1500):

            df_b = df_building[df_building["building"] == i]

            if df_b["std"].isnull().sum() == 4:

                building_id_without_data.append(i)

        

        print("Elimination of the Building without data")

        index_to_eliminate = []

        for b in building_id_without_data:

            df = data[data["building_id"] == b]

            index_to_eliminate.extend(list(df.index))

    

        index_to_keep = set(data.index).difference(set(index_to_eliminate))

        data = data[data.index.isin(index_to_keep)]

        

        

        print("Detection abnormal meter_reading at 0")

        df_building["std / mean"] = df_building["std"] / df_building["mean"]

        threshold_building = (df_building[["std / mean"]].quantile(0.75) - df_building[["std / mean"]].quantile(0.25)) + df_building[["std / mean"]].quantile(0.75)

        df_outliers = df_building[np.logical_or(df_building["std"].isnull(), df_building["std / mean"] > threshold_building[0])]

    

        df_traces = pd.DataFrame(columns=["building", "meter", "start date", "end date", "duration"])

        for i, row in df_outliers[df_outliers["std / mean"].notnull()].iterrows():

            df = data[np.logical_and(data["building_id"] == row["building"], data["meter"] == row["meter"])]

            df_0 = df[df["meter_reading"] == 0][["timestamp"]]

            if len(df_0) > 0:

                df_0["delta"] = df_0["timestamp"].diff()

                periode = 0

                date1 = df_0.iloc[0]["timestamp"]

                for i in range(1, len(df_0)):

                    if df_0.iloc[i]["delta"] == timedelta(hours=1):

                        periode += 1

                    elif periode > 0:

                        date2 = df_0.iloc[i-1]["timestamp"]

                        len_df_traces = len(df_traces)

                        df_traces.loc[len_df_traces + 1] = [int(row["building"]), int(row["meter"]), date1, date2, periode]

                        date1 = df_0.iloc[i]["timestamp"]

                        periode = 0

                    else:

                        date1 = df_0.iloc[i]["timestamp"]

        

        print("Elimination of observations whose meter_reading is at 0 for more than 5 days")

        df_traces_to_eliminate = df_traces[df_traces["duration"] >= 117]

        

        index_to_eliminate = []

        for i, row in df_traces_to_eliminate.iterrows():

            date1 = row["start date"]

            date2 = row["end date"]

            df_temp = data[np.logical_and(data["building_id"] == row["building"], data["meter"] == row["meter"])]

            df_temp = df_temp[np.logical_and(df_temp["timestamp"] >= date1, df_temp["timestamp"] <= date2)]

            index_to_eliminate.extend(list(df_temp.index))

    

        index_to_keep = set(data.index).difference(set(index_to_eliminate))

        data = data[data.index.isin(index_to_keep)]



    print('filling cells without data')

    filtre = np.logical_or(data["air_temperature"].isnull(), data["sea_level_pressure"].isnull())

    filtre2 = np.logical_or(data["wind_direction"].isnull(), data["wind_speed"].isnull())

    filtre = np.logical_or(filtre, filtre2)

    

    d2 = data[filtre][col_weather]

    d = imputer.transform(d2)

    d2 = pd.DataFrame(data=d, index=d2.index, columns=col_weather)

    data.update(d2)

    

    d2 = None

    d = None

    filtre = None

    filtre2 = None

    weather = None

    building = None

    

    

    #Ecrasement des outliers

    print("Replacements of the outliers")

    data["air_temperature"] = replaceOutliers(data["air_temperature"], seuil_min_air, seuil_max_air)

    data["sea_level_pressure"] = replaceOutliers(data["sea_level_pressure"], seuil_min_pressure, seuil_max_pressure)

    data["wind_speed"] = replaceOutliers(data["wind_speed"], seuil_min_wind, seuil_max_wind)

    #arrondi de la temperature

    data["air_temperature"] = np.round(data["air_temperature"], decimals=1)

    #Arrondir la direction du vent par dizaine

    data["wind_direction"] = data["wind_direction"].apply(lambda x: int(round(x / 10, 0) * 10))

    #remplacer 360 par 0

    data["wind_direction"] = data["wind_direction"].apply(lambda x: 0 if x == 360 else x)



    data = reduce_mem_usage(data)

    

    if "meter_reading" in list(data.columns):

        data = data[['timestamp', 'site_id', 'building_id', 'meter', 'meter_reading',

                 'primary_use', 'square_feet', 'air_temperature',

                 'sea_level_pressure', 'wind_direction', 'wind_speed']]

    else:

        data = data[['timestamp', 'site_id', 'building_id', 'meter',

                 'primary_use', 'square_feet', 'air_temperature',

                 'sea_level_pressure', 'wind_direction', 'wind_speed']]

    

    #Suppression de la colonne TIMESTAMP

    data.drop("timestamp", axis=1, inplace=True)

    

    #Suppression de la colonne building_id

    data.drop("building_id", axis=1, inplace=True)

    data.drop("site_id", axis=1, inplace=True)

    

    try:

        data.drop("row_id", axis=1, inplace=True)

    except:

        pass

    

    print("Réduction de la place mémoire")

    data = reduce_mem_usage(data)

    

    #suppression des données en double (data leakage)

    if "meter_reading" in data.columns:

        target = data["meter_reading"]

        data.drop("meter_reading", axis=1, inplace=True)

        data.drop_duplicates(inplace=True)

        filtre = data.duplicated()

        print("Nombre de ligne en double dans le train: {}".format(filtre.sum()))

        data = data[np.logical_not(filtre)]

        target = target[data.index]

        target = np.log1p(target)

    else:

        target = None

    

    print("Durée exécution: {:0.2f} secondes".format(time.time() - t0))

    return data, target, encoder, imputer, seuils
train, target, encoder, imputer, seuils = dataPreparation(path_train, path_building, path_weather_train)
pickle.dump(train, open("train.pickle", "wb"))

pickle.dump(target, open("target.pickle", "wb"))
pickle.dump(encoder, open("encoder.pickle", "wb"))

pickle.dump(imputer, open("imputer.pickle", "wb"))

pickle.dump(seuils, open("seuils.pickle", "wb"))
X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=0.3, shuffle=True, random_state=42)
lgr = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.1, num_leaves=512, random_state=45)

t0 = time.time()

lgr.fit(X_train, 

        y_train, 

        eval_set=[(X_train, y_train), (X_test, y_test)], 

        eval_metric="rmse",

        categorical_feature=[0,1,2],

        verbose=True)

print("Durée: {:0.2f} secondes".format(time.time() - t0))
train_l2, val_l2 = lgr.evals_result_["training"]["rmse"], lgr.evals_result_["valid_1"]["rmse"]

plt.plot(range(len(train_l2)), train_l2, label="train")

plt.plot(range(len(val_l2)), val_l2, label="val")

plt.legend()

plt.ylabel("score rmse")

plt.xlabel("iteration")
regr = BaggingRegressor(base_estimator=lgr, n_estimators=10, verbose=4)
regr.fit(X_train, y_train)
pickle.dump(regr, open("regr.pickle", "wb"))
regr.score(X_train, y_train)
regr.score(X_test, y_test)
test, target, encoder, imputer, seuils = dataPreparation(path_test, path_building, path_weather_test, encoder, imputer, seuils)
pickle.dump(test, open("test.pickle", "wb"))
seuils = np.linspace(start=0, stop=len(test), num=30)
target_predict_regr = []

count=0

for i in range(len(seuils) - 1):

    t0 = time.time()

    count += 1

    print("Count: " + str(i))

    print(int(seuils[i]), "-", int(seuils[i + 1] - 1))

    seuil_1 = int(seuils[i])

    seuil_2 = int(seuils[i + 1] - 1)

    res = regr.predict(test.loc[seuil_1:seuil_2])

    target_predict_regr.extend(list(res))

    print("Durée execution: {:0.2f} secondes".format(time.time() - t0))
submission = pd.DataFrame()

target_predict_regr_exp = np.round(np.expm1(target_predict_regr), decimals=4)

submission["meter_reading"] = target_predict_regr_exp

submission.index.name = "row_id"

submission.to_csv("submission_regr.csv")