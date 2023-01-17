import os, sys, random, gc, math, glob

import numpy as np

import pandas as pd

import io, timeit, os, gc, requests

from tqdm import tqdm

import warnings

import requests, json, zipfile

import re

from io import BytesIO



pd.set_option('display.max_colwidth', -1)

pd.set_option('display.max_columns', 500)

pd.set_option('display.max_rows', 4000)



import matplotlib.pyplot as plt

%matplotlib inline



path_data = "/kaggle/input/ashrae-energy-prediction/"

TRAIN_FILE = path_data + "train.csv"

TEST_FILE = path_data + "test.csv"

TRAIN_BUILDING_FILE = path_data + "building_metadata.csv"
## Memory optimization

from pandas.api.types import is_datetime64_any_dtype as is_datetime

from pandas.api.types import is_categorical_dtype



def df_optimization(df, use_float16=False, verbose=False):

    """

    Iterate through all the columns of a dataframe and modify the data type to reduce memory usage.        

    """

    

    start_mem = df.memory_usage().sum() / 1024**2

    print("Memory usage of dataframe is {:.2f} MB".format(start_mem))

    

    for col in df.columns:

        if is_datetime(df[col]) or is_categorical_dtype(df[col]):

            continue

        col_type = df[col].dtype

        

        if col_type != object:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == "int":

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

        else:

            df[col] = df[col].astype("category")

        

        if verbose:

            print("col: %s was %s and is %s" % (col, col_type, df[col].dtype))

    

    end_mem = df.memory_usage().sum() / 1024**2

    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))

    print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))

    

    return df
# Read mapping (matching done locally simply with pearson correlation)

site1_pd = pd.read_csv("/kaggle/input/ucl50buildings/site1_scrapped_50_buildings.csv", encoding="UTF-8")

site1_pd.head(10)
# Convert UCL data to ASHREA format.

# UCL data is by half hour so we have to sum per hour.

def as_ashrae_format(df, coef=1.0):

    df.rename(columns= {"Unnamed: 0":"date"}, inplace=True)

    df = df.set_index('date').T.reset_index()

    flat_pd = pd.melt(df, id_vars=["index"], var_name="date", value_name="meter_reading_scraped")

    flat_pd["timestamp"] = flat_pd["date"] + " "+ flat_pd["index"]

    flat_pd.drop(columns = ["index", "date"], inplace=True)

    flat_pd["timestamp"] = pd.to_datetime(flat_pd["timestamp"])

    flat_pd["meter"] = 0

    flat_pd["meter_reading_scraped"] = flat_pd["meter_reading_scraped"].astype(np.float64) * coef

    flat_pd = flat_pd.set_index("timestamp").sort_index().reset_index()

    flat_pd["minute"] = flat_pd["timestamp"].dt.minute

    flat_pd["hour"] = flat_pd["timestamp"].dt.hour

    flat_pd["day"] = flat_pd["timestamp"].dt.day

    flat_pd["month"] = flat_pd["timestamp"].dt.month

    flat_pd["year"] = flat_pd["timestamp"].dt.year

    flat_pd["meter_reading_scraped"] = flat_pd.groupby(["year", "month", "day", "hour"])["meter_reading_scraped"].transform(np.nansum)

    flat_pd = flat_pd[flat_pd["minute"] == 0].reset_index(drop=True)

    flat_pd.drop(columns = ["year", "month", "day", "hour", "minute"], inplace=True)

    return flat_pd
def download_building_data(download_url, building_id):

    tmp_pd = None

    r = requests.get(download_url, stream=True)

    if r.status_code == 200:

         if r.headers['Content-Disposition'].find("attachment") >= 0:

            result = re.search('attachment; filename="(.*)"', r.headers['Content-Disposition'])

            if result is not None:

                filename = result.group(1)

                print("Downloading %s: %s" % (filename, download_url))

                r.raw.decode_content = True

                in_memory = BytesIO(r.content)

                with zipfile.ZipFile(in_memory) as archive:

                    files = archive.namelist()

                    for file in files:

                        if file.find("elec") > 0:

                            tmp_pd = as_ashrae_format(pd.read_csv(archive.open(file), skiprows=3), coef=1.0)

                            tmp_pd["building_id"] = int(building_id)

                return tmp_pd
# Scrap site1 data

site1_scraped_pd = None

for idx, row in site1_pd.iterrows():

    tmp_pd = download_building_data(row["url"], row["building_id"])

    if site1_scraped_pd is None:

        site1_scraped_pd = tmp_pd

    else:

        site1_scraped_pd = pd.concat([site1_scraped_pd, tmp_pd], axis=0)
# Sort by building_id, timestamp

site1_scraped_pd = site1_scraped_pd.set_index(["building_id", "meter", "timestamp"]).sort_index().reset_index()
site1_scraped_pd.to_pickle("site1.pkl")
# Building 152 is missing.

scraped_buildings = site1_scraped_pd["building_id"].unique()

scraped_buildings
train_pd = pd.read_csv(TRAIN_FILE)

train_pd["timestamp"] = pd.to_datetime(train_pd["timestamp"], format="%Y-%m-%d %H:%M:%S")

train_pd = df_optimization(train_pd)

print(train_pd.info())

train_pd.head()
train_building_pd = pd.read_csv(TRAIN_BUILDING_FILE) 

train_building_pd = df_optimization(train_building_pd)

train_building_pd.head()
# Keep only site1

train_pd = pd.merge(train_pd, train_building_pd[["site_id", "building_id"]], on="building_id")

train_pd = train_pd.query("site_id == 1")
# Join scraped data

train_pd = pd.merge(train_pd, site1_scraped_pd, on=["building_id", "meter", "timestamp"], how="left")
train_pd.query("building_id == 138 & meter == 0").head()
# Read test data to get timestamp for further join

test_pd = pd.read_csv(TEST_FILE)

test_pd["timestamp"] = pd.to_datetime(test_pd["timestamp"], format="%Y-%m-%d %H:%M:%S")

test_pd = df_optimization(test_pd)

print(test_pd.info())

test_pd.head()
# Keep only site1

test_pd = pd.merge(test_pd, train_building_pd[["site_id", "building_id"]], on="building_id")

test_pd = test_pd.query("site_id == 1")

test_pd.head()
# Join scraped data

test_pd = pd.merge(test_pd, site1_scraped_pd, on=["building_id", "meter", "timestamp"], how="left").drop(columns=["row_id"])
full_pd = pd.concat([train_pd, test_pd], axis=0).set_index(["building_id", "meter", "timestamp"]).sort_index().reset_index()

full_pd.head(100)
# Building 152 is missing

# Blue: Original data

# Green: Scraped

for bid in scraped_buildings:

    fig, ax = plt.subplots(figsize=(24, 5))

    t1 = full_pd.query("building_id == %d & timestamp >= '2016-01-01 00:00:00' & timestamp < '2019-01-01 00:00:00' & meter == 0" % bid).set_index("timestamp")

    d = t1.plot(kind='line', y="meter_reading", ax=ax, c='blue')

    d = t1.plot(kind='line', y="meter_reading_scraped", ax=ax, c='green', alpha=0.5, grid=True)

    diff = np.sum(t1["meter_reading"] - t1["meter_reading_scraped"])

    plt.title("Building id: %d (%d), diff=%.3f" % (bid, len(t1), diff))

    plt.show()