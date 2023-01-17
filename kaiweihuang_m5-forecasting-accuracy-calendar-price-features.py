# Set environment variables

import os

import warnings

import numpy as np

import pandas as pd

from math import ceil



VERSION = 1

INPUT_PATH = f"/kaggle/input/m5-forecasting-accuracy"

BASE_PATH = f"/kaggle/working/m5-forecasting-accuracy-ver{VERSION}"
# Turn off warnings



warnings.filterwarnings("ignore")
# Change directory



os.chdir(INPUT_PATH)

print(f"Change to directory: {os.getcwd()}")
# Memory usage function and merge by concat function (not to lose data type)



def format_memory_usage(total_bytes):

    unit_list = ["", "Ki", "Mi", "Gi"]

    for unit in unit_list:

        if total_bytes < 1024:

            return f"{total_bytes:.2f}{unit}B"

        total_bytes /= 1024

    return f"{total_bytes:.2f}{unit}B"



def merge_by_concat(df1, df2, columns):

    df_temp = df1[columns]

    df_temp = df_temp.merge(df2, on = columns, how = "left")

    new_columns = [column for column in list(df_temp) if column not in columns]

    df1 = pd.concat([df1, df_temp[new_columns]], axis = 1)

    return df1
# Load and check dataset



df_calendar = pd.read_csv("calendar.csv")

df_calendar.head(10)
# Select the necessary information

# For example, we can extract day, month or year information from "date" column



calendar_selected_columns = [

    "date"

    , "d"

    , "event_name_1"

    , "event_type_1"

    , "event_name_2"

    , "event_type_2"

    , "snap_CA"

    , "snap_TX"

    , "snap_WI"

]

df_calendar_features = df_calendar[calendar_selected_columns]
# Memory usage control



memory_usage_string = format_memory_usage(df_calendar_features.memory_usage().sum())

print(f"Original memory usage: {memory_usage_string}")



# Technics: converting strings to categorical variables

calendar_category_columns = [

    "event_name_1"

    , "event_type_1"

    , "event_name_2"

    , "event_type_2"

    , "snap_CA"

    , "snap_TX"

    , "snap_WI"

]

for column in calendar_category_columns:

    df_calendar_features[column] = df_calendar_features[column].astype("category")



memory_usage_string = format_memory_usage(df_calendar_features.memory_usage().sum())

print(f"Reduced memory usage: {memory_usage_string}")
# Create features

# Convert date to datetime variables and store the derivative information in int8



memory_usage_string = format_memory_usage(df_calendar_features.memory_usage().sum())

print(f"Original memory usage: {memory_usage_string}")



df_calendar_features["date"] = pd.to_datetime(df_calendar_features["date"])

df_calendar_features["day"] = df_calendar_features["date"].dt.day.astype(np.int8)

df_calendar_features["weekday"] = df_calendar_features["date"].dt.dayofweek.astype(np.int8)

df_calendar_features["week"] = df_calendar_features["date"].dt.week.astype(np.int8)

df_calendar_features["month"] = df_calendar_features["date"].dt.month.astype(np.int8)

df_calendar_features["year"] = (df_calendar_features["date"].dt.year - df_calendar_features["date"].dt.year.min()).astype(np.int8)

df_calendar_features["week_of_month"] = df_calendar_features["date"].dt.day.apply(lambda x: ceil(x / 7)).astype(np.int8)

df_calendar_features["is_weekend"] = (df_calendar_features["weekday"] >= 5).astype(np.int8)



# Technics: for column "d", we would like to store it with int16 format

df_calendar_features["d"] = df_calendar_features["d"].apply(lambda x: int(x[2:])).astype(np.int16)



memory_usage_string = format_memory_usage(df_calendar_features.memory_usage().sum())

print(f"Memory usage after columns added: {memory_usage_string}")
# Check dataset



df_calendar_features.head(10)
# Check data type



df_calendar_features.info()
# Load and check dataset



df_sell_prices = pd.read_csv("sell_prices.csv")

df_sell_prices.head(10)
# Create features

# Selling prices are not as fluctuating as we expect,

# so we only need several characteristics to capture their distribution



df_sell_prices_grouped = df_sell_prices.groupby(["store_id", "item_id"])



memory_usage_string = format_memory_usage(df_sell_prices.memory_usage().sum())

print(f"Original memory usage: {memory_usage_string}")



df_sell_prices["price_max"] = df_sell_prices_grouped["sell_price"].transform("max").astype(np.float16)

df_sell_prices["price_min"] = df_sell_prices_grouped["sell_price"].transform("min").astype(np.float16)

df_sell_prices["price_mean"] = df_sell_prices_grouped["sell_price"].transform("mean").astype(np.float16)

df_sell_prices["price_std"] = df_sell_prices_grouped["sell_price"].transform("std").astype(np.float16)

df_sell_prices["price_scaled"] = (

    (df_sell_prices["sell_price"] - df_sell_prices["price_min"])

    / (df_sell_prices["price_max"] - df_sell_prices["price_min"])

).astype(np.float16)

df_sell_prices["price_nunique"] = df_sell_prices_grouped["sell_price"].transform("nunique").astype(np.int16)

df_sell_prices["item_nunique"] = df_sell_prices.groupby(["store_id", "sell_price"])["item_id"].transform("nunique").astype(np.int16)



memory_usage_string = format_memory_usage(df_sell_prices.memory_usage().sum())

print(f"Memory usage after columns added: {memory_usage_string}")
# Check dataset



df_sell_prices.head(10)
# Join df_sell_prices and raw df_calendar



df_price_features = merge_by_concat(df_sell_prices, df_calendar[["wm_yr_wk", "month", "year", "d"]], ["wm_yr_wk"])

df_price_features.head(10)
# Create features

# Evaluate how do prices change periodically



memory_usage_string = format_memory_usage(df_price_features.memory_usage().sum())

print(f"Original memory usage: {memory_usage_string}")



df_price_features["price_mean_change_week"] = (

    df_price_features["sell_price"] / df_price_features.groupby(["store_id", "item_id", "wm_yr_wk"])["sell_price"].transform("mean")

).astype(np.float16)

df_price_features["price_mean_change_month"] = (

    df_price_features["sell_price"] / df_price_features.groupby(["store_id", "item_id", "month"])["sell_price"].transform("mean")

).astype(np.float16)

df_price_features["price_mean_change_year"] = (

    df_price_features["sell_price"] / df_price_features.groupby(["store_id", "item_id", "year"])["sell_price"].transform("mean")

).astype(np.float16)



memory_usage_string = format_memory_usage(df_price_features.memory_usage().sum())

print(f"Memory usage after columns added: {memory_usage_string}")
# Check dataset



price_selected_columns = [

    "store_id"

    , "item_id"

    , "d"

    , "sell_price"

    , "price_max"

    , "price_min"

    , "price_mean"

    , "price_std"

    , "price_scaled"

    , "price_nunique"

    , "item_nunique"

    , "price_mean_change_week"

    , "price_mean_change_month"

    , "price_mean_change_year"

]

df_price_features = df_price_features[price_selected_columns]

df_price_features.head(10)
# Memory usage control



memory_usage_string = format_memory_usage(df_price_features.memory_usage().sum())

print(f"Original memory usage: {memory_usage_string}")



# Technics: converting strings to categorical variables

price_category_columns = ["store_id", "item_id"]

for column in price_category_columns:

    df_price_features[column] = df_price_features[column].astype("category")



# Technics: for column "sell_price", we would like to store it with float16 format

df_price_features["sell_price"] = df_price_features["sell_price"].astype(np.float16)



# Technics: for column "d", we would like to store it with int16 format

df_price_features["d"] = df_price_features["d"].apply(lambda x: int(x[2:])).astype(np.int16)



memory_usage_string = format_memory_usage(df_price_features.memory_usage().sum())

print(f"Reduced memory usage: {memory_usage_string}")
# Check dataset



df_price_features.head(10)
# Check data type



df_price_features.info()
# Change to output path



try:

    os.chdir(BASE_PATH)

    print(f"Change to directory: {os.getcwd()}")

except:

    os.mkdir(BASE_PATH)

    os.chdir(BASE_PATH)

    print(f"Create and change to directory: {os.getcwd()}")
# Save pickle file



df_calendar_features.to_pickle("calendar_features.pkl")

df_price_features.to_pickle("price_features.pkl")