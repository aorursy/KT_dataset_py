# Set environment variables

import os

import time

import warnings

import numpy as np

import pandas as pd



VERSION = 1

INPUT_PATH = f"/kaggle/input/m5-forecasting-accuracy-sales-basic-features"

BASE_PATH = f"/kaggle/working/m5-forecasting-accuracy-ver{VERSION}"
# Turn off warnings



warnings.filterwarnings("ignore")
# Change directory



os.chdir(INPUT_PATH)

print(f"Change to directory: {os.getcwd()}")
# Memory usage function



def format_memory_usage(total_bytes):

    unit_list = ["", "Ki", "Mi", "Gi"]

    for unit in unit_list:

        if total_bytes < 1024:

            return f"{total_bytes:.2f}{unit}B"

        total_bytes /= 1024

    return f"{total_bytes:.2f}{unit}B"
# Set global variables



days_to_predict = 28

rolling_days = [60, 90, 180, 365]
# Load dataset from our previous work



df_rolling_features = pd.read_pickle("m5-forecasting-accuracy-ver1/sales_basic_features.pkl")

df_rolling_features.head(10)
# Get necessary columns only



df_rolling_features = df_rolling_features[["id", "d", "sales"]]

df_rolling_features.head(10)
# Create features

# Generate rolling lag features and control the memory usage



df_rolling_grouped = df_rolling_features.groupby(["id"])["sales"]



for day in rolling_days:



    start_time = time.time()

    print(f"Rolling {str(day)} Start.")



    df_rolling_features[f"rolling_{str(day)}_max"] = df_rolling_grouped.transform(lambda x: x.shift(days_to_predict).rolling(day).max()).astype(np.float16)

    df_rolling_features[f"rolling_{str(day)}_min"] = df_rolling_grouped.transform(lambda x: x.shift(days_to_predict).rolling(day).min()).astype(np.float16)

    df_rolling_features[f"rolling_{str(day)}_median"] = df_rolling_grouped.transform(lambda x: x.shift(days_to_predict).rolling(day).median()).astype(np.float16)

    df_rolling_features[f"rolling_{str(day)}_mean"] = df_rolling_grouped.transform(lambda x: x.shift(days_to_predict).rolling(day).mean()).astype(np.float16)

    df_rolling_features[f"rolling_{str(day)}_std"] = df_rolling_grouped.transform(lambda x: x.shift(days_to_predict).rolling(day).std()).astype(np.float16)



    end_time = time.time()

    print(f"Calculation time: {round(end_time - start_time)} seconds")
# Check dataset



df_rolling_features.head(120)
# Check data type



df_rolling_features.info()
# Check current memory usage



memory_usage_string = format_memory_usage(df_rolling_features.memory_usage().sum())

print(f"Current memory usage: {memory_usage_string}")
# Change to output path



try:

    os.chdir(BASE_PATH)

    print(f"Change to directory: {os.getcwd()}")

except:

    os.mkdir(BASE_PATH)

    os.chdir(BASE_PATH)

    print(f"Create and change to directory: {os.getcwd()}")
# Save pickle file



df_rolling_features.to_pickle("sales_rolling_features.pkl")