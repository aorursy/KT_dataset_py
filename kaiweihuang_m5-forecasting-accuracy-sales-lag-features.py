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
# Load dataset from our previous work



df_lag_features = pd.read_pickle("m5-forecasting-accuracy-ver1/sales_basic_features.pkl")

df_lag_features.head(10)
# Get necessary columns only



df_lag_features = df_lag_features[["id", "d", "sales"]]

df_lag_features.head(10)
# Create features

# Generate basic lag features and control the memory usage



df_lag_grouped = df_lag_features.groupby(["id"])["sales"]



for i in range(days_to_predict):



    start_time = time.time()

    print(f"Day {str(i+1)} Start.")



    df_lag_features = df_lag_features.assign(**{f"sales_lag_{str(i+1)}": df_lag_grouped.transform(lambda x: x.shift(i + 1))})

    df_lag_features[f"sales_lag_{str(i+1)}"] = df_lag_features[f"sales_lag_{str(i+1)}"].astype(np.float16)



    end_time = time.time()

    print(f"Calculation time: {round(end_time - start_time)} seconds")
# Check dataset



df_lag_features.head(30)
# Check current memory usage



memory_usage_string = format_memory_usage(df_lag_features.memory_usage().sum())

print(f"Current memory usage: {memory_usage_string}")
# Check data type



df_lag_features.info()
# Change to output path



try:

    os.chdir(BASE_PATH)

    print(f"Change to directory: {os.getcwd()}")

except:

    os.mkdir(BASE_PATH)

    os.chdir(BASE_PATH)

    print(f"Create and change to directory: {os.getcwd()}")
# Save pickle file



df_lag_features.to_pickle("sales_lag_features.pkl")