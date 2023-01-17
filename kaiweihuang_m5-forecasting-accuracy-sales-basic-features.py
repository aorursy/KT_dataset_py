# Set environment variables

import os

import warnings

import numpy as np

import pandas as pd



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



df_sales_train_validation = pd.read_csv("sales_train_validation.csv")

df_sales_train_validation.head(10)
# Add another 28 days with null values to make predictions successfully



number_of_train = 1913

days_to_predict = 28



for i in range(days_to_predict):

    prediction_d = number_of_train + (i + 1)

    df_sales_train_validation[f"d_{prediction_d}"] = np.nan

df_sales_train_validation.head(10)
# Create features

# Melt the dataframe to have "sales everyday" as a feature



index_columns = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]



df_sales_features = df_sales_train_validation.melt(

    id_vars = index_columns

    , var_name = "d"

    , value_name = "sales"

)

df_sales_features.head(10)
# Memory usage control



memory_usage_string = format_memory_usage(df_sales_features.memory_usage().sum())

print(f"Original memory usage: {memory_usage_string}")



# Technics: converting strings to categorical variables

for column in index_columns:

    df_sales_features[column] = df_sales_features[column].astype("category")



memory_usage_string = format_memory_usage(df_sales_features.memory_usage().sum())

print(f"Reduced memory usage: {memory_usage_string}")
# Load price dataset



df_sell_prices = pd.read_csv("sell_prices.csv")

df_sell_prices.head(10)
# Create features

# Items are available after that a certain time



df_available_after = df_sell_prices.groupby(["store_id","item_id"])["wm_yr_wk"].agg(["min"]).reset_index()

df_available_after.columns = ["store_id", "item_id", "available_after"]

df_available_after.head(10)
# Join df_sales_features and df_available_after



df_sales_features = merge_by_concat(df_sales_features, df_available_after, ["store_id", "item_id"])

df_sales_features.head(10)
# We can drop those rows before available date

# To achieve this, we need df_calendar's help



df_calendar = pd.read_csv("calendar.csv")

df_calendar.head(10)
# Join df_sales_features and df_calendar



df_sales_features = merge_by_concat(df_sales_features, df_calendar[["d", "wm_yr_wk"]], ["d"])

df_sales_features.head(10)
# We only need those entries after "available_after"



df_sales_features = df_sales_features[df_sales_features["wm_yr_wk"] >= df_sales_features["available_after"]]

df_sales_features = df_sales_features.reset_index(drop = True)

df_sales_features.head(10)
# Memory usage control



memory_usage_string = format_memory_usage(df_sales_features.memory_usage().sum())

print(f"Original memory usage: {memory_usage_string}")



# Technics: we know the minimum of a certain column, so we find the difference between each row and its minimum

# and store those differences in int16

df_sales_features.drop(["wm_yr_wk"], axis = 1, inplace = True)

df_sales_features["available_after"] = (df_sales_features["available_after"] - df_sales_features["available_after"].min()).astype(np.int16)



# Technics: for column "d", we would like to store it with int16 format

df_sales_features["d"] = df_sales_features["d"].apply(lambda x: int(x[2:])).astype(np.int16)



memory_usage_string = format_memory_usage(df_sales_features.memory_usage().sum())

print(f"Reduced memory usage: {memory_usage_string}")
# Sort values to easily join features later



df_sales_features.sort_values(by = ["id", "d"], inplace = True)

df_sales_features.reset_index(drop = True, inplace = True)
# Check dataset



df_sales_features.head(10)
# Check data type



df_sales_features.info()
# Change to output path



try:

    os.chdir(BASE_PATH)

    print(f"Change to directory: {os.getcwd()}")

except:

    os.mkdir(BASE_PATH)

    os.chdir(BASE_PATH)

    print(f"Create and change to directory: {os.getcwd()}")
# Save pickle file



df_sales_features.to_pickle("sales_basic_features.pkl")