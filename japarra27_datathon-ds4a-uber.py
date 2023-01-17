# Import libraries

import os

from glob import glob

import numpy as np

import pandas as pd

from sqlalchemy import create_engine

from pathlib import Path
# Change directory

folder_path = str(Path("/kaggle/input/transport-ways-behavior-ds4a/").absolute())

print(folder_path)
%%time

# Import datasets

yellow_df = pd.read_csv(f"{folder_path}/yellow_trips_new.csv")

green_df = pd.read_csv(f"{folder_path}/green_trips_new_2.csv")
%%time

# Import uber datasets

uber_2014 = pd.read_csv(f"{folder_path}/uber_trips_2014.csv")

uber_2015 = pd.read_csv(f"{folder_path}/uber_trips_2015.csv")
print("Descriptive stats uber_2014")

print("\n")

print(uber_2014.dtypes)

print("\n")

print(uber_2014.describe())

print("\n")

print(uber_2014.isnull().sum())

print("\n")

uber_2014.head()
# Analysis range of time uber2014

print(pd.to_datetime(uber_2014.pickup_datetime).describe())
print("Descriptive stats uber_2015")

print("\n")

print(uber_2015.dtypes)

print("\n")

print(uber_2015.describe())

print("\n")

print(uber_2015.isnull().sum())

print("\n")

uber_2015.head()
# Analysis range of time uber2014

print(pd.to_datetime(uber_2015.pickup_datetime).describe())
print("Descriptive stats yellow_df")

print("\n")

print(yellow_df.dtypes)

print("\n")

print(yellow_df.describe())

print("\n")

print(yellow_df.isnull().sum())

print("\n")

yellow_df.head()
# Analysis of time yellow_df

print(pd.to_datetime(yellow_df.pickup_datetime).describe())

print("\n")

print(pd.to_datetime(yellow_df.dropoff_datetime).describe())
# count of number of passenger yellow_df

print(yellow_df.passenger_count.value_counts().sort_index())
print("Descriptive stats green_df")

print("\n")

print(green_df.dtypes)

print("\n")

print(green_df.describe())

print("\n")

print(green_df.isnull().sum())

print("\n")

green_df.head()
# Analysis of time yellow_df

print(pd.to_datetime(green_df.pickup_datetime).describe())

print(pd.to_datetime(green_df.dropoff_datetime).describe())
# count of number of passenger green_df

print(green_df.passenger_count.value_counts().sort_index())
# Clean dates to standarize

def clean_dates(string):

    split = string.split(" ")

    date = split[0]

    hour = split[1]

    if hour.count(":") == 1:

        hour = hour + ":00"

    else:

        hour

    datetime = date + " " + hour

    return datetime
# Define a function to get a filename by chunks

def trips_chunks(file_name, chunk_size=1000000):

    final_df = pd.DataFrame()

    

    for chunk in pd.read_csv(

            file_name,

            chunksize=chunk_size,

            parse_dates=["pickup_datetime", "dropoff_datetime"],

            dtype={

                "pickup_longitude": np.float32,

                "pickup_latitude": np.float32,

                "dropoff_longitude": np.float32,

                "dropoff_latitude": np.float32,

                "passenger_count": np.int32,

                "trip_distance": np.float32,

                "total_amount": np.float32

            }):



        chunk["year"] = chunk.pickup_datetime.dt.year

        chunk["month"] = chunk.pickup_datetime.dt.month

        chunk["year_month"] = chunk.pickup_datetime.dt.to_period('M')

        chunk["day"] = chunk.pickup_datetime.dt.day

        chunk["weekday"] = chunk.pickup_datetime.dt.weekday

        chunk["pickup_hour"] = chunk.pickup_datetime.dt.hour

        chunk["pickup_time"] = chunk.pickup_datetime.dt.time

        chunk["dropoff_time"] = chunk.dropoff_datetime.dt.time

        chunk["difference_time"] = chunk.dropoff_datetime - chunk.pickup_datetime



        final_df = pd.concat([final_df, chunk], ignore_index=True)

        

        final_df = pd.concat([

            final_df[final_df.dropoff_datetime.between("2014-04-01 00:00:00",

                                           "2014-09-30 23:59:59")],

            final_df[final_df.dropoff_datetime.between("2015-01-01 00:00:00",

                                           "2015-06-30 23:59:59")]

        ])

        

        # Filters

        filter_distance = (final_df.trip_distance > 3)

        filter_ammount = (final_df.total_amount > 2.5)

        filter_passenger = ((final_df.passenger_count > 0) & (final_df.passenger_count <= 6))

        filter_latitude = ((final_df.pickup_latitude > 37) & (final_df.pickup_latitude < 45))

        filter_longitude = ((final_df.pickup_longitude > -78) & (final_df.pickup_longitude < -70))

        

        final_df = final_df[filter_distance & filter_ammount & filter_passenger & filter_latitude & filter_longitude]

        

    return final_df
def create_new_columns_uber(df):

    df["year"] = df.pickup_datetime.dt.year

    df["month"] = df.pickup_datetime.dt.month

    df["year_month"] = df.pickup_datetime.dt.to_period('M')

    df["day"] = df.pickup_datetime.dt.day

    df["weekday"] = df.pickup_datetime.dt.weekday

    df["pickup_hour"] = df.pickup_datetime.dt.hour

    df["pickup_time"] = df.pickup_datetime.dt.time

    return df
%%time

uber_2014 = pd.read_csv(f"{folder_path}/uber_trips_2014.csv",

                        dtype={"pickup_longitude": np.float32,

                               "pickup_latitude": np.float32,

                               "base": str})

uber_2014 = uber_2014.rename(columns={'base': 'affiliate_base'})

uber_2014["pickup_datetime"] = uber_2014.pickup_datetime.apply(clean_dates)

uber_2014["pickup_datetime"] = pd.to_datetime(uber_2014.pickup_datetime)

uber_2014 = create_new_columns_uber(uber_2014)
%%time

uber_2015 = pd.read_csv(f"{folder_path}/uber_trips_2015.csv",

                        

                        parse_dates=["pickup_datetime"],

                        dtype={"pickup_location_id": np.int32,

                               "dispatch_base": str,

                               "affiliate_base": str})

uber_2015 = create_new_columns_uber(uber_2015)
%%time

yellow_df = trips_chunks(f"{folder_path}/yellow_trips_new.csv")
yellow_df.describe()
yellow_df = yellow_df[((yellow_df.difference_time > "00:05:00") & (yellow_df.difference_time < "03:00:00"))]

yellow_df["taxi_type"] = "yellow"
# Filter the dataframe in datetime equals to uber

yellow_df = pd.concat([

    yellow_df[yellow_df.dropoff_datetime.between("2014-04-01 00:00:00",

                                                 "2014-09-30 23:59:59")],

    yellow_df[yellow_df.dropoff_datetime.between("2015-01-01 00:00:00",

                                                 "2015-06-30 23:59:59")]

])
green_df = trips_chunks(f"{folder_path}/green_trips_new_2.csv")
green_df.describe()
green_df = green_df[((green_df.difference_time > "00:05:00") & (green_df.difference_time < "03:00:00"))]

green_df["taxi_type"] = "green"
# Filter the dataframe in datetime equals to uber

green_df = pd.concat([

    green_df[green_df.dropoff_datetime.between("2014-04-01 00:00:00",

                                                "2014-09-30 23:59:59")],

    green_df[green_df.dropoff_datetime.between("2015-01-01 00:00:00",

                                               "2015-06-30 23:59:59")]

])
# Union green and yellow trips

trips_df = pd.concat([green_df, yellow_df], ignore_index=True)

trips_df = trips_df.drop("difference_time", axis=1)
trips_df.to_csv("taxi_trips.csv", encoding="utf-8", index=False, sep="|")
uber_2014.to_csv("uber_2014.csv", encoding='utf-8', index=False, sep="|")

uber_2015.to_csv("uber_2015.csv", encoding='utf-8', index=False, sep="|")
# Set your own project id here

PROJECT_ID = 'adl-innovation-project'

BUCKET_NAME = 'datathon_uber'

from google.cloud import bigquery

bigquery_client = bigquery.Client(project=PROJECT_ID)

from google.cloud import storage

storage_client = storage.Client(project=PROJECT_ID)

bucket = storage_client.get_bucket(BUCKET_NAME)
def upload_data_gcp(filename):

    blob = storage.blob.Blob(filename, bucket)

    blob.upload_from_filename(filename)
%%time

upload_data_gcp("taxi_trips.csv")

upload_data_gcp("uber_2014.csv")

upload_data_gcp("uber_2015.csv")
print(trips_df.shape)

print(yellow_df.shape)

print(green_df.shape)

print(uber_2014.shape)

print(uber_2015.shape)
from IPython.display import IFrame

IFrame('https://app.powerbi.com/view?r=eyJrIjoiNDA1NDU4MmEtOWJjOS00ZjgzLTllMTgtZTNjOTJiNWVkODQxIiwidCI6IjY0ZWM3NzE5LTgxZjYtNGNiZS05N2UxLTkzOWQyOWZlMzFhOSJ9',

       width=930,

       height=768)