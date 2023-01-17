import pandas as pd

import numpy as np

import multiprocessing as mp

import geopy.distance

import csv

from datetime import datetime, timezone

from tqdm import tqdm



tqdm.pandas()

# %load_ext nb_black
taxi = pd.read_csv(

    "../input/train.csv",

    sep=",",

    low_memory=True,

#             skiprows=lambda i: i % 10 != 0,  # Use only 1 of each n

)
taxi["YEAR"] = taxi.TIMESTAMP.apply(lambda x: datetime.fromtimestamp(x).year)

taxi["MONTH"] = taxi.TIMESTAMP.apply(lambda x: datetime.fromtimestamp(x).month)

taxi["DAY"] = taxi.TIMESTAMP.apply(lambda x: datetime.fromtimestamp(x).day)

taxi["HOUR"] = taxi.TIMESTAMP.apply(lambda x: datetime.fromtimestamp(x).hour)

taxi["WEEKDAY"] = taxi.TIMESTAMP.apply(

    lambda x: datetime.fromtimestamp(x).weekday()

)
taxi.TRIP_ID.count()
taxi = taxi.drop_duplicates()
taxi.TRIP_ID.count()
taxi = taxi[taxi.MISSING_DATA == False]
taxi.TRIP_ID.count()
taxi = taxi[taxi.POLYLINE != "[]"]
taxi.TRIP_ID.count()
taxi.reset_index(drop=True, inplace=True)
def map_trip_start(df_in):

    df_out = pd.DataFrame()

    df_out["TRIP_START"] = df_in["POLYLINE"].progress_map(lambda x: repr(eval(x)[0]))

    return df_out
p = mp.Pool(processes=mp.cpu_count())

pool_results = p.map(map_trip_start, np.array_split(taxi, mp.cpu_count()))

p.close()

p.join()



# merging parts processed by different processes

new_columns = pd.concat(pool_results, axis=0, ignore_index=True)



# merging newly calculated columns to taxi

taxi = pd.concat([taxi, new_columns], axis=1)
def process_polyline(p):

    p = eval(p)

    if len(p) > 1:

        trip_distance = 0

        top_speed = 0

        for i in range(len(p) - 1):

            distance = geopy.distance.distance(p[i], p[i + 1]).km

            trip_distance += distance

            speed = distance / 15 * 3600

            if speed > top_speed:

                top_speed = speed

        trip_time = (len(p) - 1) * 15 / 60

        avg_speed = trip_distance / trip_time * 60

        return trip_distance, trip_time, avg_speed, top_speed

    else:

        return np.NaN, np.NaN, np.NaN, np.NaN
print(

    "Trip distance: {:>5.1f} km\n"

    "Trip time:     {:>5.1f} min\n"

    "Average speed: {:>5.1f} km/h\n"

    "Top speed:     {:>5.1f} km/h".format(*process_polyline(taxi.POLYLINE[0]))

)
def map_polyline(df_in):

    df_out = pd.DataFrame()

    df_out["TRIP_DISTANCE"], df_out["TRIP_TIME"], df_out[

        "AVERAGE_SPEED"

    ], df_out["TOP_SPEED"] = zip(

        *df_in["POLYLINE"].progress_map(process_polyline)

    )

    return df_out
p = mp.Pool(processes=mp.cpu_count())

pool_results = p.map(map_polyline, np.array_split(taxi, mp.cpu_count()))

p.close()

p.join()



# merging parts processed by different processes

new_columns = pd.concat(pool_results, axis=0, ignore_index=True)



# merging newly calculated columns to taxi

taxi = pd.concat([taxi, new_columns], axis=1)
taxi.to_csv("train_extended.csv.zip", index=None, compression="zip")