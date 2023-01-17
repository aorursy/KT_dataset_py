# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Import 

import pandas as pd

import argparse

import sys

import numpy as np

import os



from pandas.api.types import is_datetime64_any_dtype as is_datetime

from pandas.api.types import is_categorical_dtype



import matplotlib.pyplot as plt



from datetime import date, timedelta

path_data = "/kaggle/input/ashrae-energy-prediction"

path_train = path_data + os.sep + "train.csv"

path_test = path_data + os.sep + "test.csv"

path_building = path_data + os.sep + "building_metadata.csv"

path_weather_train = path_data + os.sep + "weather_train.csv"

path_weather_test = path_data + os.sep + "weather_test.csv"

path_weather_train_modified = "weather_train_modified.csv"

path_weather_test_modified = "weather_test_modified.csv"
# reduce mem usage

def reduce_mem_usage(df, use_float16=False):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.

    """

    start_mem = df.memory_usage().sum() / 1024 ** 2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))



    for col in df.columns:

        if is_datetime(df[col]) or is_categorical_dtype(df[col]):

            # skip datetime type or categorical type

            continue

        col_type = df[col].dtype



        if col_type != object:

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

                if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

        else:

            df[col] = df[col].astype('category')



    end_mem = df.memory_usage().sum() / 1024 ** 2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))



    return df
# Interpolate Full timestamp

def prep(df):

    s = df[df["site_id"] == 0]["timestamp"]

    s.index = df[df["site_id"] == 0]["timestamp"]

    full_time = s.reindex(pd.date_range(s.min(), s.max(), freq="H"))

    ret = pd.DataFrame()

    ret["timestamp"] = full_time.index

    # ret["site_id"] = 0

    ap = ret.copy()

    sites = range(16)

    for s in sites:

        tmp = ap.copy()

        tmp["site_id"] = s

        ret = ret.append(tmp)

    ret = ret[ret["site_id"].notna()]

    ret = ret.merge(df, on=["site_id", "timestamp"], how="left")

    return ret
# Fill Nan with the average of values within one week before and after.

def fill_mean(df, tar_col):

    ilimit = 7

    th = 4

    df['{}_nan_group'.format(tar_col)] = (df[tar_col].isna() & df[tar_col].shift(1).notna()).where(

        df[tar_col].isna()).cumsum()

    df['{}_nan_count'.format(tar_col)] = df['{}_nan_group'.format(tar_col)].map(

        df.groupby('{}_nan_group'.format(tar_col)).size())

    for k in df.groupby('{}_nan_group'.format(tar_col)).groups.keys():

        if len(df.iloc[df.groupby('{}_nan_group'.format(tar_col)).groups[k]]) < th:

            continue

        i = 1

        while df.iloc[df.groupby('{}_nan_group'.format(tar_col)).groups[k] - 24 * i][[tar_col]].isna().any().values[0]:

            i = i + 1

        b = df.iloc[df.groupby('{}_nan_group'.format(tar_col)).groups[k] - 24 * i][tar_col].values

        val = b

        i = 1

        if set(df.groupby('{}_nan_group'.format(tar_col)).groups[k] + 24 * i).issubset(df.index):

            while df.iloc[df.groupby('{}_nan_group'.format(tar_col)).groups[k] + 24 * i][[tar_col]].isna().any().values[0]:

                i = i + 1

                if i > ilimit | set(df.groupby('{}_nan_group'.format(tar_col)).groups[k] + 24 * i).issubset(df.index):

                    i = i - 1

                    break

            a = df.iloc[df.groupby('{}_nan_group'.format(tar_col)).groups[k] + 24 * i][tar_col].values

            val = (a + b) / 2

        df.loc[df.groupby('{}_nan_group'.format(tar_col)).groups[k], "{}_mean".format(tar_col)] = val

    del df['{}_nan_group'.format(tar_col)]

    del df['{}_nan_count'.format(tar_col)]

    return df
# Interpolate with average of the above average and Akima interpolation

def interpolate_col(df, tar_col):

    cols = []

    limit = 1000



    m = "mean"

    print("interpolate {} {}".format(tar_col, m))

    cols.append("{}_{}".format(tar_col, m))

    df = fill_mean(df, tar_col)



    m = "akima"

    print("interpolate {} {}".format(tar_col, m))

    cols.append("{}_{}".format(tar_col, m))

    df["{}_{}".format(tar_col, m)] = df[tar_col].fillna(

        df[tar_col].interpolate(method=m, limit_direction="both", limit=limit))



    df["{}_blend".format(tar_col)] = df[cols].mean(axis=1)



    return df, cols
# serch modified and original feacher col 

def search_weather_col(df, search_cols):

    ret = []

    cols = df.columns

    for col in search_cols:

        ret +=[s for s in cols if col in s]

    return ret
# plot modified and original feacher col 

def plot_res(df, tar_col = 'air_temperature', year = 2016, month = 1, site = 15, t = "train"):

#     out = "wi"

#     os.makedirs(out, exist_ok=True)

    cols = search_weather_col(df, [tar_col])

    cols.sort(reverse=True)

    s = date(year, month, 1)

    e = s + timedelta(days=30)

    tmp = df[

        (df["timestamp"] >= str(s)) &

        (df["timestamp"] < str(e)) &

        (df["site_id"] == site)

    ]

    # plt.plot(tmp.index, tmp[tar_col], label="original")

    for c in cols:

        plt.plot(tmp.index, tmp[c], label=c)

    plt.legend()

    title = "{}_y{}_m{}_site{}_{}".format(tar_col, year, month, site, t)

    plt.title(title)

#     plt.savefig(out + os.sep + "{}_y{}_m{}_site{}_{}.png".format(tar_col, year, month, site, t))

    plt.show()

    plt.close("all")
def interpolate_weather(df):

    df = prep(df)

    dcols = []

    df, cols = interpolate_col(df, 'air_temperature')

    dcols.extend(cols)

#     df, cols = interpolate_col(df, 'dew_temperature')

#     dcols.extend(cols)

#     df.loc[df["site_id"] == 5, "sea_level_pressure"] = df[df["site_id"] == 1]["sea_level_pressure"].values

#     df, cols = interpolate_col(df, 'sea_level_pressure')

#     dcols.extend(cols)

    return df, dcols
# Debug

def debug(weather_test):

    # Seems to work

    # - air_temperature_y2017_m5_site15_test

    # - air_temperature_y2017_m6_site15_test

    # - air_temperature_y2018_m4_site7_test

    # - air_temperature_y2018_m6_site15_test

    # - air_temperature_y2018_m8_site9_test

    # - air_temperature_y2018_m9_site1_test

    # - air_temperature_y2018_m9_site7_test

    print("Seems to work.")

    plot_res(weather_test, tar_col="air_temperature", year=2017, month=5, site=15, t="test")

    plot_res(weather_test, tar_col="air_temperature", year=2017, month=6, site=15, t="test")

    plot_res(weather_test, tar_col="air_temperature", year=2018, month=4, site=7, t="test")

    plot_res(weather_test, tar_col="air_temperature", year=2018, month=6, site=15, t="test")

    plot_res(weather_test, tar_col="air_temperature", year=2018, month=8, site=9, t="test")

    plot_res(weather_test, tar_col="air_temperature", year=2018, month=9, site=1, t="test")

    plot_res(weather_test, tar_col="air_temperature", year=2018, month=9, site=7, t="test")

    

    # Seems to not work

    # - air_temperature_y2017_m12_site7_test

    # - air_temperature_y2018_m11_site1_test

    # - air_temperature_y2018_m11_site5_test

    # - air_temperature_y2018_m11_site7_test

    # - air_temperature_y2018_m11_site12_test

    print("Seems to not work.---")

    plot_res(weather_test, tar_col="air_temperature", year=2017, month=12, site=7, t="test")

    plot_res(weather_test, tar_col="air_temperature", year=2018, month=11, site=1, t="test")

    plot_res(weather_test, tar_col="air_temperature", year=2018, month=11, site=5, t="test")

    plot_res(weather_test, tar_col="air_temperature", year=2018, month=11, site=7, t="test")

    plot_res(weather_test, tar_col="air_temperature", year=2018, month=11, site=12, t="test")

    

    # I can't say anything

    # - air_temperature_y2017_m3_site7_test

    # - air_temperature_y2018_m8_site7_test

    # - air_temperature_y2018_m9_site5_test

    # - air_temperature_y2018_m9_site11_test

    print("I can't say anything.")    

    plot_res(weather_test, tar_col="air_temperature", year=2017, month=3, site=7, t="test")

    plot_res(weather_test, tar_col="air_temperature", year=2018, month=8, site=7, t="test")

    plot_res(weather_test, tar_col="air_temperature", year=2018, month=9, site=5, t="test")

    plot_res(weather_test, tar_col="air_temperature", year=2018, month=9, site=11, t="test")

    
# main

weather_train = pd.read_csv(path_weather_train, parse_dates=['timestamp'])

weather_train = reduce_mem_usage(weather_train, use_float16=True)

weather_train, dcols = interpolate_weather(weather_train)

weather_train.drop(dcols, axis=1, inplace=True)

weather_train.to_csv(path_weather_train_modified, index=False)



weather_test = pd.read_csv(path_weather_test, parse_dates=['timestamp'])

weather_test = reduce_mem_usage(weather_test, use_float16=True)

weather_test, dcols = interpolate_weather(weather_test)

weather_test.drop(dcols, axis=1, inplace=True)

weather_test.to_csv(path_weather_test_modified, index=False)



print("Debug.")

debug(weather_test)