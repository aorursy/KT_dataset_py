# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
PLANT_1_DATA = pd.read_csv("../input/solar-power-generation-data/Plant_1_Generation_Data.csv")
PLANT_2_DATA = pd.read_csv("../input/solar-power-generation-data/Plant_2_Generation_Data.csv")
PLANT_1_DATA
# Data Cleaning
from datetime import datetime

PLANT_1_DATA["DATE_TIME"] = pd.to_datetime(PLANT_1_DATA["DATE_TIME"])
PLANT_1_SOURCE = PLANT_1_DATA.groupby("SOURCE_KEY")
SENSOR_DATA_MEAN = PLANT_1_SOURCE.mean()

from scipy import stats

def get_standard(df):
    data = pd.DataFrame()
    feature = df.columns
    for f in feature:
        data["{}_STD".format(f)] = (df[f]-df[f].mean())/df[f].std()
    return data

def outliers_detect(df):
    Z_CRITICAL = 1.96
    std_df = get_standard(df)
    outliers_l = []
    for f in std_df.columns:
        outliers = np.where(abs(std_df[f]) > Z_CRITICAL)[0]
        for o in outliers :
            if o not in outliers_l :
                outliers_l.append(o)
    return outliers_l

def remove_outliers(df):
    outliers = outliers_detect(SENSOR_DATA_MEAN)
    SENSOR_DATA_MEAN.reset_index(level=0, inplace=True)
    OUTLIER_SENSORS = [SENSOR_DATA_MEAN.loc[o, "SOURCE_KEY"] for o in outliers]
    DATA = df[~df["SOURCE_KEY"].isin(OUTLIER_SENSORS)]
    return DATA

PLANT_1_DATA = remove_outliers(PLANT_1_DATA)
# DAILY YIELD

def get_date(df):
    DATETIME_DATA = df.DATE_TIME
    df["DATE"] = DATETIME_DATA.dt.date
    df["TIME"] = DATETIME_DATA.dt.time
    return df

PLANT_1_DATA = get_date(PLANT_1_DATA)
PLANT_1_DATA_DAILY = PLANT_1_DATA.groupby("SOURCE_KEY")
PLANT_1_DATA_15052020["DATE_TIME"].subtract(POSIX).dt.total_seconds()
import matplotlib.pyplot as plt

POSIX = pd.to_datetime("19700101")
# lets graph yield in 1 day from 1 sensor
PLANT_1_DATA_15052020 = PLANT_1_DATA_DAILY.get_group("uHbuxQJl8lW7ozc").groupby("DATE").get_group("2020-05-15").reset_index(drop=True)
feature_column = ["DC_POWER", "AC_POWER", "DAILY_YIELD", "TOTAL_YIELD"]
for column in feature_column :
    plt.plot(PLANT_1_DATA_15052020["DATE_TIME"].subtract(POSIX).dt.total_seconds().astype("int"), PLANT_1_DATA_15052020[column])
    plt.xlabel("TIME")
    plt.ylabel(column)
    plt.show()
PLANT_1_DATA_DAILY.groups.keys()
group_l = list(PLANT_1_DATA_DAILY.groups.keys())
fig, ax = plt.subplots(nrows=3,ncols=1,figsize=(10,16))
for group in group_l:
    DATA = PLANT_1_DATA_DAILY.get_group(group).groupby("DATE").get_group("2020-05-15").reset_index(drop=True)
    ax[0].plot(DATA["DATE_TIME"].subtract(POSIX).dt.total_seconds().astype("int"), DATA["DC_POWER"], label=group)
    ax[1].plot(DATA["DATE_TIME"].subtract(POSIX).dt.total_seconds().astype("int"), DATA["AC_POWER"], label=group)
    ax[2].plot(DATA["DATE_TIME"].subtract(POSIX).dt.total_seconds().astype("int"), DATA["DAILY_YIELD"], label=group)
    
handles, labels = ax[2].get_legend_handles_labels()
fig.legend(handles, labels, loc='right')
