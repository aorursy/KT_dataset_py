import numpy as np
import pandas as pd
from math import radians, sin, cos, acos
from datetime import datetime
import os

print(os.listdir("../input"))

df_train = pd.read_csv('../input/train.csv')
df_sample = df_train.sample(n=3000)
def make_date_time(df):
    year = []
    month = []
    day = []
    hour = []
    minute = []
    second = []
    for dt in df['pickup_datetime']:
        dt = dt[:-4]
        df_ = datetime.strptime(dt, '%Y-%m-%d %H:%M:%S')
        year.append(df_.year)
        month.append(df_.month)
        day.append(df_.day)
        hour.append(df_.hour)
        minute.append(df_.minute)
        second.append(df_.second)
    df['year'] = year
    df['month'] = month
    df['day'] = day
    df['hour'] = hour
    df['minute'] = minute
    df['second'] = second
    return df

def remove_same_place(df):
    drop_rows = []
    for i in range(df.shape[0]):
       if df.iloc[i]['pickup_latitude'] == df.iloc[i]['dropoff_latitude'] and df.iloc[i]['pickup_longitude'] == df.iloc[i]['dropoff_longitude']:
        drop_rows.append(i)
    df.drop(df.index[drop_rows],inplace=True)
    return df
def calculate_displacement(df):
    ls = []
    for i in range(df.shape[0]):
        x1 = df.iloc[i]['pickup_latitude']
        x2 = df.iloc[i]['dropoff_latitude']
        y1 = df.iloc[i]['pickup_longitude']
        y2 = df.iloc[i]['dropoff_longitude']
        slat = radians(y1)
        slon = radians(y2)
        elat = radians(x1)
        elon = radians(x2)
        dist = 6371.01 * acos(sin(slat)*sin(elat) + cos(slat)*cos(elat)*cos(slon - elon))
        ls.append(dist)
    df['displacement'] = ls
    return df
def calculate_rate(df):
    df['rate'] = df['fare_amount']*df['displacement']/df['passenger_count']
    return df
def hours_into_min(df):
    df['hour'] = df['hour'].apply(lambda x: x*60)
    whole_min = df['hour'] + df['minute']
    df['time'] = whole_min
    return df
df_sample = make_date_time(df_sample)
df_sample = remove_same_place(df_sample)
df_sample = calculate_displacement(df_sample)
df_sample = calculate_rate(df_sample)
df_sample = hours_into_min(df_sample)
df_sample.head()
def divide_by_month(df):
    month_dict = {}
    for i in range(1,13):
        month_dict[i] = df[df['month']==i]
    return month_dict