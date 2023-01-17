import numpy as np

import pandas as pd



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv("/kaggle/input/bikesfortutorial/Bikes.csv", sep=",", decimal=".")
df.head(7)
df.tail(10)
df.datetime
df["datetime"]
df[["datetime", "season"]]
df.iloc[0]
df.iloc[[0,3]]
df.loc[0, "datetime"]
df.loc[[1,3], ["datetime", "hum"]]
# https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior

pd.to_datetime(df.datetime, format="%Y-%m-%d %H:%M:%S")
df.datetime = pd.to_datetime(df.datetime, format="%Y-%m-%d %H:%M:%S")
df.datetime
df.datetime.dt.hour
df["hour"] = df.datetime.dt.hour
df.head()
df["weekday"] = df.datetime.dt.weekday

df["day"] = df.datetime.dt.day

df["month"] = df.datetime.dt.month

df["year"] = df.datetime.dt.year
df.head()