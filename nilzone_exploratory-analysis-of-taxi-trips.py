# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train.csv')

df.head()
print("Columns: {}".format(df.columns))

print("Shape: {}".format(df.shape))
df.info()
def create_date_columns(col, prefix, df=df):

    weekday_names = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"}

    df[col] = df[col].apply(pd.to_datetime)

    df[prefix + "hour"] = df[col].dt.hour

    df[prefix + "day"] = df[col].dt.day

    df[prefix + "weekday"] = df[col].dt.dayofweek.map(weekday_names)

    df[prefix + "weekday_index"] = df[col].dt.weekday

    df[prefix + "month"] = df[col].dt.month

    

    return df
df = create_date_columns("pickup_datetime", "pickup_")

df = create_date_columns("dropoff_datetime", "dropoff_")

df["trip_duration_min"] = df["trip_duration"] // 60

df.head()
fig, ax = plt.subplots(figsize=(10,4))

sns.countplot(x="passenger_count", data=df, hue="vendor_id", ax=ax)

plt.xlabel("Number of Passengers")
passengers = df.set_index("pickup_datetime").sort_index()

passengers["passenger_count"].resample("D").max().plot()

passengers["passenger_count"].resample("D").min().plot()

passengers["passenger_count"].resample("D").mean().plot()
plt.scatter(x=df["passenger_count"], y=df["trip_duration_min"])

plt.xlabel("Number of Passengers")

plt.ylabel("Total Duration in Minutes")
plt.hist(df["trip_duration_min"], range=[0, 50], bins=50)

plt.xlabel("Duration in Minutes")
weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

fig, ax = plt.subplots(figsize=(20, 8), ncols=2, nrows=2)

sns.countplot(x="pickup_hour", data=df, ax=ax[0][0])

sns.countplot(x="pickup_day", data=df, ax=ax[0][1])

sns.countplot(x="pickup_weekday", data=df, ax=ax[1][0], order=weekday_order)

sns.countplot(x="pickup_month", data=df, ax=ax[1][1])
fig, ax = plt.subplots(figsize=(18, 8))

sns.countplot(x="pickup_weekday", hue="pickup_hour", data=df, ax=ax)

plt.legend(loc=(1.03,0))
fig, ax = plt.subplots(figsize=(20, 5))

sns.countplot(x="pickup_hour", hue="passenger_count", data=df, ax=ax)

plt.legend(loc=(1.03,0))



fig, ax = plt.subplots(figsize=(20, 5))

sns.countplot(x="pickup_weekday", hue="passenger_count", data=df, ax=ax, order=weekday_order)

plt.legend(loc=(1.03,0))



fig, ax = plt.subplots(figsize=(20, 5))

sns.countplot(x="pickup_day", hue="passenger_count", data=df, ax=ax)

plt.legend(loc=(1.03,0))



fig, ax = plt.subplots(figsize=(20, 5))

sns.countplot(x="pickup_month", hue="passenger_count", data=df, ax=ax)

plt.legend(loc=(1.03,0))