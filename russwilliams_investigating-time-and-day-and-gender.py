import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns 
from mpl_toolkits.basemap import Basemap
%matplotlib inline
df_events = pd.read_csv("../input/events.csv", dtype={'device_id': np.str})

idx_beijing = (df_events["longitude"]>116) &\
              (df_events["longitude"]<117) &\
              (df_events["latitude"]>39.5) &\
              (df_events["latitude"]<40.5)
df_events_beijing = df_events[idx_beijing]

print("Total # events:", len(df_events))
print("Total # Beijing events:", len(df_events_beijing))


plt.figure(1, figsize=(12,6))
plt.title("Events by day")
plt.hist(df_events_beijing['timestamp'].map( lambda x: pd.to_datetime(x).dayofweek ), bins=7)
plt.show()
plt.figure(1, figsize=(12,6))
plt.title("Events by hour")
plt.hist(df_events_beijing['timestamp'].map( lambda x: pd.to_datetime(x).dayofweek*24 + pd.to_datetime(x).hour ), bins=168)
plt.show()
df_gender_age = pd.read_csv("../input/gender_age_train.csv", dtype={'device_id': np.str})
print("Total number of people in training set: ", len(df_gender_age))
df_joined = pd.merge(df_gender_age, df_events_beijing, on="device_id", how="inner")
print("Number of Beijing events in training set: ", len(df_joined))
df_female = df_joined[df_joined["gender"]=="F"]
df_male = df_joined[df_joined["gender"]=="M"]
print("Number of male events in Beijing: ", len(df_male))
print("Number of female events in Beijing: ", len(df_female))

plt.figure(1, figsize=(12,12))
plt.subplot(211)
plt.title("Female events by hour")
plt.hist(df_female['timestamp'].map( lambda x: pd.to_datetime(x).dayofweek*24 + pd.to_datetime(x).hour ), bins=168)
plt.subplot(212)
plt.title("Male events by hour")
plt.hist(df_male['timestamp'].map( lambda x: pd.to_datetime(x).dayofweek*24 + pd.to_datetime(x).hour ), bins=168)
plt.show()
df_under = df_joined[df_joined["age"]<30]
df_between = df_joined[(df_joined["age"]>=30) & (df_joined["age"]<40)]
df_over = df_joined[df_joined["age"]>=40]
print("Number of under-30s events in Beijing: ", len(df_under))
print("Number of 30-something events in Beijing: ", len(df_between))
print("Number of over-40s events in Beijing: ", len(df_over))

plt.figure(1, figsize=(12,18))
plt.subplot(311)
plt.title("Under-30s events by hour")
plt.hist(df_under['timestamp'].map( lambda x: pd.to_datetime(x).dayofweek*24 + pd.to_datetime(x).hour ), bins=168)
plt.subplot(312)
plt.title("30-something events by hour")
plt.hist(df_between['timestamp'].map( lambda x: pd.to_datetime(x).dayofweek*24 + pd.to_datetime(x).hour ), bins=168)
plt.subplot(313)
plt.title("Over-40s events by hour")
plt.hist(df_over['timestamp'].map( lambda x: pd.to_datetime(x).dayofweek*24 + pd.to_datetime(x).hour ), bins=168)
plt.show()
idx_friday_night = df_joined['timestamp'].map( lambda x: (pd.to_datetime(x).dayofweek==5) & (pd.to_datetime(x).hour < 6) )
df_friday_night = df_joined[idx_friday_night]
print("Number of Friday night events: ", len(df_friday_night))
print("Number of unique devices: ", df_friday_night["device_id"].nunique())

print("\nBeijing Night Owls:")
df_friday_night["group"].value_counts()
print("Total users: ", len(df_gender_age))
print("\nUser group counts:")
df_gender_age["group"].value_counts()