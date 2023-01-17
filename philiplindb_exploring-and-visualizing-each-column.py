import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

df = pd.read_csv("../input/metro-bike-share-trip-data.csv", low_memory=False)

format = "%Y-%m-%dT%H:%M:%S"
start_time = pd.to_datetime(df["Start Time"], format=format)
end_time = pd.to_datetime(df["End Time"], format=format)

duration = pd.DatetimeIndex(end_time - start_time)
duration = pd.DataFrame(duration.hour*60 + duration.minute)
print(duration.describe())
print(df.Duration.describe())
df.Duration = duration
plt.figure(figsize=(16, 8))
df.Duration.hist(bins=30, range=(0, 60))
plt.xlabel("Minutes")
plt.ylabel("Count")
for days, count in Counter(df["Plan Duration"].fillna("nan")).items():
    print("days = {}, count = {}, percentage = {}%".format(days, count, int(100*count/len(df))))
bikes = df["Bike ID"].dropna()
print(bikes.value_counts().describe())
plt.figure(figsize=(16, 8))
bikes.value_counts().plot(kind="hist", bins=35, range=(0,300))
plt.xlabel("Count of occurrences each bike was used")
for days, count in Counter(df["Passholder Type"]).items():
    print("type = {}, count = {}, percentage = {}%".format(days, count, int(100*count/len(df))))
for days, count in Counter(df["Trip Route Category"]).items():
    print("type = {}, count = {}, percentage = {}%".format(days, count, int(100*count/len(df))))
start_station = df["Starting Station ID"].dropna().value_counts()
end_station = df["Ending Station ID"].dropna().value_counts()

stations = pd.concat((start_station, end_station), axis=1, sort=False)
stations = stations.reset_index(drop=True)
print(len(stations))
print(stations.corr())
stations.plot(figsize=(16, 8))
plt.xlabel("Stations")
plt.ylabel("Count of each station")
plt.legend()
from math import sin, cos, sqrt, atan2, radians

lat1 = df["Starting Station Latitude"].apply(radians)
lon1 = df["Starting Station Longitude"].apply(radians)
lat2 = df["Ending Station Latitude"].apply(radians)
lon2 = df["Ending Station Longitude"].apply(radians)

dlon = lon2 - lon1
dlat = lat2 - lat1

R = 6373.0

a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

df["distance"] = R * c
df["distance"].describe()
plt.figure(figsize=(16, 8))
df.distance.hist(range=(0,4), bins=50)
plt.xlabel("km")
stations = df["Starting Station ID"].dropna().unique()

distance, std, count = list(), list(), list()
for s1 in stations:
    d_row, c_row, s_row = list(), list(), list()
    for s2 in stations:
        sdf = df[(df["Starting Station ID"] == s1) & (df["Ending Station ID"] == s2)]
        c_row.append(len(sdf))
        d = 0
        if len(sdf) > 0:
            s = np.std(sdf.distance)
            d = np.average(sdf.distance)
            if np.isnan(d): 
                d = 0
                s = 0
            if d > 5: d = 5
        d_row.append(d)
        s_row.append(s)
    distance.append(d_row)
    count.append(c_row)
    std.append(s_row)
    
distance = pd.DataFrame(data=distance, index=stations, columns=stations)
std = pd.DataFrame(data=std, index=stations, columns=stations)
count = pd.DataFrame(data=count, index=stations, columns=stations)
plt.figure(figsize=(16, 8))
sns.heatmap(count)
plt.title("Number of trips between stations")
plt.xlabel("Ending stations")
plt.ylabel("Starting stations")
plt.figure(figsize=(16, 8))
sns.heatmap(distance)
plt.title("Distance between stations")
plt.xlabel("Ending stations")
plt.ylabel("Starting stations")
plt.figure(figsize=(16, 8))
sns.heatmap(std)
plt.title("Std of the distance between stations")
plt.xlabel("Ending stations")
plt.ylabel("Starting stations")
rows = std[std > 1].sum(axis=0)
rows = rows[rows > 0]

cols = std[std > 1].sum(axis=1)
cols = cols[cols > 0]

ambiguous_stations = np.unique(list(rows.index) + list(cols.index))
print(ambiguous_stations)
df.loc[df["Trip Route Category"] == "Round Trip", "distance"].describe()
df.loc[df["Trip Route Category"] == "One Way", "distance"].describe()
df.loc[(df["Trip Route Category"] == "One Way") & (df.distance < 1e-5)].head()
year = df.loc[df["Passholder Type"] == "Flex Pass", "distance"]
month = df.loc[df["Passholder Type"] == "Monthly Pass", "distance"]
day = df.loc[df["Passholder Type"] == "Walk-up", "distance"]

data = [np.sum(year < 1e-5)/len(year), np.sum(month < 1e-5)/len(month), np.sum(day < 1e-5)/len(day)]
index = ["Flex Pass", "Monthly Pass", "Walk-up"]
zero_distance_trips = pd.DataFrame(data=data, index=index)

zero_distance_trips.plot.bar(figsize=(16, 8))
plt.figure(figsize=(16, 8))
year.plot.hist(bins=22, range=(0.1,4), alpha=0.2, density=True, label="Flex Pass")
month.plot.hist(bins=22, range=(0.1,4), alpha=0.2, density=True, label="Monthly Pass")
plt.legend()
time = pd.DatetimeIndex(start_time)
start_hour = pd.DataFrame(time.hour + time.minute/60)

fig, ax = plt.subplots(figsize=(16, 8))
for ph in ["Flex Pass", "Monthly Pass"]:
    d = start_hour[df["Passholder Type"] == ph]
    plt.hist(d.values, bins=100, range=(0,24),  density=True, label=ph, alpha=0.2)
    plt.xlabel("Hour")

plt.legend()

for ph in ["Walk-up"]:
    fig, ax = plt.subplots(figsize=(16, 8))
    d = start_hour[df["Passholder Type"] == ph]
    plt.hist(d.values, bins=100, range=(0,24),  density=True, label=ph, alpha=0.2)
    plt.title(ph)
    plt.xlabel("Hour")

plt.legend()
data = list()
for ph in ["Flex Pass", "Monthly Pass", "Walk-up"]:
    data.append(df.loc[df["Passholder Type"] == ph, "Starting Station ID"].dropna().value_counts())

stations = pd.concat(data, axis=1, sort=False)
station_names = list(stations.index)
stations = stations.reset_index(drop=True)
stations.columns = ["Flex Pass", "Monthly Pass", "Walk-up"]
stations = stations.apply(lambda s: s/s.sum())
stations.corr()
stations.plot(figsize=(16, 8))
plt.xlabel("Stations")
plt.ylabel("Count of each station")
plt.legend()
diff = stations["Monthly Pass"] - stations["Walk-up"]
diff = diff.apply(abs).nlargest(5).index.map(lambda s: station_names[s])
print(list(diff))