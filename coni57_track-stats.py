# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import matplotlib.pyplot as plt
import seaborn as sns

from itertools import groupby

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
dtypes = {
     'sessionTime' : "float32",
     'frameIdentifier' : "uint32",
     'pilot_index' : "uint8",
     'worldPositionX' : "float32",
     'worldPositionY' : "float32",
     'worldPositionZ' : "float32",
     'worldVelocityX' : "float32",
     'worldVelocityY' : "float32",
     'worldVelocityZ' : "float32",
     'worldForwardDirX' : "int32",
     'worldForwardDirY' : "int32",
     'worldForwardDirZ' : "int32",
     'worldRightDirX' : "int32",
     'worldRightDirY' : "int32",
     'worldRightDirZ' : "int32",
     'gForceLateral' : "float32",
     'gForceLongitudinal' : "float32",
     'gForceVertical' : "float32",
     'yaw' : "float32",
     'pitch' : "float32",
     'roll' : "float32",
     'speed' : "float32",
     'throttle' : "float32",
     'steer' : "float32",
     'brake' : "float32",
     'clutch': "uint8",
     'gear': "uint8",
     'engineRPM' : "uint32",
     'drs' : "bool",
     'engineTemperature': "uint8",
     'fuelMix': "uint8",
     'pitLimiterStatus': "bool",
     'fuelInTank' : "float32",
     'fuelRemainingLaps' : "float32",
     'ersStoreEnergy' : "uint32",
     'ersDeployMode' : "uint32",
     'ersHarvestedThisLapMGUK' : "uint32",
     'ersHarvestedThisLapMGUH' : "uint32",
     'ersDeployedThisLap' : "uint32",
     'carPosition' : "uint8",
     'currentLapTime' : "float32",
     'currentLapNum' : "uint8",
     'sector': "uint8",
     'lapDistance' : "float32",
     'totalDistance' : "float32",
}

fillnas = {
    'clutch' : 0,
    'gear' : 0,
    'engineRPM': 0,
    "engineTemperature" : 0,
    "fuelMix": 1,
    "pitLimiterStatus" : False,
    "ersStoreEnergy" : 4e7,
    "ersDeployMode" : 1,
    "ersHarvestedThisLapMGUK" : 0,
    "ersHarvestedThisLapMGUH" : 0,
    "ersDeployedThisLap" : 0,
    "sector" : 0
}

df = pd.read_csv("/kaggle/input/f1-2020-race-data/TelemetryData_3335673977098133433.csv")

for col, dtype in dtypes.items():
    if col in fillnas:
        df[col] = df[col].fillna(fillnas[col])
    df[col] = df[col].astype(dtype)
pilot = pd.read_csv("/kaggle/input/f1-2020-race-data/ParticipantData_3335673977098133433.csv")
session = pd.read_csv("/kaggle/input/f1-2020-race-data/SessionData_3335673977098133433.csv").iloc[0].to_dict()
print(session)
race = pd.read_csv("/kaggle/input/f1-2020-race-data/RaceTimeData_3335673977098133433.csv")
def remove_flashbacks(df, pilot=19):
    df2 = df[df["pilot_index"] == pilot]
    frame, X = df2["frameIdentifier"].values, df2[["worldPositionX", "worldPositionY", "worldPositionZ"]].values
    dist_sq = ((X[1:, :] - X[:-1, :])**2).sum(axis=1)
    idx_frame_after_flashback = np.argwhere(dist_sq > 1000).flatten() + 1 # to add the frame 0 shifted for the distance computation
    
    number_flashback = idx_frame_after_flashback.shape[0]
    pos_before_flashback = X[idx_frame_after_flashback-1]
    pos_after_flashback = X[idx_frame_after_flashback]  # position after validateing the flashback
    frames_before_flashback = frame[idx_frame_after_flashback-1]
    frames_after_flashback = frame[idx_frame_after_flashback] # first frame after validating the flashback
    
    for i in range(number_flashback):
        X_start = pos_after_flashback[i, :]
        frame_start = frames_after_flashback[i]
        idx_pos = idx_frame_after_flashback[i]
        d = ((X[idx_pos-500:idx_pos] - X_start)**2).sum(axis=1)
        start, stop = frame[idx_pos - 500 + np.argmin(d)], frame_start
        df = df[(df["frameIdentifier"] > stop) | (df["frameIdentifier"] <= start)]
        
    return df

df = remove_flashbacks(df, pilot=19)
subdf = df[(df["pilot_index"] == 19) & (df["currentLapNum"] == 2)][["worldPositionX", "worldPositionZ", "throttle", "brake", "lapDistance"]]
subdf.info()
c = [["r", "b"][x] for x in subdf["throttle"]>0.95]

plt.figure(figsize=(20, 12))
plt.scatter(subdf["worldPositionZ"], subdf["worldPositionX"], marker="o", s=1, c=c)
plt.axis('equal')
# plt.xlim(-500, -300)
# plt.ylim(-200, 0)
plt.show()
c = [["r", "b"][x] for x in subdf["brake"]>0.1]

plt.figure(figsize=(20, 12))
plt.scatter(subdf["worldPositionZ"], subdf["worldPositionX"], marker="o", s=1, c=c)
plt.axis('equal')
# plt.xlim(-500, -300)
# plt.ylim(-200, 0)
plt.show()
subdf["flat_out"] = subdf["throttle"]>0.95
subdf["braking_zone"] = subdf["brake"]>0.30
def get_distance(df, feature):
    dist = df["lapDistance"].values

    ans = 0
    for key, seq in groupby( df[feature].values):
        n = len(list(seq))
        if key:
            ans += dist[n-1] - dist[0] 
        dist = dist[n:]

    return ans
print(get_distance(subdf, "flat_out"))
print(get_distance(subdf, "braking_zone"))
df["flat_out"] = df["throttle"]>0.95
df["braking_zone"] = df["brake"]>0.30
flat = df.groupby(["pilot_index", "currentLapNum"]).apply(get_distance, feature="flat_out").reset_index()
brake = df.groupby(["pilot_index", "currentLapNum"]).apply(get_distance, feature="braking_zone").reset_index()
ans = pd.merge(flat, brake, how="left", left_on=["pilot_index", "currentLapNum"], right_on = ["pilot_index", "currentLapNum"])
ans.columns = ["pilot_index", "currentLapNum", "flat_out", "braking_zone"]
ans = pd.merge(ans, pilot, how="left", left_on=["pilot_index"], right_on = ["pilot_index"])
ans = ans[ans["currentLapNum"]<28]
ans.head()
ans.quantile(0.5)
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(30, 12))
sns.boxplot(x="teamId", y="flat_out", data=ans, ax=ax)
sns.boxplot(x="teamId", y="braking_zone", data=ans, ax=ax2)
plt.show()
def get_distance(df, feature):
    dist = df["lapDistance"].values

    ans = 0
    for key, seq in groupby( df[feature].values):
        n = len(list(seq))
        if key:
            ans += dist[n-1] - dist[0] 
        dist = dist[n:]

    return ans

df["above300"] = df["speed"]>300
df["below150"] = df["speed"]<150
above300 = df.groupby(["pilot_index", "currentLapNum"]).apply(get_distance, "above300").rename("above300").reset_index()
below150 = df.groupby(["pilot_index", "currentLapNum"]).apply(get_distance, "below150").rename("below150").reset_index()
ans = pd.merge(above300, below150, how="left", left_on=["pilot_index", "currentLapNum"], right_on = ["pilot_index", "currentLapNum"])
ans = pd.merge(ans, pilot, how="left", left_on=["pilot_index"], right_on = ["pilot_index"])
ans = ans[ans["currentLapNum"]<28]

ans[["above300", "below150"]].median()
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(30, 12))
sns.boxplot(x="teamId", y="above300", data=ans, ax=ax)
sns.boxplot(x="teamId", y="below150", data=ans, ax=ax2)
plt.show()
def get_number_gear_change(df):
    ans = 0
    for key, seq in groupby(df["gear"]):
        ans += 1
    return ans
gear = df.groupby(["pilot_index", "currentLapNum"]).apply(get_number_gear_change).rename("gear_change").reset_index()
gear = pd.merge(gear, pilot, how="left", left_on=["pilot_index"], right_on = ["pilot_index"])
gear = gear[gear["currentLapNum"]<28]
gear.head()
gear["gear_change"].median()
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(30, 12))
sns.boxplot(x="teamId", y="gear_change", data=gear, ax=ax)
sns.boxplot(x="currentLapNum", y="gear_change", data=gear, ax=ax2)
plt.show()
subdf = df[(df["pilot_index"] == 19) & (df["currentLapNum"] == 2)][["worldPositionX", "worldPositionZ", "gear", "speed"]]
plt.figure(figsize=(20, 12))
plt.scatter(subdf["worldPositionZ"], subdf["worldPositionX"], marker="o", s=1, c=subdf["speed"], cmap="cool")
plt.axis('equal')
# plt.xlim(-500, -300)
# plt.ylim(-200, 0)
plt.title("Speed on track")
plt.show()
fig, ax = plt.subplots(1, figsize=(30, 12))
sns.scatterplot(x="currentLapNum", y="LapTime", hue="pilot_index", data=race, ax=ax)
ax.hlines(race["LapTime"].median(), 0, 30)
plt.show()
fastest_lap = race[race["currentLapNum"]<28]["LapTime"].min()
fastest_avg_speed = session["trackLength"] / fastest_lap * 3.6
print(f"Fastest Average Speed : {fastest_avg_speed:.2f}km/h")
speed = session["trackLength"] / race["LapTime"].median() * 3.6
print(f"Average Speed : {speed:.2f}km/h")
def get_max_speed(df):
    idx = df["speed"].argmax()
    return df.iloc[idx][["drs", "speed"]]

speed = df.groupby(["pilot_index", "currentLapNum"]).apply(get_max_speed).reset_index()
speed = speed[speed["currentLapNum"]<28]
speed = pd.merge(speed, pilot, how="left", left_on=["pilot_index"], right_on = ["pilot_index"])
plt.figure(figsize=(20, 12))
sns.boxplot(x="teamId", y="speed", hue="drs", data=speed)
plt.show()
subdf = df[(df["pilot_index"] == 19) & (df["currentLapNum"] == 2)][["worldPositionX", "worldPositionZ", 'gForceLateral', 'gForceLongitudinal', 'gForceVertical', "lapDistance"]]
plt.plot(subdf["lapDistance"], subdf["gForceLateral"])
plt.show()
subdf = df[(df["pilot_index"] == 19)][["worldPositionX", "worldPositionZ", 'gForceLateral', 'gForceLongitudinal', 'gForceVertical', "lapDistance", "currentLapNum"]]

plt.figure(figsize=(20, 12))
for i in range(20):
    plt.plot(subdf[subdf["currentLapNum"] == i]["lapDistance"], subdf[subdf["currentLapNum"] == i]["gForceLateral"])
plt.show()
plt.figure(figsize=(20, 12))
for i in range(20):
    plt.plot(subdf[subdf["currentLapNum"] == i]["lapDistance"], subdf[subdf["currentLapNum"] == i]["gForceLongitudinal"])
plt.ylim(-5, 3)
plt.show()
def get_max_g_lat(df):
    return df["gForceLateral"].max()

acc = df.groupby(["pilot_index", "currentLapNum"]).apply(get_max_g_lat).rename("max_lat_acc").reset_index()
acc = acc[acc["currentLapNum"]<28]
acc = pd.merge(acc, pilot, how="left", left_on=["pilot_index"], right_on = ["pilot_index"])
acc.head()
fig, ax = plt.subplots(figsize=(20, 12))
sns.boxplot(x="driverId", y="max_lat_acc", data=acc, ax=ax)
ax.set_xticklabels(ax.get_xticklabels(),rotation=30)
plt.show()
fig, ax = plt.subplots(figsize=(20, 12))
sns.boxplot(x="driverId", y="max_lat_acc", data=acc[acc["teamId"] == "Mercedes"], ax=ax)
ax.set_xticklabels(ax.get_xticklabels(),rotation=30)
plt.show()
