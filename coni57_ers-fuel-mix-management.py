# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
from itertools import groupby
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression

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
team = pd.read_csv("/kaggle/input/f1-2020-race-data/ParticipantData_3335673977098133433.csv")
session = pd.read_csv("/kaggle/input/f1-2020-race-data/SessionData_3335673977098133433.csv").iloc[0].to_dict()
print(session)
for col, dtype in dtypes.items():
    if col in fillnas:
        df[col] = df[col].fillna(fillnas[col])
    df[col] = df[col].astype(dtype)
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
pilot_id = 1
lap = 15

temp = df[(df["pilot_index"] == pilot_id) & (df["currentLapNum"] == lap)]

c = [["b", "g", "r"][x] for x in temp["fuelMix"]]

plt.figure(figsize=(20, 12))
plt.scatter(temp["worldPositionZ"], temp["worldPositionX"], marker="o", s=1, c=c)
plt.axis('equal')
# plt.xlim(-500, -300)
# plt.ylim(-200, 0)
plt.show()
mixes = df.groupby(["pilot_index", "currentLapNum"]).apply(lambda x: x["fuelMix"].value_counts() / len(x)).reset_index()
mixes = mixes.pivot_table(index=["pilot_index", "currentLapNum"], values='fuelMix', columns='level_2')
mixes = mixes.fillna(0)
mixes.columns = ["Lean", "Normal", "Rich"]
mixes.head()
fig, axes = plt.subplots(5, 4, figsize=(30, 30))
for i in range(20):
    mixes[mixes.index.get_level_values(0) == i].reset_index().set_index("currentLapNum").drop("pilot_index", axis=1).plot(kind='bar', stacked=True, ax=axes[i//4][i%4])
    axes[i//4][i%4].set_title(team[team["pilot_index"] == i]["driverId"].values[0])
plt.show()
fuel = df.groupby(["pilot_index", "currentLapNum"]).agg({
    "fuelRemainingLaps" : ["min", "max", "mean"]
}).reset_index()
fuel.head()
fig, axes = plt.subplots(5, 4, figsize=(30, 30))
for i in range(20):
    temp = fuel[fuel["pilot_index"] == i]
    axes[i//4][i%4].plot(temp["currentLapNum"], temp[("fuelRemainingLaps", "max")], c="r")
    axes[i//4][i%4].plot(temp["currentLapNum"], temp[("fuelRemainingLaps", "mean")], c="b")
    axes[i//4][i%4].plot(temp["currentLapNum"], temp[("fuelRemainingLaps", "min")], c="g")
    axes[i//4][i%4].fill_between( temp["currentLapNum"], temp[("fuelRemainingLaps", "min")], temp[("fuelRemainingLaps", "max")], alpha = 0.3, color="y")
    axes[i//4][i%4].set_title(team[team["pilot_index"] == i]["driverId"].values[0])
plt.show()
def get_ratio_fuel(df):
    dist = df["lapDistance"].values

    ans = [0, 0, 0]
    for key, seq in groupby( df["fuelMix"].values):
        n = len(list(seq))
        ans[key] += (dist[n-1] - dist[0]) / session["trackLength"]
        dist = dist[n:]

    return ans

def get_fuel_diff(df):
    return df["fuelInTank"].max() - df["fuelInTank"].min()

def engine_provider(team):
    a = {
        "Ferrari": "Ferrari",
        "Mercedes": "Mercedes",
        "Renault": "Renault",
        "McLaren": "Renault",
        "Red Bull Racing": "Honda",
        "Toro Rosso": "Honda",
        "Racing Point": "Mercedes",
        "Williams": "Mercedes",
        "Alfa Romeo": "Ferrari",
        "Haas": "Ferrari"
    }
    return a[team]

mix = df.groupby(["pilot_index", "currentLapNum"]).apply(get_ratio_fuel).rename("percent_mix").reset_index()
mix["lean"] = mix["percent_mix"].apply(lambda x:x[0])
mix["normal"] = mix["percent_mix"].apply(lambda x:x[1])
mix["rich"] = mix["percent_mix"].apply(lambda x:x[2])

consumed_fuel = df.groupby(["pilot_index", "currentLapNum"]).apply(get_fuel_diff).rename("fuel_burned").reset_index()

fuel = pd.merge(mix, consumed_fuel, how="left", left_on=["pilot_index", "currentLapNum"], right_on = ["pilot_index", "currentLapNum"])
fuel = pd.merge(fuel, team, how="left", left_on=["pilot_index"], right_on = ["pilot_index"])
fuel["motor"] = fuel["teamId"].apply(engine_provider)
fuel.head()
def get_model(df):
    X = df[["lean", "normal", "rich"]].values
    y = df["fuel_burned"].values
    
    mdl = LinearRegression(fit_intercept=False)
    mdl.fit(X, y)
    
#     print(mdl.score(X, y))
    
    return mdl.coef_

ans = fuel.groupby("motor").apply(get_model)
print(ans)
a, b, c = get_model(fuel)
print(f"In Lean mix, the consumption is {a:.3f} kg/lap")
print(f"In Normal mix, the consumption is {b:.3f} kg/lap")
print(f"In Rich mix, the consumption is {c:.3f} kg/lap")
ans
filterered_fuel = fuel[fuel["lean"] == 0]
plt.scatter(filterered_fuel["normal"], filterered_fuel["fuel_burned"], alpha = 0.3)
plt.plot([0, 1], [c, b])
plt.show()
pilot_id = 19
lap = 2

temp = df[(df["pilot_index"] == pilot_id) & (df["currentLapNum"] == lap)]

c = [["b", "g", "r"][x] for x in temp["ersDeployMode"]]

plt.figure(figsize=(20, 12))
plt.scatter(temp["worldPositionZ"], temp["worldPositionX"], marker="o", s=1, c=c)
plt.axis('equal')
# plt.xlim(-500, -300)
# plt.ylim(-200, 0)
plt.show()
ers = df.groupby(["pilot_index", "currentLapNum"]).apply(lambda x: x["ersDeployMode"].value_counts() / len(x)).reset_index()
ers = ers.pivot_table(index=["pilot_index", "currentLapNum"], values='ersDeployMode', columns='level_2')
ers = ers.fillna(0)
ers.columns = ["Disable", "Normal", "OT mode"]
fig, axes = plt.subplots(5, 4, figsize=(30, 30))
for i in range(20):
    ers[ers.index.get_level_values(0) == i].reset_index().set_index("currentLapNum").drop("pilot_index", axis=1).plot(kind='bar', stacked=True, ax=axes[i//4][i%4])
    axes[i//4][i%4].set_title(team[team["pilot_index"] == i]["driverId"].values[0])
plt.show()
energy = df[df["ersStoreEnergy"] <= 4e6].groupby(["pilot_index", "currentLapNum"]).agg({
    "ersStoreEnergy" : ["min", "max", "mean"]
}).reset_index()

# some point have invalid values with 4e7 to be filtered
fig, axes = plt.subplots(5, 4, figsize=(30, 30))
for i in range(20):
    temp = energy[energy["pilot_index"] == i]
    axes[i//4][i%4].plot(temp["currentLapNum"], temp[("ersStoreEnergy", "max")], c="r")
    axes[i//4][i%4].plot(temp["currentLapNum"], temp[("ersStoreEnergy", "mean")], c="b")
    axes[i//4][i%4].plot(temp["currentLapNum"], temp[("ersStoreEnergy", "min")], c="g")
    axes[i//4][i%4].fill_between( temp["currentLapNum"], temp[("ersStoreEnergy", "min")], temp[("ersStoreEnergy", "max")], alpha = 0.3, color="y")
    axes[i//4][i%4].set_title(team[team["pilot_index"] == i]["driverId"].values[0])
plt.show()
def get_ratio_ers(df):
    dist = df["lapDistance"].values

    ans = [0, 0, 0]
    for key, seq in groupby( df["ersDeployMode"].values):
        n = len(list(seq))
        ans[key] += (dist[n-1] - dist[0]) / session["trackLength"]
        dist = dist[n:]

    return ans
ers_used = df.groupby(["pilot_index", "currentLapNum"]).agg({
    "ersDeployedThisLap" : "max"
}).reset_index()
ers = df.groupby(["pilot_index", "currentLapNum"]).apply(get_ratio_ers).rename("ers_mix").reset_index()
ers["Disable"] = ers["ers_mix"].apply(lambda x:x[0])
ers["Normal"] = ers["ers_mix"].apply(lambda x:x[1])
ers["Overtake"] = ers["ers_mix"].apply(lambda x:x[2])

ers = pd.merge(ers, ers_used, how="left", left_on=["pilot_index", "currentLapNum"], right_on = ["pilot_index", "currentLapNum"])
ers = pd.merge(ers, team, how="left", left_on=["pilot_index"], right_on = ["pilot_index"])
ers.head()
def get_model(df):
    X = df[["Disable", "Normal", "Overtake"]].values
    y = df["ersDeployedThisLap"].values
    
    mdl = LinearRegression(fit_intercept=False)
    mdl.fit(X, y)
    
    return mdl.coef_

ers.groupby(["pilot_index"]).apply(get_model)
a, b, c = get_model(ers)
print(f"In Disable mode, the consumption is {a:.3f} J/lap")
print(f"In Normal mode, the consumption is {b:.3f} J/lap")
print(f"In Overtake mode, the consumption is {c:.3f} J/lap")
filterered_ers = ers[ers["Disable"] == 0]  # normally nothing is removed
plt.scatter(filterered_ers["Normal"], filterered_ers["ersDeployedThisLap"], alpha = 0.3)
plt.plot([0, 1], [c, b])
plt.show()
ers_harvested = df[["pilot_index", "currentLapNum", 'ersHarvestedThisLapMGUK', 'ersHarvestedThisLapMGUH']].groupby(["pilot_index", "currentLapNum"]).max().reset_index()

# remove last lap as they are not always completed
ers_harvested = ers_harvested[ers_harvested["currentLapNum"] < 28]
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(30, 12))
sns.boxplot(x="currentLapNum", y='ersHarvestedThisLapMGUK', data=ers_harvested, ax=ax)
sns.boxplot(x="currentLapNum", y='ersHarvestedThisLapMGUH', data=ers_harvested, ax=ax2)
plt.show()
ers_harvested[['ersHarvestedThisLapMGUK', 'ersHarvestedThisLapMGUH']].mean()
ers_harvested = pd.merge(ers_harvested, team[["pilot_index", "driverId", "teamId"]], how="left", left_on=["pilot_index"], right_on = ["pilot_index"])
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(30, 12))
sns.boxplot(x="teamId", y='ersHarvestedThisLapMGUK', data=ers_harvested, ax=ax)
sns.boxplot(x="teamId", y='ersHarvestedThisLapMGUH', data=ers_harvested, ax=ax2)
plt.show()
print(f"The average energy harvested by the MGUK is {ers_harvested.median()['ersHarvestedThisLapMGUK']:.0f} J")
print(f"The average energy harvested by the MGUH is {ers_harvested.median()['ersHarvestedThisLapMGUH']:.0f} J")
