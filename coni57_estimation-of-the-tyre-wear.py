# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
from itertools import groupby
import matplotlib.pyplot as plt

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
for col, dtype in dtypes.items():
    if col in fillnas:
        df[col] = df[col].fillna(fillnas[col])
    df[col] = df[col].astype(dtype)
df.head()
df.info()
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
df_tyre = df[[
    "frameIdentifier",
    "pilot_index",
    "currentLapTime",
    "currentLapNum",
    "tyresSurfaceTemperature",
    "tyresInnerTemperature",
    "tyresPressure",
    "fuelInTank",
    "tyresWear",
    "actualTyreCompound",
    "tyresDamage"
]]
df_tyre.head()
wear = df_tyre["tyresDamage"].str.split("/", expand=True).astype("float32")
wear.columns = ["tyresDamage_FL", "tyresDamage_FR", "tyresDamage_RL", "tyresDamage_RR"]

surface_temp = df_tyre["tyresSurfaceTemperature"].str.split("/", expand=True).astype("float32")
surface_temp.columns = ["tyresSurfaceTemperature_FL", "tyresSurfaceTemperature_FR", "tyresSurfaceTemperature_RL", "tyresSurfaceTemperature_RR"]

inner_temp = df_tyre["tyresInnerTemperature"].str.split("/", expand=True).astype("float32")
inner_temp.columns = ["tyresInnerTemperature_FL", "tyresInnerTemperature_FR", "tyresInnerTemperature_RL", "tyresInnerTemperature_RR"]

pressure = df_tyre["tyresPressure"].str.split("/", expand=True).astype("float32")
pressure.columns = ["tyresPressure_FL", "tyresPressure_FR", "tyresPressure_RL", "tyresPressure_RR"]

df_tyre = pd.concat([df_tyre[["frameIdentifier", "pilot_index", "currentLapTime", "fuelInTank", "actualTyreCompound", "currentLapNum"]], wear, surface_temp, inner_temp, pressure], axis=1)
light_df = df_tyre[df_tyre["pilot_index"] == 19]
summary = light_df.groupby('currentLapNum').tail(1)

plt.plot(summary['currentLapNum'], summary["tyresDamage_FR"])
plt.plot(summary['currentLapNum'], summary["tyresDamage_FL"])
plt.plot(summary['currentLapNum'], summary["tyresDamage_RR"])
plt.plot(summary['currentLapNum'], summary["tyresDamage_RL"])
plt.show()
wear = summary[["tyresDamage_FL", "tyresDamage_FR", "tyresDamage_RL", "tyresDamage_RR"]].mean(axis=1).to_list()
tyre = summary["actualTyreCompound"].to_list()
wears = []
buffer = [wear[0]]
for c, prev, curr in zip(tyre[:-1], wear[:-1], wear[1:]):
    if prev > curr:
        wears.append((c, buffer))
        buffer = [curr]
    else:
        buffer.append(curr)
wears.append((c, buffer))
for c, w in wears:
    plt.plot(range(1, len(w)+1), w, label=c)
    plt.plot(range(1, len(w)+1), w, label=c)
plt.show()
result = {
    "soft" : [],
    "medium" : [],
    "hard" : [],
}

for c, w in wears:
    model = LinearRegression(fit_intercept=False)  # at lap 0, wear is 0 (no califications)
    X = np.arange(1, len(w)+1).reshape(-1, 1)
    y = np.array(w).reshape(-1, 1)
    model.fit(X, y)
    result[c].append(model.coef_[0][0])

result
def get_wear_factor(df):
    summary = df.groupby('currentLapNum').tail(1)
    summary = summary[summary["currentLapTime"] > 50]
    
    wear = summary[["tyresDamage_FL", "tyresDamage_FR", "tyresDamage_RL", "tyresDamage_RR"]].mean(axis=1).to_list()
    tyre = summary["actualTyreCompound"].to_list()
    
    wears = []
    buffer = [wear[0]]
    for c, prev, curr in zip(tyre[:-1], wear[:-1], wear[1:]):
        if prev > curr:
            wears.append((c, buffer))
            buffer = [curr]
        else:
            buffer.append(curr)
    wears.append((c, buffer))
    
    result = {
        "soft" : [],
        "medium" : [],
        "hard" : [],
    }

    for c, w in wears:
        model = LinearRegression(fit_intercept=False)  # at lap 0, wear is 0 (no califications)
        X = np.arange(1, len(w)+1).reshape(-1, 1)
        y = np.array(w).reshape(-1, 1)
        model.fit(X, y)
        result[c].append(model.coef_[0][0])
    
    return result
all_pilots = df["pilot_index"].unique()
wear_per_driver = []

for pilot_id in all_pilots:
    light_df = df_tyre[df_tyre["pilot_index"] == pilot_id]
    wear_per_driver.append(get_wear_factor(light_df))

wear_per_driver
agg_result = {
    "soft" : [],
    "medium" : [],
    "hard" : [],
}

for result in wear_per_driver:
    for key, vals in result.items():
        agg_result[key] += vals
plt.figure(figsize=(20, 12))
plt.boxplot([agg_result["soft"], agg_result["medium"], agg_result["hard"]])
plt.xticks([1, 2, 3], ["soft", "medium", "hard"])
plt.ylabel("Wear per lap in %")
plt.title("Tyre Wear")
plt.show()
