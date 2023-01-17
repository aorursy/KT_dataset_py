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
participants = pd.read_csv("/kaggle/input/f1-2020-race-data/ParticipantData_3335673977098133433.csv")

display(participants)
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
plt.figure(figsize=(20, 12))

plt.plot(df[df["pilot_index"] == 19]["worldPositionZ"], df[df["pilot_index"] == 19]["worldPositionX"])

plt.axis('equal')

plt.show()
df = remove_flashbacks(df, pilot=19)
c = [["r", "b"][x] for x in df[df["pilot_index"] == 19]["pitStatus"].isnull().values]



plt.figure(figsize=(20, 12))

plt.scatter(df[df["pilot_index"] == 19]["worldPositionZ"], df[df["pilot_index"] == 19]["worldPositionX"], marker="o", s=1, c=c)

plt.axis('equal')

# plt.xlim(-500, -300)

# plt.ylim(-200, 0)

plt.show()
df["pitStatus"].value_counts()
def get_pit_durations(df):

    sessionTime = df["sessionTime"].values

    pitStatus = df["pitStatus"].fillna("on track").values



    pit_stop_duration = []

    pitting_duration = []



    current_index = 0

    was_on_pit = False

    for val, elems in groupby(pitStatus):

        nrows = len(list(elems))

        if val == "pitting" and not was_on_pit:

            start_pit_stand = sessionTime[0]

            was_on_pit = True

        elif val == "on track" and was_on_pit:  # in case on penalty of drive thru, we may not have a "in pit area" status

            stop_pit_stand = sessionTime[0]

            pitting_duration.append(stop_pit_stand - start_pit_stand)

            was_on_pit = False

        elif val == "in pit area":

            start_pit_stop = sessionTime[0]

            stop_pit_stop = sessionTime[nrows]

            pit_stop_duration.append(stop_pit_stop - start_pit_stop)

        sessionTime=sessionTime[nrows:]

    

    return pit_stop_duration, pitting_duration
all_pilots = df["pilot_index"].unique()

pit_areas, pit_lanes = [], []

for pilot_id in all_pilots:

    light_df = df[df["pilot_index"] == pilot_id][["sessionTime", "pilot_index", "pitStatus"]]

    pit_area, pit_lane = get_pit_durations(light_df)

    pit_areas += pit_area

    pit_lanes += pit_lane
plt.figure(figsize=(20, 12))

plt.boxplot([pit_areas, pit_lanes])

plt.xticks([1, 2], ['Time in pit area', 'Time in pit lane'])

plt.title("Time in pit")

plt.show()
import statistics



print(f"Median time in pit area : {statistics.median(pit_areas):.3f}s")

print(f"Median time in pit lane : {statistics.median(pit_lanes):.3f}s")



print(f"Fastest pit stop : {min(pit_areas):.3f}s")
df['in_pit'] = df["pitStatus"].notnull()
all_pilots = df["pilot_index"].unique()

dists_entry = []

dists_exit = []

for pilot_id in all_pilots:

    sub_df = df[df["pilot_index"] == pilot_id].copy()

    in_pit, lap_distance, lap_time = sub_df['in_pit'].values, sub_df['lapDistance'].values, sub_df['currentLapTime'].values

    for pit_bool, seq in groupby(in_pit):

        n_frames = len(list(seq))

        if pit_bool:

            if lap_distance[0] > 2000 and 300 < lap_distance[n_frames-1] < 1000: # entry is always at the end of a lap and a lap is more than 3km

                dists_entry.append(lap_distance[0])

                dists_exit.append(lap_distance[n_frames-1])

        lap_distance = lap_distance[n_frames:]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12))

ax1.boxplot(dists_entry)

ax2.boxplot(dists_exit)

ax1.set_title('Distance before entry')

ax2.set_title('Distance for the exit')

plt.show()
lap_distance = pd.read_csv("/kaggle/input/f1-2020-race-data/SessionData_3335673977098133433.csv").iloc[0]["trackLength"]

pit_distance = lap_distance - statistics.median(dists_entry) + statistics.median(dists_exit)

print(f"The pit distance is {round(pit_distance, 2)}m")
df_on_track = df.groupby(["pilot_index", "currentLapNum"]).filter(lambda x:x["in_pit"].sum()==0)
sub_df = df_on_track[["pilot_index", "currentLapNum", "lapDistance", "currentLapTime"]]
sub_df.head()
lap_time = pd.read_csv("/kaggle/input/f1-2020-race-data/RaceTimeData_3335673977098133433.csv")
def get_time_eq_entry_pit(df):

    x = df["lapDistance"].values

    t = df["currentLapTime"].values

    dx = x - statistics.median(dists_entry)

    idx_min = np.argmin(np.abs(dx))

    if dx[idx_min] < 10:  # when I finish the lap, AI behind are directly stopped so we may not find the real min

        return t[idx_min]

    else:

        return None

    

def get_time_eq_exit_pit(df):

    x = df["lapDistance"].values

    t = df["currentLapTime"].values

    dx = x - statistics.median(dists_exit)

    idx_min = np.argmin(np.abs(dx))

    if dx[idx_min] < 10:  # when I finish the lap, AI behind are directly stopped so we may not find the real min

        return t[idx_min]

    else:

        return None
entry = sub_df.groupby(["pilot_index", "currentLapNum"]).apply(get_time_eq_entry_pit).rename("entryTime").reset_index()

exit = sub_df.groupby(["pilot_index", "currentLapNum"]).apply(get_time_eq_exit_pit).rename("exitTime").reset_index()
time_info = pd.merge(entry, lap_time,  how='left', left_on=["pilot_index", "currentLapNum"], right_on = ["pilot_index", "currentLapNum"])

time_info = pd.merge(time_info, exit,  how='left', left_on=["pilot_index", "currentLapNum"], right_on = ["pilot_index", "currentLapNum"])
time_info["pit_duration_in_track"] = time_info["LapTime"] - time_info["entryTime"] + time_info["exitTime"].shift(1)  # shift of 1 car we enter in pit in lap i and exit it in lap i+1
time_info["pit_duration_in_track"].plot(kind="box")

plt.show()
t = time_info["pit_duration_in_track"].values

t = t[t>4]

print(f"Out of the lap, the time to go the pit lane is { round(t.mean()) }s")