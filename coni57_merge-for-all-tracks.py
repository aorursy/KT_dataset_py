# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import math
import re
from glob import iglob
from itertools import groupby
import statistics

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LinearRegression

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import matplotlib.pyplot as plt
import seaborn as sns
# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
"""
Preprocessing
"""

def load_data(ID):
    dtypes = {
     'sessionTime' : "float32",
     'frameIdentifier' : "uint32",
     'pilot_index' : "uint8",
     'worldPositionX' : "float32",
     'worldPositionY' : "float32",
     'worldPositionZ' : "float32",
     'gForceLateral' : "float32",
     'gForceLongitudinal' : "float32",
     'speed' : "float32",
     'throttle' : "float32",
     'brake' : "float32",
     'gear': "uint8",
     'drs' : "bool",
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
        'gear' : 0,
        "fuelMix": 1,
        "pitLimiterStatus" : False,
        "ersStoreEnergy" : 4e6,
        "ersDeployMode" : 1,
        "ersHarvestedThisLapMGUK" : 0,
        "ersHarvestedThisLapMGUH" : 0,
        "ersDeployedThisLap" : 0,
        "sector" : 0
    }
    
    telemetry = pd.read_csv(f"/kaggle/input/f1-2020-race-data/TelemetryData_{ID}.csv", usecols=[
        'sessionTime',
        'frameIdentifier',
        'pilot_index',
        'worldPositionX',
        'worldPositionY',
        'worldPositionZ',
        'gForceLateral',
        'gForceLongitudinal',
        'speed',
        'throttle',
        'brake',
        'gear',
        'drs',
        'fuelMix',
        'pitLimiterStatus',
        'fuelInTank',
        'fuelRemainingLaps',
        'ersStoreEnergy',
        'ersDeployMode',
        'ersHarvestedThisLapMGUK',
        'ersHarvestedThisLapMGUH',
        'ersDeployedThisLap',
        'carPosition',
        'currentLapTime',
        'currentLapNum',
        'sector',
        'lapDistance',
        'totalDistance',
        'tyresWear',
        'actualTyreCompound',
        'tyresDamage',
        'pitStatus'
    ], dtype={
        "pitStatus": str
    })
    session = pd.read_csv(f"/kaggle/input/f1-2020-race-data/SessionData_{ID}.csv").iloc[0].to_dict()
    participant = pd.read_csv(f"/kaggle/input/f1-2020-race-data/ParticipantData_{ID}.csv")
    race = pd.read_csv(f"/kaggle/input/f1-2020-race-data/RaceTimeData_{ID}.csv")
    
    for col, dtype in dtypes.items():
        if col in fillnas:
            telemetry[col] = telemetry[col].fillna(fillnas[col])
        telemetry[col] = telemetry[col].astype(dtype)
    
    return telemetry, session, participant, race

def get_my_id(df):
    return df[df["aiControlled"] == 0].iloc[0]["pilot_index"]

"""
Remove flashback
"""

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
        offset = min(500, idx_pos)
        d = ((X[idx_pos-offset:idx_pos] - X_start)**2).sum(axis=1)
        start, stop = frame[idx_pos - offset + np.argmin(d)], frame_start
        df = df[(df["frameIdentifier"] > stop) | (df["frameIdentifier"] <= start)]

    return df

"""
Fuel Ratio
"""

def get_ratio_fuel(df, track_length):
    dist = df["lapDistance"].values

    ans = [0, 0, 0]
    for key, seq in groupby( df["fuelMix"].values):
        n = len(list(seq))
        ans[key] += (dist[n-1] - dist[0]) / track_length
        dist = dist[n:]

    return ans

def get_fuel_diff(df):
    return df["fuelInTank"].max() - df["fuelInTank"].min()

def engine_provider(team):
    return{
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
    }[team]

def train_fuel_model(df):
    X = df[["lean", "normal", "rich"]].values
    y = df["fuel_burned"].values
    
    mdl = LinearRegression(fit_intercept=False)
    mdl.fit(X, y)

    return mdl.coef_

def get_fuel_estimation(df, track_length):
    mix = df.groupby(["pilot_index", "currentLapNum"]).apply(get_ratio_fuel, track_length).rename("percent_mix").reset_index()
    mix["lean"] = mix["percent_mix"].apply(lambda x:x[0])
    mix["normal"] = mix["percent_mix"].apply(lambda x:x[1])
    mix["rich"] = mix["percent_mix"].apply(lambda x:x[2])

    consumed_fuel = df.groupby(["pilot_index", "currentLapNum"]).apply(get_fuel_diff).rename("fuel_burned").reset_index()

    fuel = pd.merge(mix, consumed_fuel, how="left", left_on=["pilot_index", "currentLapNum"], right_on = ["pilot_index", "currentLapNum"])
    a, b, c = train_fuel_model(fuel)
    
    return {
        "fuel_consumption_lean" : a,
        "fuel_consumption_normal" : b,
        "fuel_consumption_rich" : c,
    }

"""
ERS factors
"""

def get_ratio_ers(df, track_length):
    dist = df["lapDistance"].values

    ans = [0, 0, 0]
    for key, seq in groupby( df["ersDeployMode"].values):
        n = len(list(seq))
        ans[key] += (dist[n-1] - dist[0]) / track_length
        dist = dist[n:]

    return ans

def train_ers_model(df):
    X = df[["Disable", "Normal", "Overtake"]].values
    y = df["ersDeployedThisLap"].values
    
    mdl = LinearRegression(fit_intercept=False)
    mdl.fit(X, y)
    
    return mdl.coef_

def get_ERS_estimation(df, track_length):
    ers_used = df.groupby(["pilot_index", "currentLapNum"]).agg({
        "ersDeployedThisLap" : "max"
    }).reset_index()

    ers = df.groupby(["pilot_index", "currentLapNum"]).apply(get_ratio_ers, track_length).rename("ers_mix").reset_index()
    ers["Disable"] = ers["ers_mix"].apply(lambda x:x[0])
    ers["Normal"] = ers["ers_mix"].apply(lambda x:x[1])
    ers["Overtake"] = ers["ers_mix"].apply(lambda x:x[2])

    ers = pd.merge(ers, ers_used, how="left", left_on=["pilot_index", "currentLapNum"], right_on = ["pilot_index", "currentLapNum"])
    a, b, c = train_ers_model(ers)
    
    return {
#         "ERS_Disable" : int(a),
        "ERS_Normal" : int(b),
        "ERS_Overtake" : int(c),
    }

def get_energy_recovered(df):
    lap_restriction = df["currentLapNum"].max() - 1
    ers_harvested = df[["pilot_index", "currentLapNum", 'ersHarvestedThisLapMGUK', 'ersHarvestedThisLapMGUH']].groupby(["pilot_index", "currentLapNum"]).max().reset_index()
    ers_harvested = ers_harvested[ers_harvested["currentLapNum"] < lap_restriction]
    return {
        "ersHarvested_MGUK" : int(ers_harvested.median()['ersHarvestedThisLapMGUK']),
        "ersHarvested_MGUH" : int(ers_harvested.median()['ersHarvestedThisLapMGUH']),
    }

"""
Tyre degradation
"""

def make_dataframe_tyre(df):
    df_tyre = df[[
        "frameIdentifier",
        "pilot_index",
        "currentLapTime",
        "currentLapNum",
        "fuelInTank",
        "tyresWear",
        "actualTyreCompound",
        "tyresDamage"
    ]]
    
    wear = df_tyre["tyresDamage"].str.split("/", expand=True).astype("float32")
    wear = wear.mean(axis=1).rename("wear")

    df_tyre = pd.concat([df_tyre[["frameIdentifier", "pilot_index", "currentLapTime", "fuelInTank", "actualTyreCompound", "currentLapNum"]], wear], axis=1)
    
    return df_tyre
    
def train_degradation_model(df):
    summary = df.groupby('currentLapNum').tail(1)
    summary = summary[summary["currentLapTime"] > 50]
    summary = summary[summary["wear"].notnull()]
    
    wear = summary["wear"].to_list()
    tyre = summary["actualTyreCompound"].to_list()
    
    wears = []
    buffer = [wear[0]]
    for c, prev, curr in zip(tyre[:-1], wear[:-1], wear[1:]):
        if prev > curr:
            wears.append((c, buffer))
            buffer = [curr]
        else:
#             if not math.isnan(curr):
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

def get_tyre_degradation(df):
    subdf = make_dataframe_tyre(df)
    results = subdf.groupby("pilot_index").apply(train_degradation_model).to_list()
    
    agg_result = {
        "soft" : [],
        "medium" : [],
        "hard" : [],
    }

    for result in results:
        for key, vals in result.items():
            agg_result[key] += vals
    
    
    soft = sum(agg_result["soft"]) / len(agg_result["soft"]) if len(agg_result["soft"]) > 0 else 0
    medium = sum(agg_result["medium"]) / len(agg_result["medium"]) if len(agg_result["medium"]) > 0 else 0
    hard = sum(agg_result["hard"]) / len(agg_result["hard"]) if len(agg_result["hard"]) > 0 else 0
    
    return {
        "tyre_degradation_soft" : soft,
        "tyre_degradation_medium" : medium,
        "tyre_degradation_hard" : hard,
    }

"""
Track Stats
"""

def get_distance(df, feature):
    dist = df["lapDistance"].values

    ans = 0
    for key, seq in groupby( df[feature].values):
        n = len(list(seq))
        if key:
            ans += dist[n-1] - dist[0] 
        dist = dist[n:]

    return ans

def get_power_area(df, track_length):
    
    df["flat_out"] = df["throttle"]>0.95
    df["braking_zone"] = df["brake"]>0.30
    
    flat = df.groupby(["pilot_index", "currentLapNum"]).apply(get_distance, feature="flat_out").reset_index()
    brake = df.groupby(["pilot_index", "currentLapNum"]).apply(get_distance, feature="braking_zone").reset_index()

    ans = pd.merge(flat, brake, how="left", left_on=["pilot_index", "currentLapNum"], right_on = ["pilot_index", "currentLapNum"])
    ans.columns = ["pilot_index", "currentLapNum", "percent_lap_flat_out", "percent_lap_braking"]
    ans[["percent_lap_flat_out", "percent_lap_braking"]] /= track_length
    
    lap_restriction = df["currentLapNum"].max() - 1
    ans = ans[ans["currentLapNum"] < lap_restriction]
    
    return ans[["percent_lap_flat_out", "percent_lap_braking"]].quantile(0.5).to_dict()


def get_average_gear_change(df):
    def get_number_gear_change(df):
        ans = 0
        for key, seq in groupby(df["gear"]):
            ans += 1
        return ans

    ans = df.groupby(["pilot_index", "currentLapNum"]).apply(get_number_gear_change).rename("gear_change").reset_index()
    lap_restriction = df["currentLapNum"].max() - 1
    ans = ans[ans["currentLapNum"] < lap_restriction]
    return { "average_gear_change" : int(ans["gear_change"].median()) }
    
def get_average_speed(df, track_length):
    lap_restriction = df["currentLapNum"].max() - 2
    fastest_lap = df[df["currentLapNum"]<lap_restriction]["LapTime"].min()
    fastest_avg_speed = track_length / fastest_lap * 3.6
    avg_speed = track_length / df["LapTime"].median() * 3.6
    return {
        "average_fastest_lap" : fastest_avg_speed,
        "average_median_lap" : avg_speed
    }

def get_high_and_slow_sections(df, track_length):
    df["above300"] = df["speed"]>300
    df["below150"] = df["speed"]<150
    above300 = df.groupby(["pilot_index", "currentLapNum"]).apply(get_distance, "above300").rename("percent_lap_above_300").reset_index()
    below150 = df.groupby(["pilot_index", "currentLapNum"]).apply(get_distance, "below150").rename("percent_lap_below_150").reset_index()

    lap_restriction = df["currentLapNum"].max() - 1
    ans = pd.merge(above300, below150, how="left", left_on=["pilot_index", "currentLapNum"], right_on = ["pilot_index", "currentLapNum"])
    ans = ans[ans["currentLapNum"]<lap_restriction]
    ans[["percent_lap_above_300", "percent_lap_below_150"]] /= track_length
    return ans[["percent_lap_above_300", "percent_lap_below_150"]].median().to_dict()
    
def get_median_highest_speed(df):
    def get_max_speed(df):
        idx = df["speed"].argmax()
        return df.iloc[idx][["drs", "speed"]]

    lap_restriction = df["currentLapNum"].max() - 1
    speed = df.groupby(["pilot_index", "currentLapNum"]).apply(get_max_speed).reset_index()
    speed = speed[speed["currentLapNum"]<lap_restriction]
    return {
        "highest_speed_with_drs": speed[speed["drs"] == True]["speed"].median(),
        "highest_speed_without_drs": speed[speed["drs"] == False]["speed"].median()
    }
    
"""
PIT Duration
"""
    
def get_pit_durations_per_lap(df):
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

def get_average_pit_duration(detail):
    pit_area_durations, pit_lane_durations = [], []
    for pit_area_duration, pit_lane_duration in detail:
        if len(pit_area_duration)>0:
            pit_area_durations += pit_area_duration
            pit_lane_durations += pit_lane_duration
    
    ans_pit_lane = statistics.median(pit_lane_durations) if len(pit_lane_durations) > 0 else 0
    ans_pit_area = statistics.median(pit_area_durations) if len(pit_area_durations) > 0 else 0
    return ans_pit_lane, ans_pit_area

def get_pit_lane_entry_and_exit(df):
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
#                 if lap_distance[0] > 2000 and 300 < lap_distance[n_frames-1] < 2000: # entry is always at the end of a lap and a lap is more than 3km
                dists_entry.append(lap_distance[0])
                dists_exit.append(lap_distance[n_frames-1])
            lap_distance = lap_distance[n_frames:]
            
    return statistics.median(dists_entry), statistics.median(dists_exit)

def get_time_eq_on_track(df, ref):
    x = df["lapDistance"].values
    t = df["currentLapTime"].values
    dx = x - ref
    idx_min = np.argmin(np.abs(dx))
    if dx[idx_min] < 10:  # when I finish the lap, AI behind are directly stopped so we may not find the real min
        return t[idx_min]
    else:
        return None

def get_time_out_of_pit(df, track_length, race):
    pit_data = df.groupby(["pilot_index"]).apply(get_pit_durations_per_lap).values
    pit_stop_duration, pitting_duration = get_average_pit_duration(pit_data)
    
    entry, exit = get_pit_lane_entry_and_exit(df)
    pit_length = track_length - entry + exit
    entry_t = df.groupby(["pilot_index", "currentLapNum"]).apply(get_time_eq_on_track, entry).rename("entryTime").reset_index()
    exit_t = df.groupby(["pilot_index", "currentLapNum"]).apply(get_time_eq_on_track, exit).rename("exitTime").reset_index()
    
    time_info = pd.merge(entry_t, race,  how='left', left_on=["pilot_index", "currentLapNum"], right_on = ["pilot_index", "currentLapNum"])
    time_info = pd.merge(time_info, exit_t,  how='left', left_on=["pilot_index", "currentLapNum"], right_on = ["pilot_index", "currentLapNum"])
    time_info["pit_duration_in_track"] = time_info["LapTime"] - time_info["entryTime"] + time_info["exitTime"]
    duration_track = time_info[time_info["pit_duration_in_track"] > 3]["pit_duration_in_track"].median()
    
    return {
        "pit_lane_duration" : pit_stop_duration, 
        "pit_area_duration" : pitting_duration,
        "pit_length" : pit_length,
        "duration_on_track" : duration_track,
        "time_lost_per_stop" : pit_stop_duration - duration_track
    }

"""
MAIN
"""

def process(ID):
    telemetry, session, participant, race = load_data(ID)
    session["weather"] = "Dry"
    myID = get_my_id(participant)
    telemetry = remove_flashbacks(telemetry, pilot=myID)
    fuel_data = get_fuel_estimation(telemetry, session["trackLength"])
    ers_data = get_ERS_estimation(telemetry, session["trackLength"])
    ers_recovered = get_energy_recovered(telemetry)
    tyre_degradation = get_tyre_degradation(telemetry)
    throttle = get_power_area(telemetry, session["trackLength"])
    gear_change = get_average_gear_change(telemetry)
    speed = get_average_speed(race, session["trackLength"])
    section_speed = get_high_and_slow_sections(telemetry, session["trackLength"])
    highest_speed = get_median_highest_speed(telemetry)
    pit_data = get_time_out_of_pit(telemetry, session["trackLength"], race)
    return {
        "ID" : ID,
        **session,
        **fuel_data,
        **ers_data,
        **ers_recovered,
        **tyre_degradation,
        **throttle,
        **gear_change,
        **speed,
        **section_speed,
        **highest_speed,
        **pit_data,
    }
all_results = []
for i, file in enumerate(iglob("/kaggle/input/f1-2020-race-data/TelemetryData_*.csv")):
    ID = re.search('_(\d+).', file).group(1)
    print(f"{i+1:0>2}/22 - {ID}")
    all_results.append(process(ID))

final = pd.DataFrame(all_results)
final["totalDistance"] = final["totalLaps"] * final["trackLength"]
    
rearranged = ['ID',
 'weather',
 'trackTemperature',
 'airTemperature',
 'totalLaps',
 'trackLength',
 'totalDistance',
 'trackId',
 'fuel_consumption_lean',
 'fuel_consumption_normal',
 'fuel_consumption_rich',
 'ERS_Normal',
 'ERS_Overtake',
 'ersHarvested_MGUK',
 'ersHarvested_MGUH',
 'tyre_degradation_soft',
 'tyre_degradation_medium',
 'tyre_degradation_hard',
 'percent_lap_flat_out',
 'percent_lap_braking',
 'average_gear_change',
 'average_fastest_lap',
 'average_median_lap',
 'percent_lap_above_300',
 'percent_lap_below_150',
 'highest_speed_with_drs',
 'highest_speed_without_drs',
 'pit_lane_duration',
 'pit_area_duration',
 'pit_length',
 'duration_on_track',
 'time_lost_per_stop'
]
final = final[rearranged]
final = final.round(3)
final["highest_speed_with_drs"] = final["highest_speed_with_drs"].astype(int)
final["highest_speed_without_drs"] = final["highest_speed_without_drs"].astype(int)
final[["percent_lap_flat_out", "percent_lap_braking", "percent_lap_above_300", "percent_lap_below_150"]] *= 100
final.columns
final.to_csv("/kaggle/working/final.csv", index=False)
df = pd.read_csv("/kaggle/working/final.csv")
df["dist_above_300"] = df["percent_lap_above_300"] * df["trackLength"] / 100
df["dist_below_150"] = df["percent_lap_below_150"] * df["trackLength"] / 100
df["dist_below_300"] = df["trackLength"] - df["dist_above_300"]

fig, ax = plt.subplots(figsize=(20, 12))
sns.barplot(x="trackId", y="trackLength", data = df, ax=ax, color="r", label="above 300")
sns.barplot(x="trackId", y="dist_below_300", data = df, ax=ax, color="g", label="between 150 and 300")
sns.barplot(x="trackId", y="dist_below_150", data = df, ax=ax, color="b", label="below 150")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.set_ylabel("trackLength")
ax.legend()
plt.title("Distance at given speed per Track")
plt.show()
fig, ax = plt.subplots(figsize=(20, 12))
sns.barplot(x="trackId", y="fuel_consumption_rich", data = df, ax=ax, color="r", label="Rich")
sns.barplot(x="trackId", y="fuel_consumption_normal", data = df, ax=ax, color="g", label="Normal")
sns.barplot(x="trackId", y="fuel_consumption_lean", data = df, ax=ax, color="b", label="Lean")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.set_ylabel("Consumption kg/Lap")
ax.legend()
plt.title("Fuel Mix consumption")
plt.show()
fig, ax = plt.subplots(figsize=(20, 12))
sns.barplot(x="trackId", y="tyre_degradation_soft", data = df, ax=ax, color="r", label="Soft")
sns.barplot(x="trackId", y="tyre_degradation_medium", data = df, ax=ax, color="g", label="Medium")
sns.barplot(x="trackId", y="tyre_degradation_hard", data = df, ax=ax, color="b", label="Hard")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.set_ylabel("Tyre Degradation %/Lap")
ax.legend()
plt.title("Tyre Degradation")
plt.show()
df["totalHarvested"] = df['ersHarvested_MGUK'] + df['ersHarvested_MGUH']

fig, (ax, ax2) = plt.subplots(1, 2, sharey='row', figsize=(30, 12))
sns.barplot(x="trackId", y="totalHarvested", data = df, ax=ax, color="r", label="ersHarvested_MGUH")
sns.barplot(x="trackId", y="ersHarvested_MGUK", data = df, ax=ax, color="g", label="ersHarvested_MGUK")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.set_ylabel("Energy J/Lap")
ax.legend()
ax.set_title("Energy Harvested per Lap")

sns.barplot(x="trackId", y="ERS_Normal", data = df, ax=ax2, color="b", label="ERS Consummed W/O OT")
ax2.set_title("Energy Consummed without OT per Lap")
ax2.legend()
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)

plt.show()

fig, ax = plt.subplots(figsize=(20, 12))
sns.barplot(x="trackId", y="average_gear_change", data = df, ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.set_ylabel("Gear change")
ax.legend()
plt.title("Gear change per Lap")
plt.show()
