import numpy as np 
import pandas as pd 

from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
%matplotlib inline

import os
print(os.listdir("../input"))

import warnings
warnings.filterwarnings("ignore")
data = pd.read_csv("../input/sdpd-data/vehicle_stops_2016_datasd.csv")
race_codes = pd.read_csv("../input/sdpd-race-codes/vehicle_stops_race_codes.csv")
data.head()
pd.isnull(data).sum()
race_codes
race_codes_dictionary = {}

for i in race_codes["Race Code"]:
    race_codes_dictionary[i] = race_codes.loc[race_codes["Race Code"] == i, "Description"].iloc[0]

race_codes_dictionary
set(data["subject_race"])
data = data.dropna(subset=["subject_race"])

data = data.replace("X", "O")

races = list(data["subject_race"])
subject_race = []

for i in races:
    subject_race.append(race_codes_dictionary[i])

data["Subject Race"] = subject_race
set(data["subject_sex"])
set(data["arrested"])
data.shape
print(data.keys())
data.describe()
set(data["stop_cause"])
set(data["subject_age"])
subject_ages = []

data = data.dropna(subset=["subject_age"])

for i in data["subject_age"]:
    if i == "No Age":
        subject_ages.append(-1)
    else:
        subject_ages.append(int(i))

data["subject_age"] = subject_ages

plt.hist(data["subject_age"])
set(data["subject_age"])
