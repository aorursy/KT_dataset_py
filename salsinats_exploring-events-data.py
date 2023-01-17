# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

%matplotlib inline

import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime as dt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

c = {'axes.titlesize': 24,
     'axes.labelsize': 18,
     'axes.suptitlesize': 20,
     'legend.fontsize': 20,
     'xtick.labelsize': 18,
     'ytick.labelsize': 18,
     'lines.linewidth': 3,
     'lines.markersize': 10,
     'axes.grid': False,
     'pdf.fonttype': 42,
     'ps.fonttype': 42}

# events information
events = pd.read_csv("../input/events.csv")

# gender ager train/test data
gender_age_train = pd.read_csv("../input/gender_age_train.csv")
gender_age_test = pd.read_csv("../input/gender_age_test.csv")
# devices in events dataset
events_devices = pd.unique(events.device_id)
print("# devices :", len(events_devices))
ga_train_devices = pd.unique(gender_age_train.device_id)
ga_test_devices = pd.unique(gender_age_test.device_id)

mask_train = np.in1d(ga_train_devices, events_devices)
mask_test = np.in1d(ga_test_devices, events_devices)

id_train = np.where(mask_train)[0]
id_test = np.where(mask_test)[0]
print("# devices in train:", len(ga_train_devices))
print("# devices in train inter events:", len(id_train))

print("\n# devices in test :", len(ga_test_devices))
print("# devices in test inter events:", len(id_test))
# start and end dates
timestamps = events["timestamp"].values
print("beginning :\t", np.min(timestamps))
print("end :\t\t", np.max(timestamps))
# We select the first device of the tab events
i = events_devices[0]
# We select now all the events corresponding to this device i
events_i = events[events.device_id == i]
events_i.head()
# stats about longitude
m_i = np.mean(events_i["longitude"].values)
s_i = np.std(events_i["longitude"].values)
print("mean longitude (std) : {:4.2f} ({:4.2f})".format(m_i, s_i))
# stats about latitude
m_i = np.mean(events_i["latitude"].values)
s_i = np.std(events_i["latitude"].values)
print("mean longitude (std) : {:4.2f} ({:4.2f})".format(m_i, s_i))
events_i = events_i.sort_values(by="timestamp")
events_i.head()
# displays last recorded events for device i
events_i.tail(n=5)
# remove the rows with (long. lat.) = (0, 0)
events_i_no0 = events_i[events_i.longitude != 0]
# stats about longitude
m_i = np.mean(events_i_no0["longitude"].values)
s_i = np.std(events_i_no0["longitude"].values)
print("mean longitude (std) : {:4.2f} ({:4.2f})".format(m_i, s_i))

# stats about latitude
m_i = np.mean(events_i_no0["latitude"].values)
s_i = np.std(events_i_no0["latitude"].values)
print("mean longitude (std) : {:4.2f} ({:4.2f})".format(m_i, s_i))
# we first convert timestamps from string to date / time format and store the value in a new column "time_f"
time_t = lambda x: dt.strptime(x, "%Y-%m-%d %H:%M:%S")
events_i_no0["time_f"] = events_i_no0["timestamp"].map(time_t)
# we do that again store the value in the new column "time_i" and shift the row of 1 step
events_i_no0["time_i"] = events_i_no0["timestamp"].map(time_t)
events_i_no0["time_i"] = events_i_no0["time_i"].shift(1)
events_i_no0.head()
# we now define a function to compute the time interval between two timestamps in second and apply it to our dataframe

def time_interval(df):
    """
    INPUTS:
    - df: dataframe with columns 'time_f' and 'time_i'
    
    OUTPUTS:
    - time in second between the two timestamps 'time_f' and 'time_i'
    """
    return (df["time_f"] - df["time_i"]).total_seconds()

events_i_no0["time_interval"] = events_i_no0.apply(time_interval, axis=1)
events_i_no0.head()
# first, we drop rows with nan
events_i_no0 = events_i_no0.dropna()

time_int_i = events_i_no0["time_interval"].values

# normalized histogram
sns.set_style("whitegrid")
sns.set_context("paper", rc=c)

plt.figure(figsize=(10, 10))
bins = np.linspace(0, 10000, 30)
h = plt.hist(time_int_i, bins, normed=1, color="b", label="29182687948017175", alpha=0.5)

plt.legend()
plt.xlabel("time in s")
plt.title("Time intervals between events")
# normalized histogram
sns.set_style("whitegrid")
sns.set_context("paper", rc=c)

fig = plt.figure(figsize=(10, 20))

bins = np.linspace(0, 600000, 30)

for j in range(5):
    ax = plt.subplot2grid((5, 1), (j, 0))
    i = events_devices[j]
    events_i = events[events.device_id == i]
    events_i_no0 = events_i[events_i.longitude != 0]
    events_i_no0["time_f"] = events_i_no0["timestamp"].map(time_t)
    events_i_no0["time_i"] = events_i_no0["timestamp"].map(time_t)
    events_i_no0["time_i"] = events_i_no0["time_i"].shift(1)
    events_i_no0["time_interval"] = events_i_no0.apply(time_interval, axis=1)
    events_i_no0 = events_i_no0.dropna()
    time_int_i = events_i_no0["time_interval"].values
    h = ax.hist(time_int_i, bins, normed=1, label=str(i), alpha=0.5)
    ax.legend()
    ax.set_xlabel("time in s")
    ax.set_ylim([0., 0.00003])

fig.tight_layout()

