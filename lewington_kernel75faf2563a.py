# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("/kaggle/input/ashrae-energy-prediction/weather_train.csv")
targ = pd.read_csv("/kaggle/input/ashrae-energy-prediction/train.csv")
meta = pd.read_csv("/kaggle/input/ashrae-energy-prediction/building_metadata.csv")
meta.head()
x = pd.merge(targ, meta, on="building_id")
targ.head()
1.608716 * 10**5
3.104670 * 10** 2
2.333423 * 10**3
3.968185 * 10 ** 2
x.describe()
3.104670 * 10 ** 2
x.describe()
x.shape
high = targ[targ["meter_reading"] > 500000]
high.shape
subset = targ.sample(300000)
subset["timestamp"] = pd.to_datetime(subset["timestamp"])
plt.scatter(subset["timestamp"], subset["meter_reading"])
x.shape
x["meter_reading"].scatter()
x.corr()

print(data.shape)
print(targ.shape)
data.head()
pd.merge(targ, data)
data.head()
data["site_id"].value_counts()
sub1 = data[data["site_id"] == 1]
plt.plot(sub1["timestamp"], sub1["air_temperature"])
sub2 = data[data["site_id"] == 0]
plt.plot(sub2["timestamp"], sub2["air_temperature"])
sub["timestamp"].min()
def confidence_plot(df, col, target, rc={'figure.figsize':(15,10)}):
    sns.set(rc=rc)
    ax = sns.countplot(x=col, data=df)
    ax2 = ax.twinx()
    ax.set_xticklabels(ax.get_xticklabels(),rotation=80)    
    ax2 = sns.pointplot(x=col, y=target, data=df, color='black', legend=False, errwidth=0.5)
    ax.grid(False)

data["diff"] = data["air_temperature"] - data["sea_level_pressure"] 
data["diff"].min()
corr = data.corr()
corr
sub = data[data["site_id"] == 1]
sub2 = data[data["site_id"] == 2]


_, ax = plt.subplots(2, figsize=(20, 10))

print(type(tup))
print(type(fig))
print(type(ax))


ax[0].plot(sub["timestamp"], sub["wind_direction"])
ax[1].plot(sub["timestamp"], sub["wind_speed"])
# ax[2].plot(sub["timestamp"], sub["sea_level_pressure"])


rows = 2
cols = 3

fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(20, 10))
for i in range(rows):
    for j in range(cols):
        site_id = i+j
        print(site_id)
        axes[i, j].plot(data[data["site_id"] == site_id]["timestamp"], data[data["site_id"] == i+j]["air_temperature"])
        axes[i, j].plot(data[data["site_id"] == site_id]["timestamp"], data[data["site_id"] == i+j]["dew_temperature"])        
