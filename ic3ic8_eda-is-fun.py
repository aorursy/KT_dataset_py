import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
plt.style.use('seaborn')

import os
print(os.listdir("../input"))
train = pd.read_csv("../input/PM25_train.csv")

train["Date_Time"] = train["Date"] +" "+ train["Time"]
train["Date_Time"] = pd.to_datetime(train["Date_Time"])

sector =  train.groupby("device_id")
dict_df_data = {}
for key, value in sector.groups.items():
    dict_df_data[key] = train.iloc[value, :]
dict_df_data_per_hour = {}

for key, value in dict_df_data.items():
        dict_df_data_per_hour[key] = value.groupby(value["Date_Time"].dt.round("1h")).mean()
df_total = pd.DataFrame()
for index in dict_df_data_per_hour:
        one = dict_df_data_per_hour[index][["PM2.5"]]
        one.columns = [index]
        df_total = pd.concat([df_total, one], axis=1)
df_mean_day = pd.DataFrame()
for i in dict_df_data_per_hour:
        one = df_total[i].values[:24*29].reshape(29,24)
        df_one = pd.DataFrame(np.nanmean(one,  axis=0), columns = [i])
        df_mean_day = pd.concat([df_mean_day, df_one], axis=1)
index = []
for i in range(8, 25):
        index.append(i)
for i in range(1, 8):
        index.append(i)
df_mean_day.index = index
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(df_mean_day,  vmin=30, vmax=50, cmap="YlGnBu")
plt.show()

