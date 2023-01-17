# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

import datetime

import matplotlib.dates as mdates

import matplotlib.dates as dates

from datetime import datetime,timedelta
#../input/temperature-readings-iot-devices/IOT-temp.csv

IoT_temp = pd.read_csv("../input/temperature-readings-iot-devices/IOT-temp.csv",parse_dates=['noted_date'])

IoT_temp.head()
IoT_temp.head(100)
IoT_temp.info()
IoT_temp.describe()
IoT_temp.describe(include='O')
fig,ax = plt.subplots(1,figsize=(7,5))

sns.countplot(IoT_temp['out/in'])

plt.show()
fig,ax = plt.subplots(1,figsize=(15,8))

sns.kdeplot(IoT_temp.loc[IoT_temp['out/in']=='Out','temp'],shade=True,ax=ax, label="outside temp")

sns.kdeplot(IoT_temp.loc[IoT_temp['out/in']=='In','temp'],shade=True,ax=ax,label="inside temp")

plt.show()
fig,ax = plt.subplots(1,figsize=(15,8))

sns.distplot(IoT_temp.loc[IoT_temp['out/in']=='Out','temp'],ax=ax)

sns.distplot(IoT_temp.loc[IoT_temp['out/in']=='In','temp'],ax=ax)

plt.show()
IoT_temp.iloc[99]["noted_date"]
IoT_temp.iloc[0]["noted_date"]
# Load data

df = IoT_temp

df["noted_date"] = pd.to_datetime(df["noted_date"])



start_datetime = df.iloc[99]["noted_date"]

end_datetime = df.iloc[0]["noted_date"]

title_label = start_datetime.strftime("%Y/%m/%d")



fig = plt.figure(figsize=(20,4), dpi=200)

ax = fig.add_subplot(1,1,1)



ax.bar(x=df["noted_date"], height=df["temp"], width=0.008, color="lightgreen", align="edge")

ax.set_xlim(start_datetime, end_datetime)

ax.set_title(title_label, fontsize=10)

ax.set_facecolor("white")

ax.grid(axis="both", which="both", linewidth=0.5, linestyle="dashed", alpha=0.5)



# 主目盛設定

# x軸目盛のラベルに時刻を表示すると補助目盛のラベルと被ってしまうので日付だけ。

ax.xaxis.set_major_locator(dates.DayLocator())

ax.xaxis.set_major_formatter(dates.DateFormatter("%m/%d %a"))

ax.tick_params(axis="x", which="major", labelsize=8)

ax.tick_params(axis="y", which="major", labelsize=8)



# 補助目盛設定

ax.xaxis.set_minor_locator(dates.HourLocator(interval=3))

ax.xaxis.set_minor_formatter(dates.DateFormatter("\n\n%H:%M"))

ax.tick_params(axis="x", which="minor", labelsize=6)



#外のデータと、中のデータでわける

df1 = IoT_temp[IoT_temp['out/in']=='Out']

df2 = IoT_temp[IoT_temp['out/in']=='In']

df1
# データフレームの準備



start_datetime = df.iloc[99]["noted_date"]

end_datetime = df.iloc[0]["noted_date"]



target_df1 = df1[(df1["noted_date"] > start_datetime)&(df1["noted_date"] < end_datetime)]

target_df2 = df2[(df2["noted_date"] > start_datetime)&(df2["noted_date"] < end_datetime)]





# 描画

fig = plt.figure(figsize=(20,4), dpi=200)

ax = fig.add_subplot(1,1,1)





ax.set_xlim(start_datetime, end_datetime)

ax.set_title(title_label, fontsize=10)

ax.set_facecolor("white")

ax.grid(axis="both", which="both", linewidth=0.5, linestyle="dashed", alpha=0.5)



ax.plot(target_df1["noted_date"], target_df1["temp"])

ax.plot(target_df2["noted_date"], target_df2["temp"])





# 軸目盛の設定

ax.xaxis.set_major_locator(dates.DayLocator())

ax.xaxis.set_major_formatter(dates.DateFormatter("%m/%d %a"))



# 軸目盛ラベルの回転

labels = ax.get_xticklabels()

plt.setp(labels, rotation=45, fontsize=10);



ax.grid()
# building new features for time stamp.



def features_build(df):

    df['Date'] = pd.to_datetime(df['noted_date'])

    df['Year'] = df['noted_date'].dt.year

    df['Month'] = df.Date.dt.month

    df['Day'] = df.Date.dt.day

    df['WeekOfYear'] = df.Date.dt.weekofyear

    

features_build(IoT_temp)

fig ,ax = plt.subplots(1,figsize=(8,5))

sns.scatterplot(x="Month", y="temp", hue="out/in", data=IoT_temp,ax=ax)

plt.show()

# plotting discrete tempt values for month time stamp.
