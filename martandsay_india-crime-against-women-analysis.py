

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import math

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/RS_Session_246_AU_98_1.1.csv")
df.tail()
df.drop(df.index[[36]], inplace=True)
df.isnull().sum() # There are no missing values
plt.figure(figsize=(15, 15));

df.groupby(["State/UT", "2014 - Cases registered", "2015 - Cases registered", "2016 - Cases registered"]).sum().plot(kind="bar",width=0.7, figsize=(15, 10), title="State Vs Total Case Registered Against Women");

plt.plot();
df.head()
plt.figure(figsize=(15, 15));

df.groupby(["State/UT", "2014 - Total rape Cases", "2015 - Total rape Cases", "2016 - Total rape Cases"]).sum().plot(kind="bar",width=0.7, figsize=(15, 10), title="State Vs Total Rape Case Registered Against Women");

plt.plot();
df[["State/UT", "2014 - Cases registered", "2014 - Total rape Cases"]].head()
# adding rape percentage in 2014

df["Rape_2014_perc"] = (df["2014 - Total rape Cases"]/df["2014 - Cases registered"])*100

df["Rape_2014_perc"] = df["Rape_2014_perc"].map(lambda x: round(x, 2))
df.head()
# Same add for 2015 & 2016

df["Rape_2015_perc"] = (df["2015 - Total rape Cases"]/df["2015 - Cases registered"])*100

df["Rape_2015_perc"] = df["Rape_2015_perc"].map(lambda x: round(x, 2))

df["Rape_2016_perc"] = (df["2016 - Total rape Cases"]/df["2016 - Cases registered"])*100

df["Rape_2016_perc"] = df["Rape_2016_perc"].map(lambda x: round(x, 2))
df.head()


ax=df.groupby('State/UT')["Rape_2014_perc", "Rape_2015_perc", "Rape_2016_perc"].sum().plot.bar(stacked=True, figsize=(15, 15), title="State Vs Rape Percentage");

ax.set_xlabel("States");

ax.set_ylabel("Percentage(%)");
df2 = pd.read_csv("../input/RS_Session_246_AS11.csv")
df2.head()
# Remove total column

df2.drop(df2.index[[37, 38, 29]], inplace=True)
df_only_states_ut = df2.iloc[:37, :]
df_only_states_ut.drop(df_only_states_ut.index[29], inplace=True)
df_only_states_ut
df_only_states_ut.groupby("States/UTs").sum().plot.barh(figsize=(15,25), width=0.7);
df2["total_crime"] = df2.sum(axis = 1, skipna = True)
df2.head()
df2.groupby(["States/UTs"])["total_crime"].sum().plot.bar(figsize=(15, 10))