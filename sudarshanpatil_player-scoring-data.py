#load libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import warnings as ws

ws.filterwarnings("ignore")
df = pd.read_csv("../input/ipl-2020-data/2008-2019 IPL data/2008-2019 IPL data/most_runs_average_strikerate_2008_to_2019.csv")

df.head()
# EDA Starts
# Top 10 run scorer's in the the IPL

sns.set()

fig,ax = plt.subplots(figsize =(15,6))

top = df.sort_values(by="total_runs", ascending  =False).head(10)[["batsman", "total_runs"]]

sns.barplot(x = top["batsman"], y = top["total_runs"], palette="magma", ax=ax)

ax.set(title  ="Top 10 Scoring batsman in IPL")

plt.xticks(rotation =  30)

plt.show()
# Top 10 run Dismissed batsman in the the IPL

sns.set()

fig,ax = plt.subplots(figsize =(15,6))

dismissed = df.sort_values(by="out", ascending  =False).head(10)[["batsman", "out"]]

sns.barplot(x = dismissed["batsman"], y = dismissed["out"], palette="inferno", ax=ax)

ax.set(title  ="Top 10 Dismissed batsman in IPL")

plt.xticks(rotation =  30)

plt.show()
# Top 10 high Averaged players

sns.set()

fig,ax = plt.subplots(figsize =(15,6))

top_avg = df.sort_values(by="average", ascending  =False).head(10)[["batsman", "average", "total_runs"]].sort_values(by ="total_runs", ascending = False).head(5)

sns.barplot(x = top_avg["batsman"], y = top_avg["average"], palette="inferno", ax=ax)

ax.set(title  ="Top 10 high averaging batsman in IPL")

plt.xticks(rotation =  30)

plt.show()
temp = df.sort_values(by="numberofballs", ascending  =False).head(10)[["batsman","total_runs", "numberofballs"]]

temp["avg"] = temp["total_runs"] /  temp["numberofballs"]
sns.set()

fig,ax = plt.subplots(figsize =(15,6))

sns.barplot(x = temp["batsman"], y = temp["avg"], palette="rainbow", ax=ax)

ax.set(title  ="Balls / Run Ratio")

plt.xticks(rotation =  30)

plt.show()
# Top 10 Strike Rate holders in IPL

averaging = df.sort_values(by= "strikerate", ascending = False)
sns.set()

fig,ax = plt.subplots(figsize =(15,6))

strike_2000 = averaging[averaging["total_runs"] > 2000].head(10)

sns.barplot(x=strike_2000["batsman"], y =strike_2000["strikerate"], palette="mako")

ax.set(title= "Top 10 Strike rate holders (UPTO 2000 Runs)")

plt.show()

# Highest averaging batsman upto 3000 Runs are

sns.set()

fig,ax = plt.subplots(figsize =(15,6))

strike_3000 = averaging[averaging["total_runs"] > 3000].head(10)

sns.barplot(x=strike_3000["batsman"], y =strike_3000["strikerate"], palette="Purples_r")

ax.set(title= "Top 10 Strike rate holders (UPTO 3000 Runs)")

plt.show()

# Highest averaging batsman upto 4000 Runs are

sns.set()

fig,ax = plt.subplots(figsize =(15,6))

strike_4000 = averaging[averaging["total_runs"] > 4000].head(10)

sns.barplot(x=strike_4000["batsman"], y =strike_4000["strikerate"], palette="Spectral_r")

ax.set(title= "Top 10 Strike rate holders (UPTO 4000 Runs)")

plt.show()

# Highest averaging batsman upto 5000 Runs are

sns.set()

fig,ax = plt.subplots(figsize =(15,6))

strike_5000 = averaging[averaging["total_runs"] > 5000].head(10)

sns.barplot(x=strike_5000["batsman"], y =strike_5000["strikerate"], palette="Spectral_r")

ax.set(title= "Top 10 Strike rate holders (UPTO 5000 Runs)")

plt.show()
# Runs per out

runs_per_innings = df.sort_values(by="total_runs", ascending  =False).head(10)[["batsman", "total_runs", "out"]]

runs_per_innings["runs_per_inn"]=  runs_per_innings["total_runs"] / runs_per_innings["out"]

sns.set()

fig,ax = plt.subplots(figsize =(15,6))

sns.barplot(x=runs_per_innings["batsman"], y =runs_per_innings["runs_per_inn"], palette="Spectral_r")

ax.set(title= "Runs / innings by the top Scoring batsman")

plt.show()
