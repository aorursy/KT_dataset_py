import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# The original dataset

df = pd.read_csv("../input/UK_Traffic_Accidents_2015.csv")

df.head()
df = pd.read_csv("../input/UK_Traffic_Accidents_2015.csv", usecols=["Day_of_Week", "Time"])

df.dropna(inplace=True)

df.head()
df['Time'] = df['Time'].map(lambda x: str(x)[:-3])

df.head()
# Convert "Day_of_Week" to string

df["Day_of_Week"] = df["Day_of_Week"].astype(str)
# Adding column for ordering the days of the week

df['Day'] = df["Day_of_Week"]

df.head()
df["Day_of_Week"] = df["Day_of_Week"].replace("1", "Sunday")

df["Day_of_Week"] = df["Day_of_Week"].replace("2", "Monday")

df["Day_of_Week"] = df["Day_of_Week"].replace("3", "Tuesday")

df["Day_of_Week"] = df["Day_of_Week"].replace("4", "Wednesday")

df["Day_of_Week"] = df["Day_of_Week"].replace("5", "Thursday")

df["Day_of_Week"] = df["Day_of_Week"].replace("6", "Friday")

df["Day_of_Week"] = df["Day_of_Week"].replace("7", "Saturday")



df.head()
# Using .ctrosstab() to create a pivot table

df_pivot = pd.crosstab(df["Day_of_Week"], df["Time"])

df_pivot
# Making the index chronological

df_pivot = df_pivot.reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])

df_pivot
fig, ax = plt.subplots(figsize=(30,8))

graph = sns.heatmap(df_pivot, cmap="Blues", linecolor="white", linewidths=0.1)



ax.set_title("Number of traffic-related deaths per day & hour combination", y=1.3, fontsize=30, fontweight="bold")

ax.set_xlabel("")

ax.set_ylabel("")



#from matplotlib import rcParams

#rcParams['axes.titlepad'] = 130 # Space between the title and graph



locs, labels = plt.yticks() # Rotating row labels

plt.setp(labels, rotation=0) # Rotating row labels



ax.xaxis.tick_top() # x axis on top

ax.xaxis.set_label_position('top') # x axis on top



graph.tick_params(axis='both',labelsize=15) # Tick label size

graph