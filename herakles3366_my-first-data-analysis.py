import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns
data_frame_00 = pd.read_csv("/kaggle/input/corona-virus-report/country_wise_latest.csv")

data_frame_00.head()
print("Columns : \n",data_frame_00.columns)
f, ax = plt.subplots(figsize=(12, 12))

sns.heatmap(data_frame_00.corr(), annot = True, linewidth = .5, fmt = ".2f", ax = ax)

plt.show()
data_frame_00['Deaths / 100 Cases'].plot(kind = "hist", bins = 50, alpha = 0.7, color = "green", grid = True, linestyle = ":", label = "Deaths / 100 Cases")

plt.legend()

plt.xlabel("x axis")

plt.ylabel("y axis")

plt.show()
data_frame_00['Recovered / 100 Cases'].plot(kind = "hist", bins = 50, alpha = 0.7, color = "green", grid = True, linestyle = ":", label = "Recovered / 100 Cases")

plt.legend()

plt.xlabel("x axis")

plt.ylabel("y axis")

plt.show()
data_frame_00['1 week change'].plot(kind = "line", figsize = (10,10), alpha = 0.8, color = "red", grid = True, label = "1 week change")

data_frame_00['1 week % increase'].plot(kind = "line", figsize = (10,10), alpha = 0.9, color = "blue", grid = True, label = "1 week % increase")

plt.legend()

plt.xlabel("x axis")

plt.ylabel("y axis")

plt.show()
data_frame_00['1 week change'].plot(kind = "line", figsize = (10,10), alpha = 0.8, color = "red", grid = True, label = "1 week change")

data_frame_00['Confirmed last week'].plot(kind = "line", figsize = (10,10), alpha = 0.9, color = "blue", grid = True, label = "Confirmed last week")

plt.legend()

plt.xlabel("x axis")

plt.ylabel("y axis")

plt.show()
data_frame_00.plot(kind = "scatter", x = "Active", y = "New recovered", alpha = 0.5, color = "red")

plt.xlabel("New cases")

plt.ylabel("New deaths")

plt.title("Attack Defense Scatter Plot")

plt.show()
count = {"Eastern Mediterranean":0,"Europe":0,"Africa":0,"Americas":0,"Western Pacific":0,"South-East Asia":0}



count["Eastern Mediterranean"] = np.sum(data_frame_00[data_frame_00['WHO Region'] == "Eastern Mediterranean"].Confirmed)

count["Europe"] = np.sum(data_frame_00[data_frame_00['WHO Region'] == "Europe"].Confirmed)

count["Africa"] = np.sum(data_frame_00[data_frame_00['WHO Region'] == "Africa"].Confirmed)

count["Americas"] = np.sum(data_frame_00[data_frame_00['WHO Region'] == "Americas"].Confirmed)

count["Western Pacific"] = np.sum(data_frame_00[data_frame_00['WHO Region'] == "Western Pacific"].Confirmed)

count["South-East Asia"] = np.sum(data_frame_00[data_frame_00['WHO Region'] == "South-East Asia"].Confirmed)





plt.pie(count.values(), labels = count.keys())

plt.title("Distribution the Covid-19 according to the continents")

plt.show()
filter = data_frame_00["Confirmed"] > 20000

data_frame_00[filter]
data_frame_00[np.logical_and(data_frame_00["Recovered / 100 Cases"] > 50, data_frame_00["Deaths / 100 Cases"] < 30 )]["Country/Region"].head()