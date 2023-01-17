#packages etc

import pandas as pd

import matplotlib.pyplot as plt #plotting library

import seaborn as sns



%matplotlib inline

pd.set_option('display.max_rows', 10) #ridade arv
df1 = pd.read_csv("../input/matches.csv")

df1
print("Mängu keskmine pikkus:", round(df1["duration"].mean() / 60, 2), "min")
df2 = pd.read_csv("../input/participants.csv")

df2
mitte_g = df2["ss1"] + df2["ss2"]

flash_g = 0

for arv in mitte_g:

    flash_g += (20 - arv)

flash_g
flash_f = df2["ss1"].sum()

flash_f
df3 = pd.read_csv("../input/stats1.csv")

df3
#line plot

#linewidth = width of line, alpha = opacity, grid = grid, linestyle = style of line

df3.pinksbought.plot(kind = "line", color = "y", label = "control wards bought", linewidth = 1, alpha = 0.5, grid = True, linestyle = ":")

df3.wardsplaced.plot(color = "r", label = "wards placed", linewidth = 1, alpha = 0.5, grid = True, linestyle = "-.")

plt.legend(loc = "upper right") #legend = puts label into plot

plt.xlabel("x axis")

plt.ylabel("y axis")

plt.title("Line Plot")      
df3.groupby("win").aggregate({"wardsplaced": ["sum"]})
#scatter plot 

df3.plot(kind = "scatter", x = "physdmgtochamp", y = "magicdmgtochamp", alpha = 0.5, color = "red")

plt.xlabel("AD")

plt.ylabel("AP")

plt.title("AD vs AP")
kills = df3[["doublekills", "triplekills", "quadrakills", "pentakills", "legendarykills"]]

kills
#correlational map

f, ax = plt.subplots(figsize = (10, 10))

sns.heatmap(kills.corr(), annot = True, linewidths = .5, fmt = ".2f", ax = ax)
df4 = pd.read_csv("../input/teamstats.csv")

objects = df4[["baronkills", "dragonkills", "harrykills"]]

objects
objects.plot.hist(grid = True, rwidth = 0.95); 