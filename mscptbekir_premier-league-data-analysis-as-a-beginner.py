# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/results.csv")
data.columns
data.info()  #for data info
data.corr() #corelation
f, ax = plt.subplots(figsize=(8,8)) #sizeofplot
sns.heatmap(data.corr(), annot=True, linewidths= .5, fmt=".1f", ax = ax) #for heatmap details
plt.show()
data.head(10) #top10 values
# Line Plot
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
data.home_goals.plot(kind = "line", color = "r", label = "Home Goals", linewidth = 1, alpha = 0.5, grid = True, linestyle =":")
data.away_goals.plot(color = "b", label = "Away Goals", linewidth=1, alpha=0.5, grid = True, linestyle ="-.")
plt.legend(loc = "upper right")
plt.xlabel("x Axis")
plt.ylabel("y Axis")
plt.title("from 2007 till 2018 Goals in Premier Leauge")
plt.show()
#scatter plot
data.plot(kind ="scatter", x="home_goals",y="away_goals", alpha =0.5, color="green")
plt.xlabel("Home Goals")
plt.ylabel("Away Goals")
plt.title("Home Goals - Away Goals Scatter Plot")
data.home_goals.plot(kind ="hist", bins = 20, align = "mid")
plt.xlabel("Home Goals")
plt.ylabel("Number of Matches")
plt.show()
# *** First Way ***

x = data["home_goals"] == 0
print(len(data[x]))
y = data["home_goals"] == 1
print(len(data[y]))
z = data["home_goals"] == 2
print(len(data[z]))

# *** Second Way***

#data_series = data["home_goals"]
#noGoalList = []
#oneGoalList = []
#twoGoalList = []
#for i in data_series:
#    if i == 0:
#      noGoalList.append(i)
#    elif i == 1:
#        oneGoalList.append(i)
#   elif i == 2:
#        twoGoalList.append(i)
#noGoal = len(noGoalList)
#oneGoal = len(oneGoalList)
#twoGoal = len(twoGoalList)
#print(noGoal)
#print(oneGoal)
#print(twoGoal)

data.away_goals.plot(kind ="hist", bins = 20, align = "mid", color = "red")
plt.xlabel("Away Goals")
plt.ylabel("Number of Matches")
plt.show()
# *** First Way ***

a = data["away_goals"] == 0
print(len(data[a]))
b = data["away_goals"] == 1
print(len(data[b]))
c = data["away_goals"] == 2
print(len(data[c]))

#data_series = data["away_goals"]
#noGoalListAway = []
#oneGoalListAway = []
#twoGoalListAway = []
#for i in data_series:
#    if i == 0:
#       noGoalListAway.append(i)
#    elif i == 1:
#        oneGoalListAway.append(i)
#    elif i == 2:
#        twoGoalListAway.append(i)
#noGoalAway = len(noGoalListAway)
#oneGoalAway = len(oneGoalListAway)
#twoGoalAway = len(twoGoalListAway)
#print(noGoalAway)
#print(oneGoalAway)
#print(twoGoalAway)
#------For Score Counts
#------First Way

def percentage(number1, number2):
    return (number1/number2)*100

home_wins = data["result"] == "H"
away_wins = data["result"] == "A"
draws = data["result"] == "D"
total_score = len(data[home_wins]) + len(data[away_wins]) +len(data[draws])

print("Home wins: ", len(data[home_wins]), "----> %",percentage(len(data[home_wins]), total_score))
print("Away_wins: ", len(data[away_wins]), "----> %",percentage(len(data[away_wins]), total_score))
print("Draws: ", len(data[draws]), "----> %",percentage(len(data[draws]), total_score))
print("Total Score: ", total_score)


#------For Score Counts
#------Second Way
print(data["result"].value_counts(dropna = False))
data.columns
def detectedNoneValue(featureName):
    print(data[featureName].value_counts(dropna = False))

for i in data.columns:
    detectedNoneValue(i)

#Thus, thanks to this function we can see whether there is a Nan value or not.
data.describe()
data.boxplot(column = "home_goals", by = "result")

data.boxplot(column = "away_goals", by = "result")
data2 = data.copy()

total_goals = data.away_goals + data.home_goals
data2["total_goals"] = total_goals
#This is to determine total goals are over 2.5 or under
data2["over_under"] = ["over" if i > 2.5 else "under" for i in total_goals]
boolean = data2.total_goals > 2.5
print("---Over 2.5 Goals---")

data2[boolean]
