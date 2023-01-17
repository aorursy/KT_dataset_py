# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/marble-racing/marbles.csv")
df.head()
df.site.unique()
df.race.unique()
df["Points"] = df.pole.map({

    "P1": 25, 

    "P2": 18,

    "P3": 15,

    "P4": 12,

    "P5": 10,

    "P6": 8,

    "P7": 6,

    "P8": 4,

    "P9": 2,

    "P10": 1,

    "P11": 0,

    "P12": 0,

    "P13": 0,

    "P14": 0,

    "P15": 0,

    "P16": 0

})

df.head()
Player_stats = df[["race", "marble_name", "Points", "avg_time_lap", "points"]]

Orangin = Player_stats[Player_stats.marble_name == "Anarchy"].reset_index()

Orangin
player_wise = Player_stats.groupby('marble_name').mean().reset_index()

# len(player_wise)

print("Qualifier Races ---\n" + "median Score Across all rounds: " + str(np.median(player_wise.Points.values)) + "\nmean Score Across all rounds: " + str(np.average(player_wise.Points.values)) + 

     "\nMarbulaOne Races ---\n"+ "median Score Across all rounds: " + str(np.median(player_wise.points.values)) + "\nmean Score Across all rounds: " + str(np.average(player_wise.points.values)))
x = player_wise.marble_name.values

y1 = player_wise.Points.values

y2 = player_wise.avg_time_lap.values

plt.figure(figsize=(25, 6))

plt.xlabel("Players")

plt.ylabel("Average performance among all Qualifier Rounds rounds")

plt.bar(x, y1)

plt.figure(figsize=(25, 7))

sns.set_style('whitegrid')

graph = sns.boxplot(data=Player_stats, x='marble_name', y='Points')

graph.axhline(np.median(player_wise.Points.values))

plt.xlabel("Players")

plt.ylabel("Average Performance among all Qualifier Rounds")
plt.figure(figsize=(25, 7))

sns.set_style('whitegrid')

graph = sns.boxplot(data=Player_stats, x='marble_name', y='points')

graph.axhline(np.median(player_wise.points.values))

plt.xlabel("Players")

plt.ylabel("Average performance among all rounds")
player_wise_sum = Player_stats.groupby('marble_name').sum().reset_index()

qualifier_score = player_wise_sum.Points.values

race_score = player_wise_sum.points.values

players = player_wise_sum.marble_name.values

colors = ['lightskyblue', 'gold', 'lightcoral', 'gainsboro', 'royalblue', 'lightpink', 'darkseagreen', 'sienna',

          'khaki', 'gold', 'violet', 'yellowgreen']



plt.figure(figsize=(24, 10))



ax1 = plt.subplot(1, 2, 1)

ax1 = plt.pie(qualifier_score, autopct='%0.f%%', pctdistance=0.8, colors=colors, startangle=345, shadow=True, labels=players)

plt.title("Qualifier Points achieved per Player")



ax2 = plt.subplot(1, 2, 2)

ax2 = plt.pie(race_score, autopct='%0.f%%', pctdistance=0.8, colors=colors, startangle=345, shadow=True, labels=players)

plt.title("Race Points achieved per Player")



plt.subplots_adjust(left=0.1, right=0.90)
team_wise = df[['team_name', 'Points', 'points']]

team_wise_score = team_wise.groupby('team_name').sum().reset_index()

team_wise_score.head()
plt.figure(figsize=(24, 8))

sns.set_style('whitegrid')



ax1 = plt.subplot(1, 2, 1)

ax1 = sns.barplot(data=team_wise_score, x='team_name', y='Points')

plt.xticks(rotation=75)

ax1.set_yticks(range(0, 111, 10))

plt.xlabel("Team Name")

plt.title('Qualifier Round Total Score per Team')



ax2 = plt.subplot(1, 2, 2)

ax2 = sns.barplot(data=team_wise_score, x='team_name', y='points')

plt.xticks(rotation=75)

ax2.set_yticks(range(0, 111, 10))

plt.xlabel("Team Name")

plt.title('Race Round Total Score per Team')



plt.subplots_adjust(left=0.02, right=0.98)