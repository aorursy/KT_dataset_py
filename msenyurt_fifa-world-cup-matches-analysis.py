#-*-coding:utf-8-*-
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
data = pd.read_csv('../input/WorldCupMatches.csv')
data.head()
data.tail(10)
data.columns
data.info()
data.dropna(inplace=True)
data.info()
data.tail(10)
#data['Win conditions'].fillna('empty' , inplace=True)
data['Win conditions'] == ' '
data.Year = data.Year.astype('int')
data['Home Team Goals'] = data['Home Team Goals'].astype('int')
data['Away Team Goals'] = data['Away Team Goals'].astype('int')
data['Half-time Home Goals'] = data['Half-time Home Goals'].astype('int')
data['Half-time Away Goals'] = data['Half-time Away Goals'].astype('int')
data.Attendance = data.Attendance.astype('int')
data.info()
match = data.Year.value_counts(dropna=False)
match
data.Year.plot(kind='hist' , bins=60 , figsize=(14,14))
plt.axis([1930,2014,0,80])
plt.xlabel('Years')
plt.ylabel('Match')
plt.show()
year=[]
[year.append(i) for i in data['Year'] if i not in year]
home_team_goal = [np.sum(data[data.Year == i]['Home Team Goals']) for i in year]
away_team_goal =[np.sum(data[data.Year == i]['Away Team Goals']) for i in year]
total_goal = [home_team_goal[i]+away_team_goal[i] for i in range(len(home_team_goal))]

ind = np.arange(len(year))
p1 = plt.bar(ind, home_team_goal, 0.4 , yerr=np.ones(len(year)))
p2 = plt.bar(ind, away_team_goal, 0.4 , bottom=home_team_goal,yerr=np.ones(len(year)))
plt.ylabel('Total Goal')
plt.xticks(ind, year , rotation=90)
#plt.yticks(np.arange(0, 81, 10))
plt.legend((p1[0], p2[0]), ('Home Team Goal', 'Away Team Goal'))
plt.show()
number_of_match = [data[data.Year == i].Year.count() for i in year]
rate = [total_goal[i]/number_of_match[i] for i in range(len(year))]

fig, axs = plt.subplots(figsize=(7,7))
axs.scatter(year, rate)
plt.ylabel('Rate')
plt.xlabel('Year')
plt.show()
audience = np.sum(data.groupby('Year')['Attendance'])
#total_audience = np.sum(data['Attendance'])
audience_of_year = [audience[i] for i in year]
#one of the two different method if you want to choose
#audience_of_year = [np.sum(data[data.Year == i]['Attendance']) for i in year]
audience_rate = [int(audience_of_year[i]/number_of_match[i]) for i in range(len(year))]

plt.plot(year, audience_rate, 'o-')
plt.ylabel('average of attendance')
plt.xlabel('Year')
plt.show()
team = []
data = data.replace('Germany FR', 'Germany')
data = data.replace('German DR', 'Germany')
data = data.replace('IR Iran', 'Iran')
data = data.replace('Soviet Union', 'Russia')
data = data.replace('Czechoslovakia', 'Czech Republic')
data = data.replace("C�te d'Ivoire", "Cote d'Ivoire")
data = data.replace('rn">Bosnia and Herzegovina', 'Bosnia and Herzegovina')
data = data.replace('rn">Trinidad and Tobago', 'Trinidad and Tobago')
data = data.replace('rn">Republic of Ireland', 'Republic of Ireland')
data = data.replace('rn">United Arab Emirates', 'United Arab Emirates')
data = data.replace('rn">Serbia and Montenegro', 'Serbia and Montenegro')
[team.append(i) for i in data['Home Team Name'] if i not in team]
[team.append(i) for i in data['Away Team Name'] if i not in team]
times = np.zeros(len(team))
for i in year:
    home = data[data.Year == i]['Home Team Name']
    away = data[data.Year == i]['Away Team Name']
    teams = []
    [teams.append(j) for j in home if j not in teams]
    [teams.append(j) for j in away if j not in teams]
    for j in teams:
        times[team.index(j)] += 1
        
#the another way but it is not mine @Pavan Raj
#home = data[["Year","Home Team Name"]]
#home.columns = ["year","team"]
#away = data[["Year","Away Team Name"]]
#away.columns = ["year","team"]

#home_away = pd.concat([home,away],axis=0)
#top_ten = home_away.groupby(["year","team"]).count().reset_index()
#top_ten = top_ten["team"].value_counts().reset_index()
fig, ax = plt.subplots(figsize=(22,13))
plt.bar(np.arange(len(team)), times)
plt.xticks(np.arange(len(team)), team , rotation=90)
plt.yticks(np.arange(0, 22, 2))
plt.grid()
plt.show()






