# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from collections import Counter

%matplotlib inline



# plotly

import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/WorldCupMatches.csv", encoding = 'utf8')

# clear missing value

data = data.dropna()

data["result"] = ["Win" if rows["Home Team Goals"] > rows["Away Team Goals"] else "Lost" if rows["Home Team Goals"] < rows["Away Team Goals"] else "Draw" for i,rows in data.iterrows()]

#['yes' if v == 1 else 'no' if v == 2 else 'idle' for v in l]
data.info()
data.describe()
data.head()
data.corr()
f,ax = plt.subplots(figsize=(10, 10))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
data.columns
def splitToString(splitStr): 

    array = splitStr.split()

    result = ""

    for each in array:

      result = result + each + "_"

    return result.strip("_")  



data.columns = [splitToString(each.lower()) if (len(each.split()) > 1) else each.lower() for each in data.columns]
data.columns
teamFilter = data.home_team_name == 'Brazil'

yearFilter = data.year == 1998

teamData = data[teamFilter & yearFilter]
teamData.describe()
yearlist = list(tuple(data.year.unique()))

yearlist = [str(i).split('.')[0] for i in yearlist]



year_goals_ratio = []

for i in yearlist :

    yearData = data[data.year == int(i)]

    year_golas_rate = (sum(yearData.home_team_goals) + sum(yearData.away_team_goals))/len(yearData)

    year_goals_ratio.append(year_golas_rate)

    

newData = pd.DataFrame({"yearlist":yearlist,"year_goals_ratio":year_goals_ratio})

newIndex = newData["yearlist"].sort_values(ascending=False).index.values

sorted_data = newData.reindex(newIndex)

plt.figure(figsize = (9,15))

sns.barplot(x=sorted_data.yearlist,y=sorted_data.year_goals_ratio)

plt.xticks(rotation= 90)

plt.xlabel('Year')

plt.ylabel('Goals')

plt.title('Goals per year')

plt.show()

yearlist = list(data.year.unique())

yearlist2 = ["YEAR-"+str(i).split('.')[0] for i in yearlist]

year_goals = []

for i in yearlist :

    yearData = data[data.year == int(i)]

    year_goals_rate = (sum(yearData.home_team_goals+yearData.away_team_goals))/len(yearData)

    year_goals.append(year_goals_rate)

    

newData = pd.DataFrame({"yearlist":yearlist2,"year_goals":year_goals})

newIndex = (newData["year_goals"].sort_values(ascending=False)).index.values

sorted_data = newData.reindex(newIndex)



f,ax = plt.subplots(figsize = (9,15))



ax = sns.barplot(x=sorted_data["year_goals"],y=sorted_data["yearlist"])

ax.set(xlabel='Goals of year', ylabel='Year',title = "Golas of year")

plt.show()
pltData = data[data.year > 2000]

hometeamlist = list(pltData.home_team_name.unique())

home_team_goals_ratio = []

for i in hometeamlist :

    teamFilter = pltData.home_team_name == i

    filterData = pltData[teamFilter]

    goals_ratio = sum(filterData.home_team_goals)/len(filterData)

    home_team_goals_ratio.append(goals_ratio)



newData = pd.DataFrame({"hometeamlist":hometeamlist,"home_team_goals_ratio":home_team_goals_ratio})

newIndex = (newData["home_team_goals_ratio"].sort_values(ascending=False)).index.values

sorted_data = newData.reindex(newIndex)



plt.figure(figsize=(15,10))

sns.barplot(x=sorted_data["hometeamlist"],y=sorted_data["home_team_goals_ratio"])

plt.xticks(rotation=90)

plt.xlabel("Home Team ")

plt.ylabel("Goals Rate")

plt.title("Golas Rate Given home Team")

plt.show()
# using Plotly library

# Creating trace1

plotlyData = data[data.year >1970]

trace1 = go.Scatter(

                    x = plotlyData.year,

                    y = plotlyData.home_team_goals,

                    mode = "lines",

                    name = "Home Team Golas",

                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),

                    text= data.home_team_name)

# Creating trace2

trace2 = go.Scatter(

                    x = plotlyData.year,

                    y = plotlyData.away_team_goals,

                    mode = "lines+markers",

                    name = "Away Team Golas",

                    marker = dict(color = 'rgba(80, 26, 80, 0.8)'),

                    text= plotlyData.away_team_name)

data2 = [trace1, trace2]

layout = dict(title = 'Home Team Golas and Away Team Golas vs Year ',

              xaxis= dict(title= 'Year',ticklen= 5,zeroline= False)

             )

fig = dict(data = data2, layout = layout)

iplot(fig)



fig, ax = plt.subplots()

teamData.home_team_goals.plot(kind='Bar')

ax.set_xticklabels(teamData.away_team_name)

plt.title("Brazil Teams Goal")

plt.show()
goalData = data[yearFilter]

goalData.home_team_goals.plot(kind="line",color="b",label="Home Team Goals",linewidth=1,grid=True)

goalData.away_team_goals.plot(kind="line",color="r",label="Away Team Goals",linewidth=1,grid=True)

plt.legend(loc="upper right")

plt.show()
### VISUAL EXPLORATORY DATA ANALYSIS

teamData.boxplot(column="home_team_goals",by="away_team_goals")

plt.show()
#melted data

melted=pd.melt(frame=teamData.head(),id_vars="away_team_name",value_vars=['away_team_goals','home_team_goals'])

melted
# revers of melted

melted.pivot(index ="away_team_name",columns='variable',values="value")
#Concat dataframe rows

data1 = data.head()

data2 = data.tail()

concatData = pd.concat([data1,data2],axis=0,ignore_index=True)

concatData
data1 = data.home_team_name.head()

data2 = data.home_team_goals.head()

data3 = data.away_team_name.head()

data4 = data.away_team_goals.head()

concatData = pd.concat([data1,data2,data3,data4],axis=1)

concatData
def uniquelist(li) : 

    if len(li) == len(set(li)):

        return li

    else : 

        return list(set(li))



dictionary = {"home_team_name":uniquelist(data["home_team_name"].head(10)),"away_team_name":uniquelist(data["away_team_name"].head(10))}

print(dictionary)
teamData.dtypes

    
teamData["attendance"] = teamData["attendance"].astype("object")

teamData.away_team_name.value_counts()
assert teamData.home_team_name.notnull().all()