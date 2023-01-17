# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt
import seaborn as sns

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/WorldCupMatches.csv')

data.columns
data.head(10)
f,ax = plt.subplots(figsize=(13, 13))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax,cmap="Greens")
plt.show()
att1 = data.groupby("Year")["Attendance"].mean().reset_index()
att1["Year"] = att1["Year"].astype(int)
plt.figure(figsize=(12,6))
ax = sns.pointplot(att1["Year"],att1["Attendance"],color="k")
ax.set_facecolor("w")
plt.grid(True,color="grey",alpha=.3)
plt.title("Average attendence by year",color='b')
plt.show()
data.Attendance.max()
data[data['Attendance']==data['Attendance'].max()]

gh = data[["Year","Home Team Goals"]]
gh.columns = ["year","goals"]
gh["type"] = "Home Team Goals"

ga = data[["Year","Away Team Goals"]]
ga.columns = ["year","goals"]
ga["type"] = "Away Team Goals"

gls = pd.concat([ga,gh],axis=0)
plt.figure(figsize=(15,8))
sns.violinplot(gls["year"],gls["goals"],
               hue=gls["type"],split=True,inner="point",palette="Set1",scale = "count",saturation = 1)
plt.grid(True)
plt.title("Home and away goals by year",color='b')
plt.show()
#Team interactions comparator

def teams_performance(team1,team2):
    
    list_teams = [team1,team2]
    data["total_goals"] = data["Home Team Goals"] + data["Away Team Goals"]
    df_new = data[(data["Home Team Name"].isin(list_teams))]
    df_new = df_new[df_new["Away Team Name"].isin(list_teams)]
   
    print ("Total Matches       : ", df_new.shape[0])
    print ("Match Years         : ", df_new["Year"].unique().tolist())
    print ("Stadiums played     : ", df_new["Stadium"].unique().tolist(),"\n")
    print ("Match Cities        : ", df_new["City"].unique().tolist())
    print ("Average Attendance  : ", np.around(df_new["Attendance"].mean(),0) , "per game.")
    print ("Average total goals : ", np.around(df_new["total_goals"].mean(),2), "goals per game.")
   
teams_performance("Turkey","Brazil")
teams_performance("Spain","Germany")
teams_performance("USA","England")
