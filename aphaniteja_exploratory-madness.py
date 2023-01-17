# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

from matplotlib import pyplot as plt
%matplotlib inline
import matplotlib as mpl
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib import rcParams
#rcParams['figure.figsize'] = 20, 10
plt.style.use('ggplot')
players_2010=pd.read_csv("../input/Players_2010.csv")
players_2011=pd.read_csv("../input/Players_2011.csv")
players_2012=pd.read_csv("../input/Players_2012.csv")
players_2013=pd.read_csv("../input/Players_2013.csv")
players_2014=pd.read_csv("../input/Players_2014.csv")
players_2015=pd.read_csv("../input/Players_2015.csv")
players_2016=pd.read_csv("../input/Players_2016.csv")
players_2017=pd.read_csv("../input/Players_2017.csv")
players=pd.concat([players_2010,players_2011,players_2012,players_2013,players_2014,players_2015,players_2016,players_2017])
players.shape
players.head()
seeds = pd.read_csv('../input/NCAATourneySeeds.csv')
tourney_results = pd.read_csv('../input/NCAATourneyCompactResults.csv')
regular_results = pd.read_csv('../input/RegularSeasonCompactResults.csv')
tourney_results.head()
tourney_results["W-LScore"]=tourney_results["WScore"]-tourney_results["LScore"]
fig=plt.figure(figsize=(10,10))
ax1=fig.add_subplot(2, 2, 1)
tourney_results.groupby("Season")["WScore"].mean().plot(kind="bar",ax=ax1)
ax2=fig.add_subplot(2, 2, 2)
tourney_results.groupby("Season")["LScore"].mean().plot(kind="bar",ax=ax2)
ax3=fig.add_subplot(2, 2, 3)
tourney_results.groupby("Season")["W-LScore"].mean().plot(kind="bar",ax=ax3)
plt.tight_layout()
fig=plt.figure(figsize=(15,10))
(tourney_results
.pipe((sns.violinplot,"data"),x="Season",y="W-LScore"))
fig=plt.figure(figsize=(15,10))
(tourney_results
.pipe((sns.boxplot,"data"),x="Season",y="W-LScore"))
regular_results["W-LScore"]=regular_results["WScore"]-regular_results["LScore"]
fig=plt.figure(figsize=(15,10))
(regular_results
.pipe((sns.boxplot,"data"),x="Season",y="W-LScore")
)
fig=plt.figure(figsize=(15,10))
(regular_results
.pipe((sns.violinplot,"data"),x="Season",y="W-LScore")
)
teams=pd.read_csv("../input/Teams.csv")
len(teams)
regular_results_team=pd.merge(regular_results,teams,how="left",left_on="WTeamID",right_on="TeamID")
regular_results_team.drop(["TeamID","FirstD1Season","LastD1Season"],axis=1,inplace=True)
regular_results_team.rename(columns={"TeamName":"WTeamName"},inplace=True)
regular_results_team=pd.merge(regular_results_team,teams,how="left",left_on="LTeamID",right_on="TeamID")
regular_results_team.drop(["TeamID","FirstD1Season","LastD1Season"],axis=1,inplace=True)
regular_results_team.rename(columns={"TeamName":"LTeamName"},inplace=True)
regular_results_team.head()
fig=plt.figure(figsize=(15,10))
(regular_results_team
.groupby("WTeamName",as_index=False)
["W-LScore"]
.mean()
.pipe((sns.stripplot,"data"),x="WTeamName",y="W-LScore") 
)
team_mean_WLScore=(
regular_results_team
.groupby("WTeamName",as_index=False)
["W-LScore"]
.mean()
)
team_mean_WLScore["quality"]=(
pd.cut(team_mean_WLScore["W-LScore"],bins= 5,labels=["bad", "medium1","medium2" ,"good","great"])
)
team_mean_WLScore.head()
fig=plt.figure(figsize=(20,8))
ax1=fig.add_subplot(1,2,1)
ax1=(team_mean_WLScore
.query("quality=='great'")
.pipe((sns.stripplot,"data"),x="WTeamName",y="W-LScore",ax=ax1) 
)
ax1.set_title("The Great")

ax2=fig.add_subplot(1,2,2)
ax2=(team_mean_WLScore
.query("quality=='bad'")
.pipe((sns.stripplot,"data"),x="WTeamName",y="W-LScore",ax=ax2) 
)
ax2.set_title("The bad")









