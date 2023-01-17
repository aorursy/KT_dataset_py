 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt   # ploting graphs
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import preprocessing
cities = pd.read_csv('../input/WCities.csv')
gamecities = pd.read_csv('../input/WGameCities.csv')
tourneycompactresults = pd.read_csv('../input/WNCAATourneyCompactResults.csv')
tourneyseeds = pd.read_csv('../input/WNCAATourneySeeds.csv')
tourneyslots = pd.read_csv('../input/WNCAATourneySlots.csv')
regseasoncompactresults = pd.read_csv('../input/WRegularSeasonCompactResults.csv')
seasons = pd.read_csv('../input/WSeasons.csv')
teamspellings = pd.read_csv('../input/WTeamSpellings.csv',encoding="latin-1")
teams = pd.read_csv('../input/WTeams.csv')
#print(cities.head())  # cityid, cityname and state code
print('file -gamecities' , 1)
print(gamecities.head()) # 
print('file - tourneycompactresults' ,2)
print(tourneycompactresults.head())
print('file - tourneyseeds ' ,3)
print(tourneyseeds.head())
print('file -tourneyslots' ,4)
print(tourneyslots.head())
print('file -regseasoncompactresults' ,5)
print(regseasoncompactresults.head())
print('file-seasons ' ,6)
print(seasons.head())
print('file-teamspellings ' ,7)
print(teamspellings.head())
print('file-teams ' ,8)
print(teams.head())
# counting number of wins home and away
print(regseasoncompactresults.shape)
result=regseasoncompactresults["WLoc"].value_counts()
h=result[0]
a=result[1]
n=result[2]
#print(h/regseasoncompactresults.shape[0],a/regseasoncompactresults.shape[0],n/regseasoncompactresults.shape[0])
print(regseasoncompactresults["WLoc"].value_counts('H'))
regseasoncompactresults["WLoc"].value_counts().plot.pie()
#tourneycompactresults
print(tourneycompactresults["WLoc"].value_counts('H'))
#print(tourneycompactresults["WLoc"].value_counts().plot(kind='barh'))
tourneycompactresults["WLoc"].value_counts().plot.pie()
#regseasoncompactresults["WTeamID"].value_counts()

#top 30
regseasoncompactresults["WTeamID"].value_counts()[:30].plot(kind='bar')
#bottom 30 teams
regseasoncompactresults["LTeamID"].value_counts()[:30].plot(kind='bar')
regseasoncompactresults.plot(kind="scatter", x="LScore", y="WScore")
