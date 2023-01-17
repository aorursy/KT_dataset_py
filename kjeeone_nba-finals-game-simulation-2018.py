

import pandas as pd 
import random as rnd
import numpy as np 
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
gdf = pd.read_csv('../input/nba.games.stats.csv')
#Create Dataframes for Golden State & Cleveland 
gswdf = gdf[gdf.Team == 'GSW']
cldf = gdf[gdf.Team == 'CLE']
#Convert date to pandas date time, and remove all dates not from the most recent season
gswdf.Date = gswdf.Date.apply(lambda x: pd.to_datetime(x, format='%Y-%m-%d', errors='ignore'))
gswdf = gswdf[gswdf['Date'] > pd.to_datetime('20171001', format='%Y%m%d', errors='ignore')]

cldf.Date = cldf.Date.apply(lambda x: pd.to_datetime(x, format='%Y-%m-%d', errors='ignore'))
cldf = cldf[cldf['Date'] > pd.to_datetime('20171001', format='%Y%m%d', errors='ignore')]
#Compare the points of the two teams over the course of the season (GS == Blue, CLE == Orange) 
gswdf.TeamPoints.hist()
cldf.TeamPoints.hist()
#Compare the points of scored against the two teams over the course of the season (GS == Blue, CLE == Orange) 
gswdf.OpponentPoints.hist()
cldf.OpponentPoints.hist()
#Compute baseline statistics for the teams to be used in the simulation: Mean & STD for points scored and points scored against 
gswmeanpts = gswdf.TeamPoints.mean()
clmeanpts = cldf.TeamPoints.mean()
gswsdpts = gswdf.TeamPoints.std()
clsdpts = cldf.TeamPoints.std()

gswmeanopp = gswdf.OpponentPoints.mean()
clmeanopp = cldf.OpponentPoints.mean()
gswsdopp = gswdf.OpponentPoints.std()
clsdopp = cldf.OpponentPoints.std()

print("Golden State Points Mean ", gswmeanpts)
print("Golden State Points SD ", gswsdpts)
print("Cleveland Points Mean ", clmeanpts)
print("Cleveland Points SD ", clsdpts)

print("Golden State OppPoints Mean ", gswmeanopp)
print("Golden State OppPoints SD ", gswsdopp)
print("Cleveland OppPoints Mean ", clmeanopp)
print("Cleveland OppPoints SD ", clsdopp)
#Generate single games simulation code. This averages the points scored by each team with the points scored against the other team. 
#I use the gaussian function from the random module. This randomly samples from a distribution with the mean and std that you give the function
#If the random samples for GSWScore is greater than those of CLScore then it returns 1 
def gameSim():
    GSWScore = (rnd.gauss(gswmeanpts,gswsdpts)+ rnd.gauss(clmeanopp,clsdopp))/2
    CLScore = (rnd.gauss(clmeanpts,clsdpts)+ rnd.gauss(gswmeanopp,gswsdopp))/2
    if int(round(GSWScore)) > int(round(CLScore)):
        return 1
    elif int(round(GSWScore)) < int(round(CLScore)):
        return -1
    else: return 0
#Sample Game Sim 1 = GSW Win, -1 = CLE Win, 0 = Tie ( I know Ties Can't really happen, but for this case it gives us complete probabilities) 
gameSim()
#This runs the gameSim() code multiple times. In simulation you evaluate randomness over time to reach a limit.
# Running this function thousands of times should get us close to the expected win probability for each team.
# ns = number of simulations 
def gamesSim(ns):
    #gamesout = []
    team1win = 0
    team2win = 0
    tie = 0
    for i in range(ns):
        gm = gameSim()
        #gamesout.append(gm)
        if gm == 1:
            team1win +=1 
        elif gm == -1:
            team2win +=1
        else: tie +=1 
    print('GSW Win ', team1win/(team1win+team2win+tie),'%')
    print('CLE Win ', team2win/(team1win+team2win+tie),'%')
    print('Tie ', tie/(team1win+team2win+tie), '%')
    return #gamesout
gamesSim(10)
gamesSim(100)
gamesSim(1000)
gamesSim(10000)