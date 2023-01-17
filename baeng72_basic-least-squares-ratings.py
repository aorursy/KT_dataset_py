# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


seasonsData = pd.read_csv("../input/RegularSeasonCompactResults.csv")
tournamentsData = pd.read_csv("../input/NCAATourneyCompactResults.csv")
sampleSubmission = pd.read_csv("../input/SampleSubmissionStage1.csv")
seasonsData.head()
#just want some basic data, with encoding
games = pd.DataFrame()
games['Season'] = seasonsData['Season']
games['Winner'] = seasonsData['WTeamID']
games['Loser'] = seasonsData['LTeamID']
games['WLoc'] = seasonsData['WLoc']
#margin of victory
games['MOV'] = seasonsData['WScore'] - seasonsData['LScore']
games.head()
tournamentsGames = pd.DataFrame()
tournamentsGames['Season'] = tournamentsData['Season']
tournamentsGames['Winner'] = tournamentsData['WTeamID']
tournamentsGames['Loser'] = tournamentsData['LTeamID']
tournamentsGames['MOV'] = tournamentsData['WScore'] - tournamentsData['LScore']
tournamentsGames.head()
def getSeasonGames(games,season):
    seasonGames = pd.DataFrame()
    seasonGames = games.loc[games['Season']== season].copy()
    return seasonGames
#Build "Massey" style, least squares matrix and vector for a season
def getSeasonMatrixVector(games,num_teams):
    #This matrix, M, is really the LHS of the NORMAL equations of OLS: X'X (where X' is the transpose)
    #p is the RHS, X'y    
    M = np.zeros([num_teams,num_teams])
    p = np.zeros(num_teams)
    for row in games.itertuples():
        wid = row.Winner
        lid = row.Loser
        mov = row.MOV
        M[wid,wid] += 1
        M[lid,lid] += 1
        M[wid,lid] -=1
        M[lid,wid] -= 1
        p[wid] += mov
        p[lid] -= mov
    #columns will be a linear combination, i.e. matrix is singular, quick fix.
    M[num_teams-1,]=1
    p[num_teams-1]=0
    
    return M,p
        
from sklearn.preprocessing import LabelEncoder
seasons = list(set(games['Season']))
seasonsRatings = {}
for season in range(len(seasons)):
    seasonGames = getSeasonGames(games,seasons[season])
    #Encode teams, so we can index them from 0 to N (I don't know if teams change, year to year, so do it every year)
    le = LabelEncoder()
    teams = pd.DataFrame()
    teams['Team'] = pd.concat([seasonGames['Winner'],seasonGames['Loser']])
    le.fit(teams['Team'])
    num_teams = len(le.classes_)
    seasonGames['Winner']=le.transform(seasonGames['Winner'])
    seasonGames['Loser'] = le.transform(seasonGames['Loser'])
    M,p = getSeasonMatrixVector(seasonGames,num_teams)
    #do this brute force way...could be done with decomposition, gaussian elimination, yo qué sé?
    b = np.linalg.inv(M).dot(p)
    ratings = dict(zip(le.classes_,b))
    seasonsRatings[seasons[season]] = ratings
    
    
    
def getTournamentResults(tournamentGames,ratings):
    x = np.zeros([tournamentGames.shape[0]*2,2])
    idx = 0
    for tournamentGame in tournamentGames.itertuples():        
        wid = tournamentGame.Winner
        lid = tournamentGame.Loser
        winRat = ratings[wid]
        loseRat = ratings[lid]
        pred = winRat - loseRat
        predCopy = pred
        win = 0
        if predCopy > 0:
            win = 1
        if predCopy < 0:
            predCopy = -predCopy
        x[idx,0] = predCopy
        x[idx,1] = win
        predCopy = -predCopy
        loss = 0
        if pred < 0:
            loss = 1
        x[idx + tournamentGames.shape[0],0] = predCopy
        x[idx + tournamentGames.shape[0],1] = loss
        idx+=1
    return x
tournamentsSeasons = list(set(tournamentsData['Season']))
tournamentsSeasons.remove(2015)
tournamentsSeasons.remove(2014)
tournamentsSeasons.remove(2016)
tournamentsSeasons.remove(2017)
tournamentResults = np.zeros([0,2])
for season in range(len(tournamentsSeasons)):
    tournamentGames = getSeasonGames(tournamentsGames,tournamentsSeasons[season])
    #ratings
    ratings = seasonsRatings[tournamentsSeasons[season]]
    x = getTournamentResults(tournamentGames,ratings)    
    tournamentResults = np.concatenate((tournamentResults,x),axis=0) 
#empirical cdf, thanks stack exchange!
def ecdf(x):
    xs = np.sort(x)
    ys = np.arange(1, len(xs)+1)/float(len(xs))
    return xs, ys
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
xs,ys = ecdf(tournamentResults[:,0])
plt.plot(xs, ys, label="handwritten", marker=">", markerfacecolor='none')
mu = 0
sd = 12#12 seems to be about right.
x = np.linspace(-40,40, 80)
CY = np.cumsum(mlab.normpdf(x,mu,sd))

plt.plot(x,CY)
plt.show()
#Borrowed from Basic Starter Kernel. Thanks!
def get_year_t1_t2(ID):
    """Return a tuple with ints `year`, `team1` and `team2`."""
    return (int(x) for x in ID.split('_'))

from scipy.stats import norm
X_test = np.zeros(shape=(sampleSubmission.shape[0], 1))
for ii, row in sampleSubmission.iterrows():
    season, t1, t2 = get_year_t1_t2(row.ID)
    ratings = seasonsRatings[season]
    wrat = ratings[t1]
    lrat = ratings[t2]
    pred = wrat - lrat
    prob = norm(0,12).cdf(pred)
    X_test[ii,0] = prob
sampleSubmission.Pred = X_test
sampleSubmission.head()
sampleSubmission.to_csv('submissionlq.csv', index=False)
