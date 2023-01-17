# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder

seasonsData = pd.read_csv("../input/WRegularSeasonCompactResults.csv")
tournamentsData = pd.read_csv("../input/WNCAATourneyCompactResults.csv")
sampleSubmission = pd.read_csv("../input/WSampleSubmissionStage1.csv")
seasonsData.head()
#just want some basic data, teams, margin of victory, location
games = pd.DataFrame()
games['Season'] = seasonsData['Season']
games['WTeamID'] = seasonsData['WTeamID']
games['LTeamID'] = seasonsData['LTeamID']
games['MOV'] = seasonsData['WScore'] - seasonsData['LScore']
games['WLoc'] = seasonsData['WLoc']
games.head()
#get tourney results in similar format
tournamentsGames = pd.DataFrame()
tournamentsGames['Season'] = tournamentsData['Season']
tournamentsGames['Winner'] = tournamentsData['WTeamID']
tournamentsGames['Loser'] = tournamentsData['LTeamID']
tournamentsGames['MOV'] = tournamentsData['WScore'] - tournamentsData['LScore']
tournamentsGames.head()
#the paper describe how each team has a w, u, and v with lengths equal to the number of games.
#these vectors are concatenated respectively to give a u and v matrix, which is manipulated in various models
#there's also an x vector which is 1 for home ground, 0 otherwize
def buildUVWMatrices(games,num_teams):
    num_games = games.shape[0]
    U = np.zeros([num_games,num_teams]) #U is the 1..n u column vectors
    V = np.zeros([num_games,num_teams]) #V is the 1..n v column vectors
    W = np.zeros([num_games,num_teams]) #W is the sum of U and V
    y = np.zeros(num_games)             #y contains the margins of victory
    x = np.zeros(num_games)             #x contains 1 for home, zero otherwise
    idx = 0
    for row in games.itertuples():
        #convert WTeamID,LTeamID to HomeTeam, AwayTeam
        wid = row.WTeamID
        lid = row.LTeamID
        wloc = row.WLoc
        if wloc == 'H':      
            homeTeam = wid
            awayTeam = lid
            mov = row.MOV    
            xi=1
        elif wloc=='N':
            homeTeam = wid
            awayTeam = lid
            mov = row.MOV
            xi=0
        else:
            homeTeam = lid
            awayTeam = wid
            mov = -row.MOV
            xi=1
        
        if xi == 1:
            U[idx,homeTeam] = 1    #u = 1 if x = 1, else 0
        else:
            V[idx,homeTeam]=1      #v = 1 if x = 0, else 0
        V[idx,awayTeam] = -1       #v = -1 for away team
        y[idx] = mov    
        x[idx]=xi         
        idx += 1
    W = U + V
    return U,V,W,y,x
#function to return a season's games. I suck at python, so I'm sure there's a better way to do this.
def getSeasonGames(games,season):
    seasonGames = pd.DataFrame()
    seasonGames = games.loc[games['Season']== season].copy()
    return seasonGames
#Model 1 is least-squares ratings, like Massey's and Stefani's, etc.
#build model 1 yijk = Bi - Bj
def buildModel1(games,num_teams):
    
    num_games = games.shape[0]
    U,V,W,y,x = buildUVWMatrices(games,num_teams)
    W.resize([num_games+1,num_teams])
    y.resize(num_games+1)
    W[num_games,]=1 # avoid singular matrix
    y[num_games]=0
    return W,y    
#I could put this in a function, as I resuse it, but....
#go through seasons data and calculate the ratings using Model 1. 
seasons = list(set(games['Season']))
seasonsRatingsModel1 = {}
for season in range(len(seasons)):
    seasonGames = getSeasonGames(games,seasons[season])
    #Encode teams, so we can index them from 0 to N (I don't know if teams change, year to year, so do it every year)
    le = LabelEncoder()
    teams = pd.DataFrame()
    teams['Team'] = pd.concat([seasonGames['WTeamID'],seasonGames['LTeamID']])
    le.fit(teams['Team'])
    num_teams = len(le.classes_)
    seasonGames['WTeamID']=le.transform(seasonGames['WTeamID'])
    seasonGames['LTeamID'] = le.transform(seasonGames['LTeamID'])
    X,y = buildModel1(seasonGames,num_teams)
    #Calculate X'X and X'y (normal equations)
    M = X.T.dot(X)
    p = X.T.dot(y)
    b = np.linalg.inv(M).dot(p)
    ratings = dict(zip(le.classes_,b))
    seasonsRatingsModel1[seasons[season]] = ratings
#get results from a list of tournament games using given ratings.
#Tourney games have no home advantage (as normally understood), so even though ratings were calculated 
#with home advantage, we're not using it for tourney predictions.
def getTournamentResults(tournamentGames,ratings):
    x = np.zeros(tournamentGames.shape[0]*2)
    idx = 0
    for tournamentGame in tournamentGames.itertuples():        
        wid = tournamentGame.Winner
        lid = tournamentGame.Loser
        winRat = ratings[wid]
        loseRat = ratings[lid]
        pred = winRat - loseRat
        predCopy = pred        
        if predCopy < 0:
            predCopy = -predCopy
        x[idx] = predCopy
        predCopy = -predCopy
        x[idx + tournamentGames.shape[0]] = predCopy        
        idx+=1
    return x


#go through all tournaments, except 2014-2017 and predict them from that season's ratings.
#keep a copy of the predictions so we can work out the gaussian of predicted outcome.
tournamentsSeasons = list(set(tournamentsData['Season']))
tournamentsSeasons.remove(2015)
tournamentsSeasons.remove(2014)
tournamentsSeasons.remove(2016)
tournamentsSeasons.remove(2017)
tournamentResults = np.zeros(0)
for season in range(len(tournamentsSeasons)):
    tournamentGames = getSeasonGames(tournamentsGames,tournamentsSeasons[season])
    #ratings
    ratings = seasonsRatingsModel1[tournamentsSeasons[season]]
    x = getTournamentResults(tournamentGames,ratings)    
    tournamentResults = np.concatenate((tournamentResults,x),axis=0)    
#a simple empirical cdf I found on stack-exchange, thanks!
def ecdf(x):
    xs = np.sort(x)
    ys = np.arange(1, len(xs)+1)/float(len(xs))
    return xs, ys
#plot out ecdf and a normal curve that closely matches 
#(it might take a few guesses to get a close match if doing it by eye)
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
xs,ys = ecdf(tournamentResults)
plt.plot(xs, ys, label="handwritten", marker=">", markerfacecolor='none')
mu = 0
sd = 16#here's one I prepared earlier.
x = np.linspace(-40,40, 80)
CY = np.cumsum(mlab.normpdf(x,mu,sd))

plt.plot(x,CY)
plt.show()
#Borrowed from Basic Starter Kernel by Julia Elliot. Thanks !
def get_year_t1_t2(ID):
    """Return a tuple with ints `year`, `team1` and `team2`."""
    return (int(x) for x in ID.split('_'))
from scipy.stats import norm
#what is says on the box. pass in submission data, dictionary of ratings and a standard deviation.
def predictSubmission(submission,modelRatings,sd):
    X_test = np.zeros(shape=(submission.shape[0], 1))
    for ii, row in submission.iterrows():
        season, t1, t2 = get_year_t1_t2(row.ID)
        ratings = modelRatings[season]
        wrat = ratings[t1]
        lrat = ratings[t2]
        pred = wrat - lrat
        prob = norm(0,sd).cdf(pred)
        X_test[ii,0] = prob
        submission.Pred = X_test
#I snaffled this code from Kaggle user Caudwell (and made some minor adjustments), thanks!
from sklearn.metrics import log_loss
import pandas as pd
def logLoss(submission):
    df = sampleSubmission.copy()#read_csv("e:\Kaggle\March 2018\\NCAAMEN\submissionlqx.csv")
    df['Season'] = submission['ID'].apply(lambda x: int(x[0:4]))
    df['team1id'] = submission['ID'].apply(lambda x: int(x[5:9]))
    df['team2id'] = submission['ID'].apply(lambda x: int(x[10:14]))
    results = tournamentsData.copy()# pd.read_csv("e:\Kaggle\March 2018\\NCAAMEN\\NCAATourneyCompactResults.csv")
    #results = results.loc[results['DayNum'] > 135].copy()
    results['team1id'] = results[['WTeamID','LTeamID']].min(axis=1)
    results['team2id'] = results[['WTeamID','LTeamID']].max(axis=1)
    df = pd.merge(df, results, how='inner', on=['Season','team1id','team2id'])
    df['result'] = (df['WTeamID'] == df['team1id']).astype(int)
    print(log_loss(df['result'], df['Pred']))
#predict using basic model/model 1
predictSubmission(sampleSubmission,seasonsRatingsModel1,15)
#calculate LogLoss for basic model with estimated sd.
logLoss(sampleSubmission)
#predict using basic model/model 1
predictSubmission(sampleSubmission,seasonsRatingsModel1,13)
#calculate LogLoss for basic model with 13 sd.
logLoss(sampleSubmission)
#predict using basic model/model 1
predictSubmission(sampleSubmission,seasonsRatingsModel1,11)
#calculate LogLoss for basic model with 11 sd.
logLoss(sampleSubmission)
#predict using basic model/model 1
predictSubmission(sampleSubmission,seasonsRatingsModel1,9)
#calculate LogLoss for basic model with estimated sd.
logLoss(sampleSubmission)
#Model 2 yijk = lambda + Bi - Bj (lambda = 1 if home,0 otherwise )
# this can be expressed as y = (x,W)
def buildModel2(games,num_teams):
    num_games = games.shape[0]
    U,V,W,y,x = buildUVWMatrices(games,num_teams)
    W = np.insert(W,0,values=x,axis=1)
    #W.reshape([num_games,num_teams+1])
    W.resize([num_games+1,num_teams+1])
    y.resize(num_games+1)
    W[num_games,]=1
    y[num_games]=0
    return W,y
#calculate ratings using regular season results for each season.
seasons = list(sorted(set(games['Season'])))
seasonsRatingsModel2 = {}
for season in range(len(seasons)):
    seasonGames = getSeasonGames(games,seasons[season])
    #Encode teams, so we can index them from 0 to N (I don't know if teams change, year to year, so do it every year)
    le = LabelEncoder()
    teams = pd.DataFrame()
    teams['Team'] = pd.concat([seasonGames['WTeamID'],seasonGames['LTeamID']])
    le.fit(teams['Team'])
    num_teams = len(le.classes_)
    seasonGames['WTeamID']=le.transform(seasonGames['WTeamID'])
    seasonGames['LTeamID'] = le.transform(seasonGames['LTeamID'])
    X,y = buildModel2(seasonGames,num_teams)
    #Calculate X'X and X'y (normal equations)
    M = X.T.dot(X)
    p = X.T.dot(y)
    b = np.linalg.inv(M).dot(p)
    homeCourt = b[0]
    bNew=np.delete(b,0)#remove homecourt from ratings list
    print("Season=",seasons[season],"HomeCourt=",homeCourt)
    ratings = dict(zip(le.classes_,bNew))
    
    seasonsRatingsModel2[seasons[season]] = ratings
#predict tourney results using Model 2
tournamentsSeasons = list(set(tournamentsData['Season']))
tournamentsSeasons.remove(2015)
tournamentsSeasons.remove(2014)
tournamentsSeasons.remove(2016)
tournamentsSeasons.remove(2017)
tournamentResultsModel2 = np.zeros(0)
for season in range(len(tournamentsSeasons)):
    tournamentGames = getSeasonGames(tournamentsGames,tournamentsSeasons[season])
    #ratings
    ratings = seasonsRatingsModel2[tournamentsSeasons[season]]
    x = getTournamentResults(tournamentGames,ratings)    
    tournamentResultsModel2 = np.concatenate((tournamentResultsModel2,x),axis=0)   
#plot curve to see what standard deviation this model has.
xs,ys = ecdf(tournamentResultsModel2)
plt.plot(xs, ys, label="handwritten", marker=">", markerfacecolor='none')
mu = 0
sd = 15#here's one I prepared earlier
x = np.linspace(-40,40, 80)
CY = np.cumsum(mlab.normpdf(x,mu,sd))

plt.plot(x,CY)
plt.show()
#predict tourney using model 2 ratings with estimated standard deviation.
predictSubmission(sampleSubmission,seasonsRatingsModel2,15)#no home advantage when predicting tourney games
#calculate log loss for model 2
logLoss(sampleSubmission)
#predict tourney using model 2 ratings with estimated standard deviation.
predictSubmission(sampleSubmission,seasonsRatingsModel2,13)#no home advantage when predicting tourney games
#calculate log loss for model 2
logLoss(sampleSubmission)
#predict tourney using model 2 ratings with estimated standard deviation.
predictSubmission(sampleSubmission,seasonsRatingsModel2,11)#no home advantage when predicting tourney games
#calculate log loss for model 2
logLoss(sampleSubmission)
#predict tourney using model 2 ratings with estimated standard deviation.
predictSubmission(sampleSubmission,seasonsRatingsModel2,9)#no home advantage when predicting tourney games
#calculate log loss for model 2
logLoss(sampleSubmission)
#predict tourney using model 2 ratings with estimated standard deviation.
predictSubmission(sampleSubmission,seasonsRatingsModel2,7)#no home advantage when predicting tourney games
#calculate log loss for model 2
logLoss(sampleSubmission)
#Model 3 yijk = Ai - Bj (if at home), Bi - Bj (if neutral)
def buildModel3(games,num_teams):
    num_games = games.shape[0]
    U,V,W,y,x = buildUVWMatrices(games,num_teams)
    X = np.concatenate((U,V),axis=1)
    X.resize([num_games+1,num_teams*2])
    y.resize(num_games+1)
    X[num_games,]=1
    y[num_games]=0
    return X,y
    
#calculate ratings for Model 3 for regular seasons results
seasons = list(sorted(set(games['Season'])))
seasonsRatingsModel3 = {}
for season in range(len(seasons)):
    seasonGames = getSeasonGames(games,seasons[season])
    #Encode teams, so we can index them from 0 to N (I don't know if teams change, year to year, so do it every year)
    le = LabelEncoder()
    teams = pd.DataFrame()
    teams['Team'] = pd.concat([seasonGames['WTeamID'],seasonGames['LTeamID']])
    le.fit(teams['Team'])
    num_teams = len(le.classes_)
    seasonGames['WTeamID']=le.transform(seasonGames['WTeamID'])
    seasonGames['LTeamID'] = le.transform(seasonGames['LTeamID'])
    X,y = buildModel3(seasonGames,num_teams)
    #calculate X'X and X'y (normal equations).
    M = X.T.dot(X)
    p = X.T.dot(y)
    
    
    #if we have singular (rank < number of columns), use numpy's least square solver
    if np.linalg.matrix_rank(M)<2*num_teams:
        b = np.linalg.lstsq(X,y)[0]
    else:
        b = np.linalg.inv(M).dot(p) 
    print("Season=",seasons[season],", Matrix Rank=",np.linalg.matrix_rank(M),", Teams=",num_teams*2,", HomeCourt=",b[0])        
    #remove all home court ratings, because there are no home courts in tourney(s) 
    bNew = np.delete(b,num_teams)
    ratings = dict(zip(le.classes_,bNew))
    seasonsRatingsModel3[seasons[season]] = ratings
#calculate tournament results (except 2014-2017) so we can estimate standard deviation....
tournamentsSeasons = list(set(tournamentsData['Season']))
tournamentsSeasons.remove(2015)
tournamentsSeasons.remove(2014)
tournamentsSeasons.remove(2016)
tournamentsSeasons.remove(2017)
tournamentResultsModel3 = np.zeros(0)
for season in range(len(tournamentsSeasons)):
    tournamentGames = getSeasonGames(tournamentsGames,tournamentsSeasons[season])
    #ratings
    ratings = seasonsRatingsModel3[tournamentsSeasons[season]]
    x = getTournamentResults(tournamentGames,ratings)    
    tournamentResultsModel3 = np.concatenate((tournamentResultsModel3,x),axis=0)    
#plot curve of predictions so we can 'eyeball' estimated sd.
xs,ys = ecdf(tournamentResultsModel3)
plt.plot(xs, ys, label="handwritten", marker=">", markerfacecolor='none')
mu = 0
sd = 16
x = np.linspace(-40,40, 80)
CY = np.cumsum(mlab.normpdf(x,mu,sd))

plt.plot(x,CY)
plt.show()
predictSubmission(sampleSubmission,seasonsRatingsModel3,16)#no home advantage when predicting tourney games
logLoss(sampleSubmission)
predictSubmission(sampleSubmission,seasonsRatingsModel3,14)#no home advantage when predicting tourney games
logLoss(sampleSubmission)
predictSubmission(sampleSubmission,seasonsRatingsModel3,12)#no home advantage when predicting tourney games
logLoss(sampleSubmission)
predictSubmission(sampleSubmission,seasonsRatingsModel3,10)#no home advantage when predicting tourney games
logLoss(sampleSubmission)
predictSubmission(sampleSubmission,seasonsRatingsModel3,8)#no home advantage when predicting tourney games
logLoss(sampleSubmission)
#predict using basic model/model 1
predictSubmission(sampleSubmission,seasonsRatingsModel1,11)
sampleSubmission.to_csv('submission.csv', index=False)
#Thanks to Kaggle user the1owl for sklearn code
from sklearn import *
seasons = list(sorted(set(games['Season'])))
seasonsRatingsModel3 = {}
for season in range(len(seasons)):
    seasonGames = getSeasonGames(games,seasons[season])
    #Encode teams, so we can index them from 0 to N (I don't know if teams change, year to year, so do it every year)
    le = LabelEncoder()
    teams = pd.DataFrame()
    teams['Team'] = pd.concat([seasonGames['WTeamID'],seasonGames['LTeamID']])
    le.fit(teams['Team'])
    num_teams = len(le.classes_)
    seasonGames['WTeamID']=le.transform(seasonGames['WTeamID'])
    seasonGames['LTeamID'] = le.transform(seasonGames['LTeamID'])
    X,y = buildModel3(seasonGames,num_teams)    
    reg = linear_model.Lasso(fit_intercept=False)
    reg.fit(X, y)
    b = reg.coef_
    print("Season=",seasons[season],", Matrix Rank=",np.linalg.matrix_rank(M),", Teams=",num_teams*2,", HomeCourt=",b[0])        
    #remove all home court ratings, because there are no home courts in tourney(s) 
    bNew = np.delete(b,num_teams)
    ratings = dict(zip(le.classes_,bNew))
    seasonsRatingsModel3[seasons[season]] = ratings

#calculate tournament results (except 2014-2017) so we can estimate standard deviation....
tournamentsSeasons = list(set(tournamentsData['Season']))
tournamentsSeasons.remove(2015)
tournamentsSeasons.remove(2014)
tournamentsSeasons.remove(2016)
tournamentsSeasons.remove(2017)
tournamentResultsModel3 = np.zeros(0)
for season in range(len(tournamentsSeasons)):
    tournamentGames = getSeasonGames(tournamentsGames,tournamentsSeasons[season])
    #ratings
    ratings = seasonsRatingsModel3[tournamentsSeasons[season]]
    x = getTournamentResults(tournamentGames,ratings)    
    tournamentResultsModel3 = np.concatenate((tournamentResultsModel3,x),axis=0)    
#plot curve of predictions so we can 'eyeball' estimated sd.
xs,ys = ecdf(tournamentResultsModel3)
plt.plot(xs, ys, label="handwritten", marker=">", markerfacecolor='none')
mu = 0
sd = 16
x = np.linspace(-40,40, 80)
CY = np.cumsum(mlab.normpdf(x,mu,sd))

plt.plot(x,CY)
plt.show()
predictSubmission(sampleSubmission,seasonsRatingsModel3,16)#no home advantage when predicting tourney games
logLoss(sampleSubmission)
#Thanks to Kaggle user the1owl for sklearn code
seasons = list(sorted(set(games['Season'])))
seasonsRatingsModel3 = {}
for season in range(len(seasons)):
    seasonGames = getSeasonGames(games,seasons[season])
    #Encode teams, so we can index them from 0 to N (I don't know if teams change, year to year, so do it every year)
    le = LabelEncoder()
    teams = pd.DataFrame()
    teams['Team'] = pd.concat([seasonGames['WTeamID'],seasonGames['LTeamID']])
    le.fit(teams['Team'])
    num_teams = len(le.classes_)
    seasonGames['WTeamID']=le.transform(seasonGames['WTeamID'])
    seasonGames['LTeamID'] = le.transform(seasonGames['LTeamID'])
    X,y = buildModel3(seasonGames,num_teams)    
    reg = linear_model.HuberRegressor()
    reg.fit(X, y)
    b = reg.coef_
    print("Season=",seasons[season],", Matrix Rank=",np.linalg.matrix_rank(M),", Teams=",num_teams*2,", HomeCourt=",b[0])        
    #remove all home court ratings, because there are no home courts in tourney(s) 
    bNew = np.delete(b,num_teams)
    ratings = dict(zip(le.classes_,bNew))
    seasonsRatingsModel3[seasons[season]] = ratings
#calculate tournament results (except 2014-2017) so we can estimate standard deviation....
tournamentsSeasons = list(set(tournamentsData['Season']))
tournamentsSeasons.remove(2015)
tournamentsSeasons.remove(2014)
tournamentsSeasons.remove(2016)
tournamentsSeasons.remove(2017)
tournamentResultsModel3 = np.zeros(0)
for season in range(len(tournamentsSeasons)):
    tournamentGames = getSeasonGames(tournamentsGames,tournamentsSeasons[season])
    #ratings
    ratings = seasonsRatingsModel3[tournamentsSeasons[season]]
    x = getTournamentResults(tournamentGames,ratings)    
    tournamentResultsModel3 = np.concatenate((tournamentResultsModel3,x),axis=0)    
#plot curve of predictions so we can 'eyeball' estimated sd.
xs,ys = ecdf(tournamentResultsModel3)
plt.plot(xs, ys, label="handwritten", marker=">", markerfacecolor='none')
mu = 0
sd = 16
x = np.linspace(-40,40, 80)
CY = np.cumsum(mlab.normpdf(x,mu,sd))

plt.plot(x,CY)
plt.show()
predictSubmission(sampleSubmission,seasonsRatingsModel3,16)#no home advantage when predicting tourney games
logLoss(sampleSubmission)
predictSubmission(sampleSubmission,seasonsRatingsModel3,14)#no home advantage when predicting tourney games
logLoss(sampleSubmission)
predictSubmission(sampleSubmission,seasonsRatingsModel3,12)#no home advantage when predicting tourney games
logLoss(sampleSubmission)
predictSubmission(sampleSubmission,seasonsRatingsModel3,10)#no home advantage when predicting tourney games
logLoss(sampleSubmission)
predictSubmission(sampleSubmission,seasonsRatingsModel3,8)#no home advantage when predicting tourney games
logLoss(sampleSubmission)
#Thanks to Kaggle user the1owl for sklearn code
seasons = list(sorted(set(games['Season'])))
seasonsRatingsModel3 = {}
for season in range(len(seasons)):
    seasonGames = getSeasonGames(games,seasons[season])
    #Encode teams, so we can index them from 0 to N (I don't know if teams change, year to year, so do it every year)
    le = LabelEncoder()
    teams = pd.DataFrame()
    teams['Team'] = pd.concat([seasonGames['WTeamID'],seasonGames['LTeamID']])
    le.fit(teams['Team'])
    num_teams = len(le.classes_)
    seasonGames['WTeamID']=le.transform(seasonGames['WTeamID'])
    seasonGames['LTeamID'] = le.transform(seasonGames['LTeamID'])
    X,y = buildModel3(seasonGames,num_teams)    
    reg = linear_model.Ridge(fit_intercept=False)
    reg.fit(X, y)
    b = reg.coef_
    print("Season=",seasons[season],", Matrix Rank=",np.linalg.matrix_rank(M),", Teams=",num_teams*2,", HomeCourt=",b[0])        
    #remove all home court ratings, because there are no home courts in tourney(s) 
    bNew = np.delete(b,num_teams)
    ratings = dict(zip(le.classes_,bNew))
    seasonsRatingsModel3[seasons[season]] = ratings
#calculate tournament results (except 2014-2017) so we can estimate standard deviation....
tournamentsSeasons = list(set(tournamentsData['Season']))
tournamentsSeasons.remove(2015)
tournamentsSeasons.remove(2014)
tournamentsSeasons.remove(2016)
tournamentsSeasons.remove(2017)
tournamentResultsModel3 = np.zeros(0)
for season in range(len(tournamentsSeasons)):
    tournamentGames = getSeasonGames(tournamentsGames,tournamentsSeasons[season])
    #ratings
    ratings = seasonsRatingsModel3[tournamentsSeasons[season]]
    x = getTournamentResults(tournamentGames,ratings)    
    tournamentResultsModel3 = np.concatenate((tournamentResultsModel3,x),axis=0)    
#plot curve of predictions so we can 'eyeball' estimated sd.
xs,ys = ecdf(tournamentResultsModel3)
plt.plot(xs, ys, label="handwritten", marker=">", markerfacecolor='none')
mu = 0
sd = 16
x = np.linspace(-40,40, 80)
CY = np.cumsum(mlab.normpdf(x,mu,sd))

plt.plot(x,CY)
plt.show()
predictSubmission(sampleSubmission,seasonsRatingsModel3,16)#no home advantage when predicting tourney games
logLoss(sampleSubmission)
predictSubmission(sampleSubmission,seasonsRatingsModel3,14)#no home advantage when predicting tourney games
logLoss(sampleSubmission)
predictSubmission(sampleSubmission,seasonsRatingsModel3,12)#no home advantage when predicting tourney games
logLoss(sampleSubmission)
predictSubmission(sampleSubmission,seasonsRatingsModel3,10)#no home advantage when predicting tourney games
logLoss(sampleSubmission)
