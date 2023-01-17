# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder


seasonsDataDetailed = pd.read_csv("../input/RegularSeasonDetailedResults.csv")
tournamentsData = pd.read_csv("../input/NCAATourneyCompactResults.csv")
sampleSubmission = pd.read_csv("../input/SampleSubmissionStage2.csv")
seasonsDataDetailed.head()
tournamentsGames = pd.DataFrame()
tournamentsGames['Season'] = tournamentsData['Season']
tournamentsGames['Winner'] = tournamentsData['WTeamID']
tournamentsGames['Loser'] = tournamentsData['LTeamID']
tournamentsGames.head()
def getSeasonGames(games,season):
    seasonGames = pd.DataFrame()
    seasonGames = games.loc[games['Season']== season].copy()
    return seasonGames
from sklearn.preprocessing import normalize
def buildMatrixWin(games,num_teams):
    A = np.zeros([num_teams,num_teams])  #square matrix  
    for row in games.itertuples():
        wid = row.WTeamID
        lid = row.LTeamID        
        A[wid,lid] += 1     #positive
    #normalize A
    A_norm = normalize(A,norm='l1',axis=0)
    
    return A_norm

def buildMatrixScore(games,num_teams):
    A = np.zeros([num_teams,num_teams])    
    for row in games.itertuples():
        wid = row.WTeamID
        lid = row.LTeamID
        wscore = row.WScore
        lscore = row.LScore        
        A[wid,lid] += wscore
        #A[lid,wid] += lscore
    #normalize A
    A_norm = normalize(A,norm='l1',axis=0)
    
    return A_norm
#I found this on the internet, but didn't not where, thanks and apologies to the author
def powerMethodBase(A,x0,iter):
 """ basic power method """
 for i in range(iter):
  x0 = A.dot(x0)# dot(A,x0)
  x0 = x0/np.linalg.norm(x0,1)
 return x0
from sklearn.preprocessing import LabelEncoder
seasons = list(sorted(set(seasonsDataDetailed['Season'])))
#seasons.remove(2018)
seasonsRatings = {}
for season in range(len(seasons)):
    seasonGames = getSeasonGames(seasonsDataDetailed,seasons[season])
    #Encode teams, so we can index them from 0 to N (I don't know if teams change, year to year, so do it every year)
    le = LabelEncoder()
    teams = pd.DataFrame()
    teams['Team'] = pd.concat([seasonGames['WTeamID'],seasonGames['LTeamID']])
    le.fit(teams['Team'])
    num_teams = len(le.classes_)
    seasonGames['WTeamID']=le.transform(seasonGames['WTeamID'])
    seasonGames['LTeamID'] = le.transform(seasonGames['LTeamID'])
    A = buildMatrixWin(seasonGames,num_teams)
    b0 = np.ones(A.shape[0])
    b0 /= num_teams
    print("Season=",seasons[season])
    b = powerMethodBase(A,b0,100)    
    ratings = dict(zip(le.classes_,b))
    seasonsRatings[seasons[season]] = ratings
#Borrowed from Basic Starter Kernel. Thanks!
def get_year_t1_t2(ID):
    """Return a tuple with ints `year`, `team1` and `team2`."""
    return (int(x) for x in ID.split('_'))
from scipy.stats import norm
def predictSubmission(submission,modelRatings,sd):
    
    X_test = np.zeros(shape=(submission.shape[0], 1))
    for ii, row in submission.iterrows():
        season, t1, t2 = get_year_t1_t2(row.ID)
        pred = 0
        prob = 0.5        
        ratings = modelRatings[season]
        if len(ratings)>0:
            wrat = ratings[t1]
            lrat = ratings[t2]
            pred = wrat - lrat
            prob = norm(0,sd).cdf(pred)
        X_test[ii,0] = prob
    submission.Pred = X_test
predictSubmission(sampleSubmission,seasonsRatings,0.015)
sampleSubmission.head()
from sklearn.metrics import log_loss
import pandas as pd
def logLoss(submission):
    
    
    df = sampleSubmission.copy()#read_csv("e:\Kaggle\March 2018\\NCAAMEN\submissionlqx.csv")

    df['Season'] = submission['ID'].apply(lambda x: int(x[0:4]))

    df['team1id'] = submission['ID'].apply(lambda x: int(x[5:9]))

    df['team2id'] = submission['ID'].apply(lambda x: int(x[10:14]))

    results = tournamentsData.copy()# pd.read_csv("e:\Kaggle\March 2018\\NCAAMEN\\NCAATourneyCompactResults.csv")

    results = results.loc[results['DayNum'] > 135].copy()

    results['team1id'] = results[['WTeamID','LTeamID']].min(axis=1)

    results['team2id'] = results[['WTeamID','LTeamID']].max(axis=1)

    df = pd.merge(df, results, how='inner', on=['Season','team1id','team2id'])

    df['result'] = (df['WTeamID'] == df['team1id']).astype(int)

    print(log_loss(df['result'], df['Pred']))
logLoss(sampleSubmission)
predictSubmission(sampleSubmission,seasonsRatings,0.02                )
logLoss(sampleSubmission)
predictSubmission(sampleSubmission,seasonsRatings,0.025                )
logLoss(sampleSubmission)
from sklearn.preprocessing import LabelEncoder
seasons = list(sorted(set(seasonsDataDetailed['Season'])))

seasonsRatingsScores = {}
for season in range(len(seasons)):
    seasonGames = getSeasonGames(seasonsDataDetailed,seasons[season])
    #Encode teams, so we can index them from 0 to N (I don't know if teams change, year to year, so do it every year)
    le = LabelEncoder()
    teams = pd.DataFrame()
    teams['Team'] = pd.concat([seasonGames['WTeamID'],seasonGames['LTeamID']])
    le.fit(teams['Team'])
    num_teams = len(le.classes_)
    seasonGames['WTeamID']=le.transform(seasonGames['WTeamID'])
    seasonGames['LTeamID'] = le.transform(seasonGames['LTeamID'])
    A = buildMatrixScore(seasonGames,num_teams)
    b0 = np.ones(A.shape[0])
    
    b = powerMethodBase(A,b0,100)        
    ratings = dict(zip(le.classes_,b))
    seasonsRatingsScores[seasons[season]] = ratings
predictSubmission(sampleSubmission,seasonsRatingsScores,0.03)
logLoss(sampleSubmission)
predictSubmission(sampleSubmission,seasonsRatingsScores,0.025)
logLoss(sampleSubmission)
predictSubmission(sampleSubmission,seasonsRatingsScores,0.02)
logLoss(sampleSubmission)
predictSubmission(sampleSubmission,seasonsRatingsScores,0.015)
logLoss(sampleSubmission)
def buildMatrixHome(games,num_teams):
    A = np.zeros([num_teams,num_teams])
    for row in games.itertuples():
        wid = row.WTeamID
        lid = row.LTeamID
        home = 1
        away = 0
        if row['WLoc'] == 'N':
            home=0
        elif row['WLoc'] == 'A':
            home=0
            away=1
        A[wid,lid] += home
        A[lid,wid] += away
    A_norm = normalize(A,norm='l1',axis=0)
    
    return A_norm
def buildMatrixFGM(games,num_teams):
    A = np.zeros([num_teams,num_teams])    
    for row in games.itertuples():
        wid = row.WTeamID
        lid = row.LTeamID
        wfgm = row.WFGM
        lfgm = row.LFGM
        A[wid,lid] += wfgm
        #A[lid,wid] += lfgm
    #normalize A
    for i in range(num_teams):
        colSum = sum(A[:,i])
        if colSum==0:
            
            A[:,i] = 1/num_teams
        else:
            A[:,i] /= colSum
    return A  
def buildMatrixFGA(games,num_teams):
    A = np.zeros([num_teams,num_teams])    
    for row in games.itertuples():
        wid = row.WTeamID
        lid = row.LTeamID
        wfgm = row.WFGA
        lfgm = row.LFGA
        A[wid,lid] += wfgm
        #A[lid,wid] += lfgm
    #normalize A
    A_norm = normalize(A,norm='l1',axis=0)
    
    return A_norm   
def buildMatrixFGM3(games,num_teams):
    A = np.zeros([num_teams,num_teams])    
    for row in games.itertuples():
        wid = row.WTeamID
        lid = row.LTeamID
        wfgm = row.WFGM3
        lfgm = row.LFGM3
        A[wid,lid] += wfgm
        #A[lid,wid] += lfgm
    #normalize A
    A_norm = normalize(A,norm='l1',axis=0)
    
    return A_norm   
def buildMatrixFGA3(games,num_teams):
    A = np.zeros([num_teams,num_teams])    
    for row in games.itertuples():
        wid = row.WTeamID
        lid = row.LTeamID
        wfgm = row.WFGA3
        lfgm = row.LFGA3
        A[wid,lid] += wfgm
        #A[lid,wid] += lfgm
    #normalize A
    A_norm = normalize(A,norm='l1',axis=0)
    
    return A_norm   

def buildEnsembleMatrix(games,num_teams,weights):
    win = buildMatrixWin(games,num_teams)
    score = buildMatrixScore(games,num_teams)
    fgm = buildMatrixFGM(games,num_teams)
    fga = buildMatrixFGA(games,num_teams)
    fgm3 = buildMatrixFGM3(games,num_teams)
    fga3 = buildMatrixFGA3(games,num_teams)
    A = np.zeros([num_teams,num_teams])
    #apply weighting, weights must add up to 1.
    A = win*weights[0] + score * weights[1] +fgm*weights[2] + fga*weights[3]+fgm3*weights[4] + fga3*weights[5]
    return A
    
from sklearn.preprocessing import LabelEncoder
seasons = list(sorted(set(seasonsDataDetailed['Season'])))
seasonsRatingsEnsemble = {}
for season in range(len(seasons)):
    seasonGames = getSeasonGames(seasonsDataDetailed,seasons[season])
    #Encode teams, so we can index them from 0 to N (I don't know if teams change, year to year, so do it every year)
    le = LabelEncoder()
    teams = pd.DataFrame()
    teams['Team'] = pd.concat([seasonGames['WTeamID'],seasonGames['LTeamID']])
    le.fit(teams['Team'])
    num_teams = len(le.classes_)
    seasonGames['WTeamID']=le.transform(seasonGames['WTeamID'])
    seasonGames['LTeamID'] = le.transform(seasonGames['LTeamID'])
    weights=[1/6,1/6,1/6,1/6,1/6,1/6] #world's most bogus weighting scheme
    A = buildEnsembleMatrix(seasonGames,num_teams,weights)
    b0 = np.ones(A.shape[0])
    print("Season=",seasons[season],", Sum=",sum(b))
    b = powerMethodBase(A,b0,100)        
    ratings = dict(zip(le.classes_,b))
    seasonsRatingsEnsemble[seasons[season]] = ratings
predictSubmission(sampleSubmission,seasonsRatingsEnsemble,0.03)                 
logLoss(sampleSubmission)
predictSubmission(sampleSubmission,seasonsRatingsEnsemble,0.025  )            
logLoss(sampleSubmission)
predictSubmission(sampleSubmission,seasonsRatingsEnsemble,0.02)                 
logLoss(sampleSubmission)
predictSubmission(sampleSubmission,seasonsRatingsEnsemble,0.0225)                 
logLoss(sampleSubmission)

