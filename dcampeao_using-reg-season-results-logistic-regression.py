import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
data_dir = '../input/'
tourresults = pd.read_csv(data_dir + 'WNCAATourneyCompactResults.csv')
tourseeds = pd.read_csv(data_dir + 'WNCAATourneySeeds.csv')
seasonresults = pd.read_csv(data_dir + 'WRegularSeasonCompactResults.csv')
tourresults.tail()
def GetWinsLosses(seasonresults, year, teamID):
    #This function returns the number of wins and losses for a given teamId in a specified season.
    Spec_season = seasonresults['Season']==year
    teamIDW = seasonresults['WTeamID']==teamID
    teamIDL = seasonresults['LTeamID']==teamID
    wins = len(seasonresults[Spec_season & teamIDW])
    losses = len(seasonresults[Spec_season & teamIDL])
    return [wins, losses]

def seed_to_int(seed):
    #Get just the digits from the seeding. Return as int
    s_int = int(seed[1:3])
    return s_int
#Specify a Season
year = 2018

df_seeds = tourseeds[tourseeds['Season']==year]
df_tour = tourresults[tourresults['Season']==year]

df_season = seasonresults[seasonresults['Season']==year]

df_seeds['seed_int'] = df_seeds.Seed.apply(seed_to_int)
df_seeds.drop(labels=['Seed'], inplace=True, axis=1)
seasonresultsNew = seasonresults.drop(['DayNum','WScore','LScore','WLoc','NumOT'],axis=1)
Teams = df_seeds.TeamID.sort_values().unique()
n_test_games = int(len(Teams)*(len(Teams)-1)/2)

#Dummies to get Winner and Looser seeds
df_winseeds = df_seeds.rename(columns={'TeamID':'WTeamID', 'seed_int':'WSeed'})
df_lossseeds = df_seeds.rename(columns={'TeamID':'LTeamID', 'seed_int':'LSeed'})

CompleteDF = pd.DataFrame()
ii=0
IDlist = []
for t1 in Teams:
    #Seeds
    t1_seed = df_seeds[(df_seeds.TeamID == t1) & (df_seeds.Season == year)].seed_int.values[0]
    CompleteDF.loc[ii,'seed'] = t1_seed
    
    #Wins and Losses
    [winst1, lossest1] = GetWinsLosses(seasonresults, year, t1)

    CompleteDF.loc[ii,'Season'] = year

    #Fill Team1 Data
    CompleteDF.loc[ii,'TeamID'] = t1
    CompleteDF.loc[ii,'Wins'] = winst1
    CompleteDF.loc[ii,'Losses'] = lossest1

    ii+=1

CompleteDF.tail()
BaseDF = seasonresults[seasonresults['Season']==year]
BaseDF.tail()
WinerIDlessthanLoserID = BaseDF['WTeamID']<BaseDF['LTeamID']
BaseDF.loc[WinerIDlessthanLoserID, 'Team1'] = BaseDF['WTeamID']
BaseDF.loc[~WinerIDlessthanLoserID, 'Team1'] = BaseDF['LTeamID']
BaseDF.loc[WinerIDlessthanLoserID, 'Team2'] = BaseDF['LTeamID']
BaseDF.loc[~WinerIDlessthanLoserID, 'Team2'] = BaseDF['WTeamID']

BaseDF.loc[WinerIDlessthanLoserID, 'Result'] = 1
BaseDF.loc[~WinerIDlessthanLoserID, 'Result'] = 0

BaseDF = BaseDF.drop(['DayNum','WScore','LScore','WLoc','NumOT','WTeamID','LTeamID'],axis=1)

BaseDF.tail()
FinalDF = pd.merge(left=BaseDF, right=CompleteDF, how='outer', left_on=['Season', 'Team1'], right_on=['Season', 'TeamID'])
FinalDF.drop(['TeamID'],axis=1,inplace=True)
FinalDF = FinalDF.rename(columns={'seed':'seedT1','Wins':'WinsT1','Losses':'LossesT1'})

FinalDF = pd.merge(left=FinalDF, right=CompleteDF, how='right', left_on=['Season', 'Team2'], right_on=['Season', 'TeamID'])
FinalDF.drop(['TeamID'],axis=1,inplace=True)
FinalDF = FinalDF.rename(columns={'seed':'seedT2','Wins':'WinsT2','Losses':'LossesT2'})
FinalDF = FinalDF.dropna()
FinalDF['Seed'] = FinalDF['seedT1']-FinalDF['seedT2']
FinalDF['Wins'] = FinalDF['WinsT1']-FinalDF['WinsT2']
FinalDF['Losses'] = FinalDF['LossesT1']-FinalDF['LossesT2']

FinalDF.drop(['seedT1','seedT2','WinsT1','WinsT2','LossesT1','LossesT2'],axis=1,inplace=True)

FinalDF_full = FinalDF.copy()
FinalDF_full.head()
X = FinalDF.drop(['Season','Result','Team1','Team2'],axis=1)
y = FinalDF.Result.values

logreg = LogisticRegression(C=0.1)
logreg.fit(X, y)
#Get all possible teams
Teams = df_seeds.TeamID.sort_values().unique()
n_test_games = int(len(Teams)*(len(Teams)-1)/2)

X_test = pd.DataFrame()

ii=0
IDlist = []
for t1 in Teams:
    for t2 in Teams:
        if t1 < t2:
            t1_seed = df_seeds[(df_seeds.TeamID == t1) & (df_seeds.Season == year)].seed_int.values[0]
            t2_seed = df_seeds[(df_seeds.TeamID == t2) & (df_seeds.Season == year)].seed_int.values[0]
            
            X_test.loc[ii,'Seed'] = t1_seed - t2_seed
             
            #Wins and Losses
            [winst1, lossest1] = GetWinsLosses(seasonresults, year, t1)
            [winst2, lossest2] = GetWinsLosses(seasonresults, year, t2)
            
            X_test.loc[ii,'Wins'] = winst1-winst2
            X_test.loc[ii,'Losses'] = lossest1-lossest2
            ii+=1
            
            IDlist.append('{}_{}_{}'.format(year,t1,t2))
preds = logreg.predict_proba(X_test)[:,1]

clipped_preds = np.clip(preds, 0.01, 0.99)
df_final_season = pd.DataFrame(columns=['ID','Pred'])
df_final_season.Pred = clipped_preds
df_final_season.ID = IDlist

df_final_season.tail()
BaseDF = tourresults
WinerIDlessthanLoserID = BaseDF['WTeamID']<BaseDF['LTeamID']
BaseDF.loc[WinerIDlessthanLoserID, 'Team1'] = BaseDF['WTeamID']
BaseDF.loc[~WinerIDlessthanLoserID, 'Team1'] = BaseDF['LTeamID']
BaseDF.loc[WinerIDlessthanLoserID, 'Team2'] = BaseDF['LTeamID']
BaseDF.loc[~WinerIDlessthanLoserID, 'Team2'] = BaseDF['WTeamID']

BaseDF.loc[WinerIDlessthanLoserID, 'Result'] = 1
BaseDF.loc[~WinerIDlessthanLoserID, 'Result'] = 0

BaseDF['Team1'] = BaseDF['Team1'].astype(int)
BaseDF['Team2'] = BaseDF['Team2'].astype(int)

BaseDF.loc[:,'ID']=BaseDF['Season'].astype(str)+'_'+BaseDF['Team1'].astype(str)+'_'+BaseDF['Team2'].astype(str)

isSeason = BaseDF['Season'] == year
BaseDF = BaseDF[isSeason]        

BaseDF = BaseDF.drop(['DayNum','WScore','LScore','WLoc','NumOT','WTeamID','LTeamID','Team1','Team2','Season'],axis=1)
BaseDF['ID']=IDlist
BaseDF['Result']=0
BaseDF.head()
PredsAndResults = pd.merge(left=BaseDF,right=df_final_season, on='ID')
PredsAndResults['Loss']=PredsAndResults['Result']*np.log(PredsAndResults['Pred'])+(1-PredsAndResults['Result'])*np.log(1-PredsAndResults['Pred'])
-PredsAndResults.Loss.sum()/len(PredsAndResults)
PredsAndResults.head()
df_final_season.to_csv('Submission.csv', index=False)