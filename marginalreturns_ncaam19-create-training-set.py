#Create a training set based on the difference between two team's season long stats

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
#bring in data from 2003 on

tourney_results = pd.read_csv('../input/datafiles/NCAATourneyCompactResults.csv')

tourney_results = tourney_results.loc[tourney_results['Season'] >= 2003]



training_set = pd.DataFrame()

training_set['Result'] = np.random.randint(0,2,len(tourney_results.index))

training_set['Season'] = tourney_results['Season'].values

training_set['Team1'] = training_set['Result'].values * tourney_results['WTeamID'].values + (1-training_set['Result'].values) * tourney_results['LTeamID'].values 

training_set['Team2'] = (1-training_set['Result'].values) * tourney_results['WTeamID'].values + training_set['Result'].values * tourney_results['LTeamID'].values 

# Calculate Delta Seeds

seeds = pd.read_csv('../input/datafiles/NCAATourneySeeds.csv')

seeds['Seed'] =  pd.to_numeric(seeds['Seed'].str[1:3], downcast='integer',errors='coerce')



def delta_seed(row):

    cond = (seeds['Season'] == row['Season'])

    return seeds[cond & (seeds['TeamID'] == row['Team1'])]['Seed'].iloc[0] - seeds[cond & (seeds['TeamID'] == row['Team2'])]['Seed'].iloc[0]



training_set['deltaSeed'] = training_set.apply(delta_seed,axis=1)
# Calculate Delta Ordinals

mo = pd.read_csv('../input/masseyordinals/MasseyOrdinals.csv')

mo = mo[(mo['RankingDayNum'] == 128) & (mo['Season'] >= 2003)] # See Note on MO



def delta_ord(row):

    cond =  (mo['Season'] == row['Season'])

    cond1 = (mo['TeamID'] == row['Team1']) & cond

    cond2 = (mo['TeamID'] == row['Team2']) & cond

    t1 = mo[cond1]['OrdinalRank'].mean()

    t2 = mo[cond2]['OrdinalRank'].mean()

    return  t1-t2



training_set['deltaMO'] = training_set.apply(delta_ord,axis=1)
training_set
# Calculate weighted win pct based on location

season_results = pd.read_csv('../input/datafiles/RegularSeasonDetailedResults.csv')

season_results['LLoc'] = season_results['WLoc']

season_results.loc[season_results['WLoc'] == 'H','LLoc'] = 'A'

season_results.loc[season_results['WLoc'] == 'A','LLoc'] = 'H'

season_results['WLocWeight'] = season_results['WLoc']

season_results.loc[season_results['WLoc'] == 'H','WLocWeight'] = 0.6

season_results.loc[season_results['WLoc'] == 'N','WLocWeight'] = 1

season_results.loc[season_results['WLoc'] == 'A','WLocWeight'] = 1.4

season_results['LLocWeight'] = season_results['LLoc']

season_results.loc[season_results['LLoc'] == 'H','LLocWeight'] = 1.6

season_results.loc[season_results['LLoc'] == 'N','LLocWeight'] = 1

season_results.loc[season_results['LLoc'] == 'A','LLocWeight'] = 0.6



record = pd.DataFrame({'Adj. wins': season_results.groupby(['Season','WTeamID'])['WLocWeight'].sum()}).reset_index();

adjlosses = pd.DataFrame({'Adj. losses': season_results.groupby(['Season','LTeamID'])['LLocWeight'].sum()}).reset_index();

wins = pd.DataFrame({'wins': season_results.groupby(['Season','WTeamID']).size()}).reset_index();

losses = pd.DataFrame({'losses': season_results.groupby(['Season','LTeamID']).size()}).reset_index();

record = record.merge(adjlosses, how='outer', left_on=['Season','WTeamID'], right_on=['Season','LTeamID'])

record = record.merge(wins, how='outer', left_on=['Season','WTeamID'], right_on=['Season','WTeamID'])

record = record.merge(losses, how='outer', left_on=['Season','WTeamID'], right_on=['Season','LTeamID'])

record = record.fillna(0)

record['games'] = record['wins']+record['losses']



def delta_winPct(row):

    cond1 = (record['Season'] == row['Season']) & (record['WTeamID'] == row['Team1'])

    cond2 = (record['Season'] == row['Season']) & (record['WTeamID'] == row['Team2'])

    return (record[cond1]['Adj. wins']/record[cond1]['games']).mean() - (record[cond2]['Adj. wins']/record[cond2]['games']).mean()



training_set['deltaWinPct'] = training_set.apply(delta_winPct,axis=1)
#create O and D ratings for each team and game

season_results['WORtg'] = season_results['WScore']/(season_results['WFGA'] - season_results['WOR'] + season_results['WTO'] + season_results['WFTA']*.475)

season_results['LDRtg'] = season_results['WORtg']

season_results['WDRtg'] = season_results['LScore']/(season_results['LFGA'] - season_results['LOR'] + season_results['LTO'] + season_results['LFTA']*.475)

season_results['LORtg'] = season_results['WDRtg']

#Use O and D ratings to create Net rating (O - D)

season_results['WNRtg'] = season_results['WORtg'] - season_results['WDRtg']

season_results['LNRtg'] = season_results['LORtg'] - season_results['LDRtg']

season_results.head()
ratings = pd.DataFrame({'WNRtg' : season_results.groupby(['Season','WTeamID'])['WNRtg'].mean(),

                       'LNRtg' : season_results.groupby(['Season','LTeamID'])['LNRtg'].mean()}).reset_index();



ratings.rename(index=str,columns={'level_0':'Season',

                                 'level_1':'TeamID'}, inplace=True)

ratings = ratings.merge(record, how='outer', left_on=['Season','TeamID'], right_on=['Season','LTeamID_y'],copy=False)
#create Weighted average net rating for the entire season

ratings['WANRtg'] = (ratings['WNRtg']*ratings['wins'] + ratings['LNRtg']*ratings['losses'])/ratings['games']

#create ranks for net efficiency

ratings['NRtgRank'] = ratings.groupby('Season')['WANRtg'].rank('dense',ascending=False)

ratings.head()
#merge season average net ratings for winning team

season_results = season_results.merge(ratings[['Season','TeamID','WANRtg','NRtgRank']], 

                                          how='left', 

                                          left_on = ['Season','WTeamID'], 

                                          right_on = ['Season','TeamID'],)

season_results.rename(index=str,columns={'WANRtg':'WSNRtg',

                                         'NRtgRank':'WNRtgRank'}, inplace=True)

season_results.drop(columns=['TeamID'],inplace=True)



#merge season average net ratings for losing team

season_results = season_results.merge(ratings[['Season','TeamID','WANRtg','NRtgRank']], 

                                          how='left', 

                                          left_on = ['Season','LTeamID'], 

                                          right_on = ['Season','TeamID'],)

season_results.rename(index=str,columns={'WANRtg':'LSNRtg',

                                         'NRtgRank':'LNRtgRank'}, inplace=True)

season_results.drop(columns=['TeamID'],inplace=True)
season_results.dropna(inplace=True)
#create adjusted ratings for each team, offense and defense

#WWtdORtg is Winning team's (W) Weighted (Wtd) Offense (O) Rating (Rtg)

maxteams = 351

season_results['WWtdNRtg'] = season_results['WNRtg'] / (season_results['LNRtgRank']/maxteams)

season_results['LWtdNRtg'] = season_results['LNRtg'] / (season_results['WNRtgRank']/maxteams)
#group up season points for/against

dfW = season_results.groupby(['Season','WTeamID']).sum().reset_index()

dfL = season_results.groupby(['Season','LTeamID']).sum().reset_index()



def get_points_for(row):

    wcond = (dfW['Season'] == row['Season']) & (dfW['WTeamID'] == row['WTeamID']) 

    fld1 = 'WScore'

    lcond = (dfL['Season'] == row['Season']) & (dfL['LTeamID'] == row['WTeamID']) 

    fld2 = 'LScore'

    retVal = dfW[wcond][fld1].sum()

    if len(dfL[lcond][fld2]) > 0:

        retVal = retVal + dfL[lcond][fld2].sum() 

    return retVal



def get_points_against(row):

    wcond = (dfW['Season'] == row['Season']) & (dfW['WTeamID'] == row['WTeamID']) 

    fld1 = 'LScore'

    lcond = (dfL['Season'] == row['Season']) & (dfL['LTeamID'] == row['WTeamID']) 

    fld2 = 'WScore'

    retVal = dfW[wcond][fld1].sum()

    if len(dfL[lcond][fld2]) > 0:

        retVal = retVal + dfL[lcond][fld2].sum() 

    return retVal



record['PointsFor'] = record.apply(get_points_for, axis=1)

record['PointsAgainst'] = record.apply(get_points_against, axis=1)
def get_remaining_stats(row, field):

    wcond = (dfW['Season'] == row['Season']) & (dfW['WTeamID'] == row['WTeamID']) 

    fld1 = 'W' + field

    lcond = (dfL['Season'] == row['Season']) & (dfL['LTeamID'] == row['WTeamID']) 

    fld2 = 'L'+ field

    retVal = dfW[wcond][fld1].sum()

    if len(dfL[lcond][fld2]) > 0:

        retVal = retVal + dfL[lcond][fld2].sum()

    return retVal



cols = ['FGM','FGA','FGM3','FGA3','FTM','FTA','OR','DR','Ast','TO','Stl','Blk','PF','NRtgRank','WtdNRtg',]

for col in cols:

    print("Processing",col)

    record[col] = record.apply(get_remaining_stats, args=(col,), axis=1)
def delta_stat(row, field):

    cond1 = (record['Season'] == row['Season']) & (record['WTeamID'] == row['Team1'])

    cond2 = (record['Season'] == row['Season']) & (record['WTeamID'] == row['Team2'])

    return (record[cond1][field]/record[cond1]['games']).mean() - (record[cond2][field]/record[cond2]['games']).mean()



cols = ['PointsFor','PointsAgainst','FGM','FGA','FGM3','FGA3','FTM','FTA','OR','DR','Ast','TO','Stl','Blk','PF','NRtgRank','WtdNRtg',]



for col in cols:

    print("Processing",col)

    training_set['delta' + col] = training_set.apply(delta_stat,args=(col,),axis=1)



training_set.to_csv('training_set.csv', index=False)

training_set.head()