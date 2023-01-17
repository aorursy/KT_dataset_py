import numpy as np 

import pandas as pd

import os

from sklearn import metrics

import warnings

warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

#import data

in_path = '../input/mens-machine-learning-competition-2019/'

RegularSeasonDetailedResults = pd.read_csv(in_path + 'datafiles/RegularSeasonDetailedResults.csv')

#Events_2018 = pd.read_csv(in_path + 'playbyplay_2018/Events_2018.csv')

#Players_2018 = pd.read_csv(in_path + 'playbyplay_2018/Players_2018.csv')

Events_2017 = pd.read_csv(in_path + 'playbyplay_2017/Events_2017.csv')

Players_2017 = pd.read_csv(in_path + 'playbyplay_2017/Players_2017.csv')

Events_2016 = pd.read_csv(in_path + 'playbyplay_2016/Events_2016.csv')

Players_2016 = pd.read_csv(in_path + 'playbyplay_2016/Players_2016.csv')

Events_2015 = pd.read_csv(in_path + 'playbyplay_2015/Events_2015.csv')

Players_2015 = pd.read_csv(in_path + 'playbyplay_2015/Players_2015.csv')

Events_2014 = pd.read_csv(in_path + 'playbyplay_2014/Events_2014.csv')

Players_2014 = pd.read_csv(in_path + 'playbyplay_2014/Players_2014.csv')

Events_2013 = pd.read_csv(in_path + 'playbyplay_2013/Events_2013.csv')

Players_2013 = pd.read_csv(in_path + 'playbyplay_2013/Players_2013.csv')

Events_2012 = pd.read_csv(in_path + 'playbyplay_2012/Events_2012.csv')

Players_2012 = pd.read_csv(in_path + 'playbyplay_2012/Players_2012.csv')

Events_2011 = pd.read_csv(in_path + 'playbyplay_2011/Events_2011.csv')

Players_2011 = pd.read_csv(in_path + 'playbyplay_2011/Players_2011.csv')

Events_2010 = pd.read_csv(in_path + 'playbyplay_2010/Events_2010.csv')

Players_2010 = pd.read_csv(in_path + 'playbyplay_2010/Players_2010.csv')

#Look at all unique EventType values within the Events dataset

Events_2010.EventType.unique()
#Concatenate all Events datasets into one file.

#Events = Events_2010.append([Events_2011, Events_2012, Events_2013, Events_2014, Events_2015, Events_2016, Events_2017,Events_2018 ])

#del Events_2010, Events_2011, Events_2012, Events_2013, Events_2014, Events_2015, Events_2016, Events_2017, Events_2018

#Events.shape
#Concatenate all Player datasets into one file.

#Players = Players_2010.append([Players_2011, Players_2012, Players_2013, Players_2014, Players_2015, Players_2016, Players_2017,Players_2018])

#del Players_2010, Players_2011, Players_2012, Players_2013, Players_2014, Players_2015, Players_2016, Players_2017, Players_2018

#Players.shape
#Due to size constraints, I am going to run this on the 2010 dataframes only.

Events = Events_2010

Players = Players_2010
#Query for EventTypes we will later use to derive raw stats

Events = Events.loc[Events['EventType'].isin(['miss2_lay', 'reb_off', 'made2_jump', 'miss2_jump',

       'assist', 'made3_jump', 'block', 'reb_def','miss1_free', 'made1_free', 'miss3_jump', 'turnover',

       'steal', 'made2_dunk', 'timeout_tv', 'made2_lay','reb_dead', 'made2_tip', 'miss2_dunk', 'miss2_tip'])]
Events.head(5)
#Join Player dataset and Event dataset to identify the winning WTeamID/LTeamID with the Player/Team EventType

Events = pd.merge(Events, Players, how='inner', 

                    left_on=['Season', 'EventPlayerID'],

                    right_on=['Season', 'PlayerID'])

Events = Events.sort_values(['EventID'], ascending=[1])

Events = Events.reset_index(drop=True)

Events = Events.drop(['EventID'], axis=1)

Events = Events.drop_duplicates()

Events['EventID'] = Events.index

del Players

#Derive Winning and Losing stats uniformly to match the same identifiers as represented in the RegularSeasonDetailedResults dataframe

def deriveRawStats( Events):

    Events['WScore'] = np.where(

            ( Events.EventType.str.contains('made1_free')) & (Events['WTeamID']==Events['TeamID']) , 1, np.where(

            ( Events.EventType.str.contains('made2_jump|made2_dunk|made2_lay|made2_tip')) & (Events['WTeamID']==Events['TeamID']) , 2, np.where(

            ( Events.EventType.str.contains('made3_jump')) & (Events['WTeamID']==Events['TeamID']) , 3, 0)))

    Events['WFGM'] = np.where(

            ( Events.EventType.str.contains('made2_jump|made2_dunk|made2_lay|made2_tip|made3_jump')) & (Events['WTeamID']==Events['TeamID']) , 1, 0)

    Events['WFGA'] = np.where(

            ( Events.EventType.str.contains('made2_jump|made2_dunk|made2_lay|made2_tip|made3_jump|miss2_lay|miss2_tip|miss2_jump|miss2_dunk|miss3_jump')) & (Events['WTeamID']==Events['TeamID']) , 1, 0)

    Events['WFGM3'] = np.where(

            ( Events.EventType.str.contains('made3_jump')) & (Events['WTeamID']==Events['TeamID']) , 1, 0)

    Events['WFGA3'] = np.where(

            ( Events.EventType.str.contains('made3_jump|miss3_jump')) & (Events['WTeamID']==Events['TeamID']) , 1, 0)

    Events['WFTM'] = np.where(

            ( Events.EventType.str.contains('made1_free')) & (Events['WTeamID']==Events['TeamID']) , 1, 0)

    Events['WFTA'] = np.where(

            ( Events.EventType.str.contains('made1_free|miss1_free')) & (Events['WTeamID']==Events['TeamID']) , 1, 0)

    Events['WOR'] = np.where(

            ( Events.EventType.str.contains('reb_off')) & (Events['WTeamID']==Events['TeamID']) , 1, 0)

    Events['WDR'] = np.where(

            ( Events.EventType.str.contains('reb_def')) & (Events['WTeamID']==Events['TeamID']) , 1, 0)

    Events['WAst'] = np.where(

            ( Events.EventType.str.contains('assist')) & (Events['WTeamID']==Events['TeamID']) , 1, 0)

    Events['WTO'] = np.where(

            ( Events.EventType.str.contains('turnover')) & (Events['WTeamID']==Events['TeamID']) , 1, 0)

    Events['WStl'] = np.where(

            ( Events.EventType.str.contains('steal')) & (Events['WTeamID']==Events['TeamID']) , 1, 0)

    Events['WBlk'] = np.where(

            ( Events.EventType.str.contains('block')) & (Events['WTeamID']==Events['TeamID']) , 1, 0)

    Events['WPF'] = np.where(

            ( Events.EventType.str.contains('foul_pers')) & (Events['WTeamID']==Events['TeamID']) , 1, 0)

    #Derive winning team boxscore stats per play

    Events['LScore'] = np.where(

            ( Events.EventType.str.contains('made1_free')) & (Events['LTeamID']==Events['TeamID']) , 1, np.where(

            ( Events.EventType.str.contains('made2_jump|made2_dunk|made2_lay|made2_tip')) & (Events['LTeamID']==Events['TeamID']) , 2, np.where(

            ( Events.EventType.str.contains('made3_jump')) & (Events['LTeamID']==Events['TeamID']) , 3, 0)))

    Events['LFGM'] = np.where(

            ( Events.EventType.str.contains('made2_jump|made2_dunk|made2_lay|made2_tip|made3_jump')) & (Events['LTeamID']==Events['TeamID']) , 1, 0)

    Events['LFGA'] = np.where(

            ( Events.EventType.str.contains('made2_jump|made2_dunk|made2_lay|made2_tip|made3_jump|miss2_lay|miss2_tip|miss2_jump|miss2_dunk|miss3_jump')) & (Events['LTeamID']==Events['TeamID']) , 1, 0)

    Events['LFGM3'] = np.where(

            ( Events.EventType.str.contains('made3_jump')) & (Events['LTeamID']==Events['TeamID']) , 1, 0)

    Events['LFGA3'] = np.where(

            ( Events.EventType.str.contains('made3_jump|miss3_jump')) & (Events['LTeamID']==Events['TeamID']) , 1, 0)

    Events['LFTM'] = np.where(

            ( Events.EventType.str.contains('made1_free')) & (Events['LTeamID']==Events['TeamID']) , 1, 0)

    Events['LFTA'] = np.where(

            ( Events.EventType.str.contains('made1_free|miss1_free')) & (Events['LTeamID']==Events['TeamID']) , 1, 0)

    Events['LOR'] = np.where(

            ( Events.EventType.str.contains('reb_off')) & (Events['LTeamID']==Events['TeamID']) , 1, 0)

    Events['LDR'] = np.where(

            ( Events.EventType.str.contains('reb_def')) & (Events['LTeamID']==Events['TeamID']) , 1, 0)

    Events['LAst'] = np.where(

            ( Events.EventType.str.contains('assist')) & (Events['LTeamID']==Events['TeamID']) , 1, 0)

    Events['LTO'] = np.where(

            ( Events.EventType.str.contains('turnover')) & (Events['LTeamID']==Events['TeamID']) , 1, 0)

    Events['LStl'] = np.where(

            ( Events.EventType.str.contains('steal')) & (Events['LTeamID']==Events['TeamID']) , 1, 0)

    Events['LBlk'] = np.where(

            ( Events.EventType.str.contains('block')) & (Events['LTeamID']==Events['TeamID']) , 1, 0)

    Events['LPF'] = np.where(

            ( Events.EventType.str.contains('foul_pers')) & (Events['LTeamID']==Events['TeamID']) , 1, 0)

    return Events

Events = deriveRawStats(Events)
#Expanding function will keep a running score for the Winning and losing score in descending order.

Rollup_WScore_LScore = Events.groupby(['Season', 'DayNum', 'WTeamID', 'LTeamID'])[['WScore', 'LScore']].expanding().sum()

Rollup_WScore_LScore = Rollup_WScore_LScore.reset_index()

Rollup_WScore_LScore = Rollup_WScore_LScore.add_suffix('_rollup')

Rollup_WScore_LScore.head(15)
#Join rolled up scores with the original dataset

Events = Events.join(Rollup_WScore_LScore)

Events.head(5)
#Derive elapsed Min from ElapsedSeconds

Events  ['ElapsedMin']  = Events.ElapsedSeconds/ 60

Events  ['ElapsedMin'] = round(Events  ['ElapsedMin'])

#Identify games where at the 35 minute mark, the game different between two teams are within 5 points.

CloseGames =  Events.loc[((Events['WScore_rollup'] - Events['LScore_rollup']<= 5) & (Events['WScore_rollup'] - Events['LScore_rollup']>= -1*5)) &(Events['ElapsedMin'] == 35.0) ]

CloseGames = CloseGames[['Season', 'DayNum','WTeamID', 'LTeamID']]

CloseGames = CloseGames.drop_duplicates()
#Pull all stats for identified close games

CloseScore = pd.merge(Events, CloseGames, how='inner', 

                        left_on=['Season', 'DayNum','WTeamID', 'LTeamID'],

                        right_on=['Season', 'DayNum','WTeamID', 'LTeamID'])

CloseScore = CloseScore.query('ElapsedMin >= 35.0')

#Pull OT info into the CloseScore dataframe for to later derive total game minutes played

Overtime = RegularSeasonDetailedResults[['Season', 'DayNum','WTeamID', 'LTeamID', 'NumOT']]

CloseScore = pd.merge(CloseScore, Overtime, how='inner', 

                      left_on=['Season', 'DayNum','WTeamID', 'LTeamID'],

                      right_on=['Season', 'DayNum','WTeamID', 'LTeamID'])

#Add GameNum so it can later help derive the game mins within a later section

CloseScore['GameNum'] = 1

RegularSeasonDetailedResults['GameNum'] = 1
def setWinAndLoseTeamsRecords(RegularSeasonDetailedResults):

    #Convert the data frame from one record per game to one record per team-game

    regSesW = RegularSeasonDetailedResults.rename(columns = {'WTeamID': 'TEAMID',

                                                             'WScore': 'SCORE',

                                                             'WFGM': 'FGM',

                                                             'WFGA': 'FGA',

                                                             'WFGM3': 'FGM3',

                                                             'WFGA3': 'FGA3',

                                                             'WFTM': 'FTM',

                                                             'WFTA': 'FTA',

                                                             'WOR': 'OR',

                                                             'WDR': 'DR',

                                                             'WAst': 'AST',

                                                             'WTO': 'TO',

                                                             'WStl': 'STL',

                                                             'WBlk': 'BLK',

                                                             'WPF': 'PF',

                                                             'LTeamID': 'O_TEAMID',

                                                             'LScore': 'O_SCORE',

                                                             'LFGM': 'O_FGM',

                                                             'LFGA': 'O_FGA',

                                                             'LFGM3': 'O_FGM3',

                                                             'LFGA3': 'O_FGA3',

                                                             'LFTM': 'O_FTM',

                                                             'LFTA': 'O_FTA',

                                                             'LOR': 'O_OR',

                                                             'LDR': 'O_DR',

                                                             'LAst': 'O_AST',

                                                             'LTO': 'O_TO',

                                                             'LStl': 'O_STL',

                                                             'LBlk': 'O_BLK',

                                                             'LPF': 'O_PF'

                                                            })



    regSesL = RegularSeasonDetailedResults.rename(columns = {'LTeamID': 'TEAMID',

                                                             'LScore': 'SCORE',

                                                             'LFGM': 'FGM',

                                                             'LFGA': 'FGA',

                                                             'LFGM3': 'FGM3',

                                                             'LFGA3': 'FGA3',

                                                             'LFTM': 'FTM',

                                                             'LFTA': 'FTA',

                                                             'LOR': 'OR',

                                                             'LDR': 'DR',

                                                             'LAst': 'AST',

                                                             'LTO': 'TO',

                                                             'LStl': 'STL',

                                                             'LBlk': 'BLK',

                                                             'LPF': 'PF',



                                                             'WTeamID': 'O_TEAMID',

                                                             'WScore': 'O_SCORE',

                                                             'WFGM': 'O_FGM',

                                                             'WFGA': 'O_FGA',

                                                             'WFGM3': 'O_FGM3',

                                                             'WFGA3': 'O_FGA3',

                                                             'WFTM': 'O_FTM',

                                                             'WFTA': 'O_FTA',

                                                             'WOR': 'O_OR',

                                                             'WDR': 'O_DR',

                                                             'WAst': 'O_AST',

                                                             'WTO': 'O_TO',

                                                             'WStl': 'O_STL',

                                                             'WBlk': 'O_BLK',

                                                             'WPF': 'O_PF',

                                                             })



    regSes = (regSesW, regSesL)

    regSes = pd.concat(regSes, ignore_index = True, sort = False)

    regSes = regSes[['Season','TEAMID', 'DayNum', 'SCORE', 'O_TEAMID', 'O_SCORE',

                 'FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA', 'OR', 'DR',

                 'AST', 'TO', 'STL', 'BLK', 'PF', 'NumOT',

                 'O_FGM', 'O_FGA', 'O_FGM3', 'O_FGA3', 'O_FTM', 'O_FTA', 'O_OR', 'O_DR',

                 'O_AST', 'O_TO', 'O_STL', 'O_BLK', 'O_PF', 'GameNum'

                 ]]

    return regSes
closeSes = setWinAndLoseTeamsRecords(CloseScore)

regSes = setWinAndLoseTeamsRecords(RegularSeasonDetailedResults)
def aggregateRawData(regSes):    

    regSes_Avg = regSes.groupby(['Season', 'TEAMID'])['SCORE','O_SCORE',  'FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA', 'OR', 'DR', 

                 'AST', 'TO', 'STL', 'BLK', 'PF',

                 'O_FGM', 'O_FGA', 'O_FGM3', 'O_FGA3', 'O_FTM', 'O_FTA', 'O_OR', 'O_DR', 

                 'O_AST', 'O_TO', 'O_STL', 'O_BLK', 'O_PF', 'NumOT', 'GameNum'

                               ].agg('sum').reset_index()

    return regSes_Avg
closeSes_aggregate = aggregateRawData(closeSes)

regSes_aggregate = aggregateRawData(regSes)
def GetAdvancedStats(Game_Min,NCAA_features):

    NCAA_features ['EFG']       = (NCAA_features ['FGM'] + (NCAA_features ['FGM3']*0.5))/NCAA_features ['FGA']

    NCAA_features ['TOV']       = NCAA_features ['TO']/((NCAA_features ['FGA'] + 0.44) + (NCAA_features ['FTA']+NCAA_features ['TO']))

    NCAA_features ['ORB']       = NCAA_features ['OR']/(NCAA_features ['OR'] + NCAA_features ['O_DR'])

    NCAA_features ['DRB']       = NCAA_features ['DR']/(NCAA_features ['DR'] + NCAA_features ['O_OR'])

    NCAA_features ['FTAR']      = NCAA_features ['FTA']/(NCAA_features ['FGA'])

    NCAA_features ['TS']        = NCAA_features ['SCORE']/((NCAA_features ['FGA']*2) + (0.88 * NCAA_features ['FTA']))

    NCAA_features ['ASTTO']     = (NCAA_features ['AST']/(NCAA_features ['TO']))

    NCAA_features ['ASTR']      = (NCAA_features ['AST'] * 100) / ( (NCAA_features ['FGA'] + (NCAA_features ['FTA']*0.44)) + NCAA_features ['AST'] + NCAA_features ['TO'] )

    NCAA_features ['TR']        = NCAA_features ['OR'] + NCAA_features ['DR']

    NCAA_features ['O_TR']      = NCAA_features ['O_OR'] + NCAA_features ['O_DR']

    NCAA_features ['REBP']      = 100 * (NCAA_features ['TR']) / (NCAA_features ['TR'] + NCAA_features ['O_TR'])

    NCAA_features ['POSS']      = 0.5 * ((NCAA_features ['FGA'] + 0.4 * NCAA_features ['FTA'] - 1.07 * (NCAA_features ['OR'] / (NCAA_features ['OR'] + NCAA_features ['O_DR'])) * (NCAA_features ['FGA'] - NCAA_features ['FGM']) + NCAA_features ['TO']) + (NCAA_features ['O_FGA'] + 0.4 * NCAA_features ['O_FTA'] - 1.07 * (NCAA_features ['O_OR'] / (NCAA_features ['O_OR'] + NCAA_features ['DR'])) * (NCAA_features ['O_FGA'] - NCAA_features ['O_FGM']) + NCAA_features ['O_TO']))

    NCAA_features ['O_POSS']    = 0.5 * ((NCAA_features ['O_FGA'] + 0.4 * NCAA_features ['O_FTA'] - 1.07 * (NCAA_features ['O_OR'] / (NCAA_features ['O_OR'] + NCAA_features ['DR'])) * (NCAA_features ['O_FGA'] - NCAA_features ['O_FGM']) + NCAA_features ['O_TO']) + (NCAA_features ['FGA'] + 0.4 * NCAA_features ['FTA'] - 1.07 * (NCAA_features ['OR'] / (NCAA_features ['OR'] + NCAA_features ['O_DR'])) * (NCAA_features ['FGA'] - NCAA_features ['FGM']) + NCAA_features ['TO']))

    NCAA_features ['GM']        = (Game_Min*NCAA_features ['GameNum']) + (5*NCAA_features ['NumOT'])

    NCAA_features ['PACE']      = 40 * ((NCAA_features ['POSS'] ) / (2 * (NCAA_features ['GM'] / 5)))

    NCAA_features ['DRTG']      = 100* (NCAA_features ['O_SCORE']/NCAA_features ['POSS'])

    NCAA_features ['ORTG']      = 100* (NCAA_features ['SCORE']/(NCAA_features ['O_POSS']))

    NCAA_features ['OFF3']      = (NCAA_features ['FGM3']/(NCAA_features ['FGA3']))

    NCAA_features ['DEF3']      = (NCAA_features ['O_FGM3']/(NCAA_features ['O_FGA3']))

    NCAA_features ['O_EFG']     = (NCAA_features ['O_FGM'] + (NCAA_features ['O_FGM3']*0.5))/NCAA_features ['O_FGA']

    NCAA_features ['O_TOV']     = NCAA_features ['O_TO']/((NCAA_features ['O_FGA'] + 0.44) + (NCAA_features ['O_FTA']+NCAA_features ['O_TO']))

    NCAA_features ['DEFRTG']    = 100*NCAA_features ['O_SCORE']/(NCAA_features ['O_FGA'] + NCAA_features ['O_TO'] + (0.44* NCAA_features ['O_FTA']) - NCAA_features ['O_OR'])

    NCAA_features ['OFFRTG']    = 100*NCAA_features ['SCORE']/(NCAA_features ['FGA'] + NCAA_features ['TO'] + (0.44*NCAA_features ['FTA']) - NCAA_features ['OR'])

    NCAA_features ['TOR']       = (NCAA_features ['TO'] * 100) / (NCAA_features ['FGA'] + (NCAA_features ['FTA'] * 0.44) + NCAA_features ['AST'] + NCAA_features ['TO'])

    NCAA_features ['STLTO']     = NCAA_features ['STL']/NCAA_features ['TO']

    NCAA_features ['PIE']       = (NCAA_features ['SCORE'] + NCAA_features ['FGM'] + NCAA_features ['FTM'] - NCAA_features ['FGA']  - NCAA_features ['FTA']  + NCAA_features ['DR'] + (.5 * NCAA_features ['OR']) + NCAA_features ['AST'] + NCAA_features ['STL'] + (.5 * NCAA_features ['BLK']) - NCAA_features ['PF'] - NCAA_features ['TO']) / ((NCAA_features ['SCORE'] + NCAA_features ['FGM'] + NCAA_features ['FTM'] - NCAA_features ['FGA']  - NCAA_features ['FTA']  + NCAA_features ['DR'] + (.5 * NCAA_features ['OR']) + NCAA_features ['AST'] + NCAA_features ['STL'] + (.5 * NCAA_features ['BLK']) - NCAA_features ['PF'] - NCAA_features ['TO'])  + (NCAA_features ['O_SCORE'] + NCAA_features ['O_FGM'] + NCAA_features ['O_FTM'] - NCAA_features ['O_FGA'] - NCAA_features ['O_FTA'] + NCAA_features ['O_DR'] + (.5 * NCAA_features ['O_OR']) + NCAA_features ['O_AST'] + NCAA_features ['O_STL'] + (.5 * NCAA_features ['O_BLK']) - NCAA_features ['O_PF'] - NCAA_features ['O_TO']))

    NCAA_features ['O_STLTO']   = NCAA_features ['O_STL']/NCAA_features ['O_TO']

    NCAA_features ['O_TOR']     = (NCAA_features ['O_TO'] * 100) / (NCAA_features ['O_FGA'] + (NCAA_features ['O_FTA'] * 0.44) + NCAA_features ['O_AST'] + NCAA_features ['O_TO'])

    NCAA_features ['O_FTAR']    = NCAA_features ['O_FTA']/(NCAA_features ['O_FGA'])

    NCAA_features ['O_TS']      =  NCAA_features ['O_SCORE']/((NCAA_features ['O_FGA']*2) + (0.88 * NCAA_features ['O_FTA']))

    NCAA_features ['O_ASTTO']   = (NCAA_features ['O_AST']/(NCAA_features ['O_TO']))

    NCAA_features ['O_ASTR']    = (NCAA_features ['O_AST'] * 100) / ( (NCAA_features ['O_FGA'] + (NCAA_features ['O_FTA']*0.44)) + NCAA_features ['O_AST'] + NCAA_features ['O_TO'] )

    return NCAA_features
closeSes_stats = GetAdvancedStats(5, closeSes_aggregate)

regSes_stats = GetAdvancedStats(40, regSes_aggregate)
def NCAASetWinAndLoseTeamsRecords(NCAATourneyCompactResults):

    #Convert the data frame from one record per game to one record per team-game

    NCAA_res_w = NCAATourneyCompactResults.rename(columns = {'WTeamID': 'NCAA_TEAMID',

                                                           'LTeamID': 'NCAA_O_TEAMID',

                                                           'WScore':'NCAA_SCORE',

                                                           'LScore':'NCAA_O_SCORE'

                                                             })

    NCAA_res_l = NCAATourneyCompactResults.rename(columns = {'LTeamID': 'NCAA_TEAMID',

                                                           'WTeamID': 'NCAA_O_TEAMID',

                                                           'LScore':'NCAA_SCORE',

                                                           'WScore':'NCAA_O_SCORE'

                                                             })



    NCAA_Ses = (NCAA_res_w, NCAA_res_l)

    NCAA_Ses = pd.concat(NCAA_Ses, ignore_index = True, sort = False)

    #Derive the outcome of who won[1] or loss[0]

    NCAA_Ses ['OUTCOME'] = np.where(NCAA_Ses['NCAA_SCORE']>NCAA_Ses['NCAA_O_SCORE'], 1, 0)

    NCAA_Ses = NCAA_Ses[['Season','NCAA_TEAMID', 'NCAA_O_TEAMID', 'OUTCOME']]

    return NCAA_Ses
NCAATourneyCompactResults = pd.read_csv(in_path + 'datafiles/NCAATourneyCompactResults.csv')

NCAA_Ses = NCAASetWinAndLoseTeamsRecords(NCAATourneyCompactResults)
#join ncaa and regSes_adStats ses data for primary team

NCAA_reg = pd.merge(NCAA_Ses, regSes_stats, how='inner', 

                   left_on=['Season', 'NCAA_TEAMID'], 

                   right_on=['Season', 'TEAMID'])

#join to add your opponent's regSes_adStats 

NCAA_reg = pd.merge(NCAA_reg, regSes_stats, how='inner',

                    left_on=['Season', 'NCAA_O_TEAMID'],

                    right_on=['Season', 'TEAMID'], suffixes =['', '_op'] )

NCAA_reg.shape
NCAA_all = pd.merge(NCAA_reg, closeSes_stats, how='inner', 

                        left_on=['Season', 'NCAA_TEAMID'],

                        right_on=['Season', 'TEAMID'] , suffixes = ['', '_closegamestats'])

NCAA_all = pd.merge(NCAA_all, closeSes_stats, how='inner', 

                        left_on=['Season', 'NCAA_O_TEAMID'],

                        right_on=['Season', 'TEAMID'], suffixes =['', '_closegamestats_op'] )

NCAA_all = NCAA_all.dropna()
#Add Seed

Seeds = pd.read_csv(in_path + 'datafiles/NCAATourneySeeds.csv')

Seeds['Seed'] = Seeds.Seed.str.replace('[a-zA-Z]', '')

Seeds['Seed']=Seeds['Seed'].astype('int64')

NCAA_all = pd.merge(NCAA_all, Seeds, how='inner', 

                    left_on=['Season', 'NCAA_TEAMID'],

                    right_on=['Season', 'TeamID'])

NCAA_all = pd.merge(NCAA_all, Seeds, how='inner', 

                    left_on=['Season', 'NCAA_O_TEAMID'],

                    right_on=['Season', 'TeamID'], suffixes =['', '_op'] )

NCAA_all.head(5)
#import dataframe run locally

NCAA_all = pd.read_csv('../input/ncaa-all/NCAA_all.csv')

NCAA_all.head(5)
def print_score(m):

    print ("train score :", m.score(X_train, y_train))

    print ("test score :", m.score(X_valid, y_valid))

    print ("train log loss :", metrics.log_loss(y_train['OUTCOME'].tolist(),m.predict_proba(X_train).tolist(), eps=1e-15))

    print ("valid log loss :", metrics.log_loss(y_valid['OUTCOME'].tolist(),m.predict_proba(X_valid).tolist(), eps=1e-15))

 
train = NCAA_all[ (NCAA_all.Season <= 2017)]

test = NCAA_all[NCAA_all.Season == 2018]



X_train = train[['Seed', 'Seed_op','PIE', 'EFG','O_EFG','TOV','O_TOV','ORB','DRB','FTAR','O_FTAR','PIE_op','EFG_op','O_EFG_op','TOV_op','O_TOV_op','ORB_op','DRB_op','FTAR_op','O_FTAR_op']]   

y_train = train[['OUTCOME']]

X_valid = test[['Seed', 'Seed_op','PIE', 'EFG','O_EFG','TOV','O_TOV','ORB','DRB','FTAR','O_FTAR','PIE_op','EFG_op','O_EFG_op','TOV_op','O_TOV_op','ORB_op','DRB_op','FTAR_op','O_FTAR_op']]

y_valid = test[['OUTCOME']]    

m = RandomForestClassifier(n_estimators=200, min_samples_leaf=3, 

                           n_jobs=-1, oob_score=True, random_state=0)

m.fit(X_train, y_train) 

print_score (m)
train = NCAA_all[ (NCAA_all.Season <= 2017)]

test = NCAA_all[NCAA_all.Season == 2018]



X_train = train[[ 'EFG_closegamestats','O_EFG_closegamestats','TOV_closegamestats','O_TOV_closegamestats','ORB_closegamestats','DRB_closegamestats','FTAR_closegamestats','O_FTAR_closegamestats','EFG_closegamestats_op','O_EFG_closegamestats_op','TOV_closegamestats_op','O_TOV_closegamestats_op','ORB_closegamestats_op','DRB_closegamestats_op','FTAR_closegamestats_op','O_FTAR_closegamestats_op']]   

y_train = train[['OUTCOME']]

X_valid = test[['EFG_closegamestats','O_EFG_closegamestats','TOV_closegamestats','O_TOV_closegamestats','ORB_closegamestats','DRB_closegamestats','FTAR_closegamestats','O_FTAR_closegamestats','EFG_closegamestats_op','O_EFG_closegamestats_op','TOV_closegamestats_op','O_TOV_closegamestats_op','ORB_closegamestats_op','DRB_closegamestats_op','FTAR_closegamestats_op','O_FTAR_closegamestats_op']]

y_valid = test[['OUTCOME']]    

m = RandomForestClassifier(n_estimators=200, min_samples_leaf=3, 

                           n_jobs=-1, oob_score=True, random_state=0)

m.fit(X_train, y_train) 

print_score (m)
train = NCAA_all[ (NCAA_all.Season <= 2017)]

test = NCAA_all[NCAA_all.Season == 2018]



X_train = train[[ 'Seed','Seed_op','PIE', 'PIE_op', 'EFG','O_EFG','TOV','O_TOV','ORB','DRB','FTAR','O_FTAR','EFG_op','O_EFG_op','TOV_op','O_TOV_op','ORB_op','DRB_op','FTAR_op','O_FTAR_op','EFG_closegamestats','O_EFG_closegamestats','TOV_closegamestats','O_TOV_closegamestats','ORB_closegamestats','DRB_closegamestats','FTAR_closegamestats','O_FTAR_closegamestats','EFG_closegamestats_op','O_EFG_closegamestats_op','TOV_closegamestats_op','O_TOV_closegamestats_op','ORB_closegamestats_op','DRB_closegamestats_op','FTAR_closegamestats_op','O_FTAR_closegamestats_op']]   

y_train = train[['OUTCOME']]

X_valid = test[[ 'Seed','Seed_op','PIE', 'PIE_op','EFG','O_EFG','TOV','O_TOV','ORB','DRB','FTAR','O_FTAR','EFG_op','O_EFG_op','TOV_op','O_TOV_op','ORB_op','DRB_op','FTAR_op','O_FTAR_op','EFG_closegamestats','O_EFG_closegamestats','TOV_closegamestats','O_TOV_closegamestats','ORB_closegamestats','DRB_closegamestats','FTAR_closegamestats','O_FTAR_closegamestats','EFG_closegamestats_op','O_EFG_closegamestats_op','TOV_closegamestats_op','O_TOV_closegamestats_op','ORB_closegamestats_op','DRB_closegamestats_op','FTAR_closegamestats_op','O_FTAR_closegamestats_op']]

y_valid = test[['OUTCOME']]    

m = RandomForestClassifier(n_estimators=200, min_samples_leaf=3, 

                           n_jobs=-1, oob_score=True, random_state=0)

m.fit(X_train, y_train) 

print_score (m)
train = NCAA_all[ (NCAA_all.Season <= 2017)]

test = NCAA_all[NCAA_all.Season == 2018]



X_train = train[[ 'Seed','Seed_op','PIE', 'PIE_op', 'EFG_closegamestats','O_EFG_closegamestats','TOV_closegamestats','O_TOV_closegamestats','ORB_closegamestats','DRB_closegamestats','FTAR_closegamestats','O_FTAR_closegamestats','EFG_closegamestats_op','O_EFG_closegamestats_op','TOV_closegamestats_op','O_TOV_closegamestats_op','ORB_closegamestats_op','DRB_closegamestats_op','FTAR_closegamestats_op','O_FTAR_closegamestats_op']]   

y_train = train[['OUTCOME']]

X_valid = test[[ 'Seed','Seed_op','PIE', 'PIE_op','EFG_closegamestats','O_EFG_closegamestats','TOV_closegamestats','O_TOV_closegamestats','ORB_closegamestats','DRB_closegamestats','FTAR_closegamestats','O_FTAR_closegamestats','EFG_closegamestats_op','O_EFG_closegamestats_op','TOV_closegamestats_op','O_TOV_closegamestats_op','ORB_closegamestats_op','DRB_closegamestats_op','FTAR_closegamestats_op','O_FTAR_closegamestats_op']]

y_valid = test[['OUTCOME']]    

m = RandomForestClassifier(n_estimators=200, min_samples_leaf=3, 

                           n_jobs=-1, oob_score=True, random_state=0)

m.fit(X_train, y_train) 

print_score (m)