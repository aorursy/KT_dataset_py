# ------------------------- #
# --- Import librairies --- #

import numpy as np 
import pandas as pd 
from scipy.stats import zscore
from sklearn.ensemble import RandomForestClassifier

from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
from plotly.graph_objs import * 

init_notebook_mode()
# -------------------- #
# --- Data section --- # 

# --- Players data 

WPlayers = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament/WPlayers.csv')

# --- Teams data 

WTeams = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament/WDataFiles_Stage1/WTeams.csv')
WTeamConferences = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament/WDataFiles_Stage1/WTeamConferences.csv')

# --- Seasons data (starting 1998)

WSeasons = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament/WDataFiles_Stage1/WSeasons.csv')

# --- Seasons tourney seeds & slots (starting 1998)

WncaaSeeds = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament/WDataFiles_Stage1/WNCAATourneySeeds.csv')
WncaaSlots = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament/WDataFiles_Stage1/WNCAATourneySlots.csv')

# --- Regular seasons data (compact since 1998, detailed since 2010)

WrsCompactResults = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament/WDataFiles_Stage1/WRegularSeasonCompactResults.csv')
WrsDetailedResults = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament/WDataFiles_Stage1/WRegularSeasonDetailedResults.csv')

# --- NCAA data (compact since 1998, detailed since 2010)

WncaaCompactResults = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament/WDataFiles_Stage1/WNCAATourneyCompactResults.csv')
WncaaDetailedResults = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament/WDataFiles_Stage1/WNCAATourneyDetailedResults.csv')

# --- Cities data (starting 2010)

WGameCities = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament/WDataFiles_Stage1/WGameCities.csv')
Cities = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament/WDataFiles_Stage1/Cities.csv')

# --- Submission files

SubmissionsStage1 = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament/WSampleSubmissionStage1_2020.csv')
# --- Define usefull functions

def logloss(true_label, predicted, eps=1e-15):
    """
        Compute the logloss value of a specific prediction
    """
    
    p = np.clip(predicted, eps, 1 - eps)
    if true_label == 1:
        return -np.log(p)
    return -np.log(1 - p)

def reduce_mem_usage(df, verbose=True):
    """
        Usefull function to reduce the memory consumed by a dataframe
    """
    
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df

def compute_round(df):
    """
        Compute the exact value of a round for a specific ncaa game
    """
    
    df['Round'] = 0 

    for i in df.index: 

        if df['Season'][i]<2003 :

            if df['DayNum'][i] == 137 :
                df['Round'][i]= 1 
            elif df['DayNum'][i] == 138: 
                df['Round'][i]= 1 

            elif df['DayNum'][i] == 139 :
                df['Round'][i]= 2 
            elif df['DayNum'] [i] == 140 :
                df['Round'][i]= 2 

            elif df['DayNum'][i] ==145 :
                df['Round'][i]= 3 
            elif df['DayNum'][i] ==147 :
                df['Round'][i]= 4 
            elif df['DayNum'][i] ==151: 
                df['Round'][i]= 5
            else: #df['DayNum'][i]==153:
                df['Round'][i]= 6


        else :   
            df['Round'][i] = 0 
            if df['Season'][i]<2015 : 
                if df['DayNum'][i] ==138 :
                    df['Round'][i]= 1 
                elif df['DayNum'][i] ==139: 
                    df['Round'][i]= 1
                elif df['DayNum'][i] == 140 :
                    df['Round'][i]= 2
                elif df['DayNum'][i] ==141:
                    df['Round'][i]= 2 
                elif df['DayNum'][i] ==145 :
                    df['Round'][i]= 3 
                elif df['DayNum'][i] ==146:
                    df['Round'][i]= 3 
                elif df['DayNum'][i] ==147:
                    df['Round'][i]= 4
                elif df['DayNum'][i] ==148:
                    df['Round'][i]= 4 
                elif df['DayNum'][i] ==153: 
                    df['Round'][i]= 5
                else: #df['DayNum'][i]==155:
                    df['Round'][i]= 6

            else :  
                if df['Season'][i]<2017 : 

                    if df['DayNum'][i] ==137:
                        df['Round'][i]= 1
                    elif df['DayNum'][i] ==138:
                        df['Round'][i]= 1 
                    elif df['DayNum'][i] ==139 or df['DayNum'][i] ==140:
                        df['Round'][i]= 2 
                    elif df['DayNum'][i] ==144 or df['DayNum'][i] ==145:
                        df['Round'][i]= 3 
                    elif df['DayNum'][i] ==146 or df['DayNum'][i] ==147:
                        df['Round'][i]= 4 
                    elif df['DayNum'][i] ==153: 
                        df['Round'][i]= 5
                    else: # df['DayNum'][i]==155:
                        df['Round'][i]= 6

                else : 
                    if df['DayNum'][i] ==137 or df['DayNum'][i] ==138:
                        df['Round'][i]= 1 
                    elif df['DayNum'][i] ==139 or df['DayNum'][i] ==140:
                        df['Round'][i]= 2 
                    elif df['DayNum'][i] ==144 or df['DayNum'][i] ==145:
                        df['Round'][i]= 3 
                    elif df['DayNum'][i] ==146 or df['DayNum'][i] ==147:
                        df['Round'][i]= 4 
                    elif df['DayNum'][i] ==151: 
                        df['Round'][i]= 5
                    else: 
                        df['Round'][i]= 6
    return df
# ------------------------ #
# --- Merging datasets --- #

WrsCompactResults = reduce_mem_usage(WrsCompactResults)
WGameCities = reduce_mem_usage(WGameCities)
WTeams = reduce_mem_usage(WTeams)

WGameCities = WGameCities.merge(Cities, how = 'left', on = 'CityID')
WTeams = WTeams.merge(WPlayers, how = 'inner', on = 'TeamID')

WrsCompactResults = WrsCompactResults\
                        .merge(WncaaSeeds, how = 'inner', left_on = ['WTeamID', 'Season'], right_on = ['TeamID', 'Season'])\
                        .drop('TeamID', axis = 1)\
                        .rename(columns = {'Seed' : 'WSeed'})\
                        .merge(WncaaSeeds, how = 'inner', left_on = ['LTeamID', 'Season'], right_on = ['TeamID', 'Season'])\
                        .drop('TeamID', axis = 1)\
                        .rename(columns = {'Seed' : 'LSeed'})

WrsCompactResults['SeedDiff'] = WrsCompactResults.apply(lambda row : int(row['WSeed'][-2:]) - int(row['LSeed'][-2:]), axis = 1)
WrsCompactResults['ScoreDiff'] = WrsCompactResults.apply(lambda row : int(row['WScore']) - int(row['LScore']), axis = 1)
WrsCompactResults['ExpectedWin'] = WrsCompactResults.SeedDiff.apply(lambda x : x < 0)
WTeams.head()
WPlayers.head()
Wrs_res = WrsCompactResults.groupby('Season').ExpectedWin.mean()

iplot(
    Figure(
        data = [Scatter(x = Wrs_res.index , y = Wrs_res.values)],
        layout = Layout(
            title = 'Expected win rates (WSeed > LSeed) evolution [REGULAR SEASON]',
            yaxis = dict(title = 'Expected win rate', range = [0.4, 0.8]),
            xaxis = dict(title = 'Season')
        )
    )
)
WncaaCompactResults = WncaaCompactResults\
                        .merge(WncaaSeeds, how = 'inner', left_on = ['WTeamID', 'Season'], right_on = ['TeamID', 'Season'])\
                        .drop('TeamID', axis = 1)\
                        .rename(columns = {'Seed' : 'WSeed'})\
                        .merge(WncaaSeeds, how = 'inner', left_on = ['LTeamID', 'Season'], right_on = ['TeamID', 'Season'])\
                        .drop('TeamID', axis = 1)\
                        .rename(columns = {'Seed' : 'LSeed'})

WncaaCompactResults['SeedDiff'] = WncaaCompactResults.apply(lambda row : int(row['WSeed'][-2:]) - int(row['LSeed'][-2:]), axis = 1)
WncaaCompactResults['ScoreDiff'] = WncaaCompactResults.apply(lambda row : int(row['WScore']) - int(row['LScore']), axis = 1)
WncaaCompactResults['ExpectedWin'] = WncaaCompactResults.SeedDiff.apply(lambda x : x < 0)
WncaaCompactResults.head()
Wncaa_res = WncaaCompactResults.groupby('Season').ExpectedWin.mean()

iplot(
    Figure(
        data = [Scatter(x = Wncaa_res.index , y = Wncaa_res.values)],
        layout = Layout(
            title = 'Expected win rates (WSeed > LSeed) evolution [NCAA]',
            yaxis = dict(title = 'Expected win rate', range = [0.4, 0.9]),
            xaxis = dict(title = 'Season')
        )
    )
)
# --- [Regular Season PROCESSING]

WrsDetailedResults = WrsDetailedResults\
                        .merge(WncaaSeeds, how = 'inner', left_on = ['WTeamID', 'Season'], right_on = ['TeamID', 'Season'])\
                        .drop('TeamID', axis = 1)\
                        .rename(columns = {'Seed' : 'WSeed'})\
                        .merge(WncaaSeeds, how = 'inner', left_on = ['LTeamID', 'Season'], right_on = ['TeamID', 'Season'])\
                        .drop('TeamID', axis = 1)\
                        .rename(columns = {'Seed' : 'LSeed'})

WrsDetailedResults['SeedDiff'] = WrsDetailedResults.apply(lambda row : int(row['WSeed'][-2:]) - int(row['LSeed'][-2:]), axis = 1)
WrsDetailedResults['ScoreDiff'] = WrsDetailedResults.apply(lambda row : int(row['WScore']) - int(row['LScore']), axis = 1)
WrsDetailedResults['ExpectedWin'] = WrsDetailedResults.SeedDiff.apply(lambda x : x < 0)

# --- Computing the four factors for both teams

WrsDetailedResults['WShooting'] = WrsDetailedResults.apply(lambda row : (row['WFGM'] + 0.5 * row['WFGM3'])  / row['WFGA'], axis = 1)
WrsDetailedResults['WTurnovers'] = WrsDetailedResults.apply(lambda row : row['WTO'] / (row['WFGA'] + 0.44 * row['WFTA'] + row['WTO']), axis = 1)
WrsDetailedResults['WORebounding'] = WrsDetailedResults.apply(lambda row : row['WOR'] / (row['WOR'] + row['LDR']), axis = 1)
WrsDetailedResults['WDRebounding'] = WrsDetailedResults.apply(lambda row : row['WDR'] / (row['WDR'] + row['LOR']), axis = 1)
WrsDetailedResults['WFreeThrows'] = WrsDetailedResults.apply(lambda row : row['WFTA'] / row['WFGA'], axis = 1)

WrsDetailedResults['LShooting'] = WrsDetailedResults.apply(lambda row : (row['LFGM'] + 0.5 * row['LFGM3'])  / row['LFGA'], axis = 1)
WrsDetailedResults['LTurnovers'] = WrsDetailedResults.apply(lambda row : row['LTO'] / (row['LFGA'] + 0.44 * row['LFTA'] + row['LTO']), axis = 1)
WrsDetailedResults['LORebounding'] = WrsDetailedResults.apply(lambda row : row['LOR'] / (row['LOR'] + row['WDR']), axis = 1)
WrsDetailedResults['LDRebounding'] = WrsDetailedResults.apply(lambda row : row['LDR'] / (row['LDR'] + row['WOR']), axis = 1)
WrsDetailedResults['LFreeThrows'] = WrsDetailedResults.apply(lambda row : row['LFTA'] / row['LFGA'], axis = 1)

WrsDetailedResults['WFourFactorsScore'] = 40 * WrsDetailedResults['WShooting']\
                                            - 25 * WrsDetailedResults['WTurnovers']\
                                            + 20 * WrsDetailedResults['WORebounding']\
                                            + 15 * WrsDetailedResults['WFreeThrows']\
                                            - 40 * WrsDetailedResults['LShooting']\
                                            + 25 * WrsDetailedResults['LTurnovers']\
                                            + 20 * WrsDetailedResults['WDRebounding']\
                                            - 10 * WrsDetailedResults['LFreeThrows']

WrsDetailedResults['LFourFactorsScore'] = 40 * WrsDetailedResults['LShooting']\
                                            - 25 * WrsDetailedResults['LTurnovers']\
                                            + 20 * WrsDetailedResults['LORebounding']\
                                            + 15 * WrsDetailedResults['LFreeThrows']\
                                            - 40 * WrsDetailedResults['WShooting']\
                                            + 25 * WrsDetailedResults['WTurnovers']\
                                            + 20 * WrsDetailedResults['LDRebounding']\
                                            - 10 * WrsDetailedResults['WFreeThrows']

# --- The idea is to compute a quantity that is related to the a team's state of tiredness 

#WrsDetailedResults['ID'] = WrsDetailedResults['Season'].astype(str) + '_' + WrsDetailedResults['WTeamID'].astype(str) + '_' + WrsDetailedResults['LTeamID'].astype(str)

#OvertimesData = pd.concat([WrsDetailedResults[['Season', 'WTeamID', 'ID', 'NumOT']].rename(columns = {"WTeamID" : 'TeamID'}), WrsDetailedResults[['Season', 'LTeamID', 'ID', 'NumOT']].rename(columns = {"LTeamID" : 'TeamID'})])
#OvertimesData = OvertimesData[['Season', 'TeamID', 'ID', 'NumOT']].groupby(by=['Season','TeamID','ID']).sum().groupby(level=[1]).cumsum().reset_index().rename(columns = {'NumOT' : 'SumOvertimePlayedBeforeGame'})

#WrsDetailedResults = WrsDetailedResults.merge(OvertimesData[['ID', 'SumOvertimePlayedBeforeGame']], how = 'inner', on = 'ID').drop_duplicates()
# We have to split the original dataframe to retrieve data for the wining AND for the loosing team at the same time

WTeamListFeatures = ['WTeamID', 'WScore', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF', 'WShooting', 'WTurnovers', 'WORebounding', 'WDRebounding', 'WFreeThrows', 'WFourFactorsScore']
WTeamListFeaturesNames = dict(zip(WTeamListFeatures, [feature[1:] if feature[0] == 'W' else feature for feature in WTeamListFeatures]))
WTeamDataframe = WrsDetailedResults[WrsDetailedResults.Season <= 2015][WTeamListFeatures].rename(columns = WTeamListFeaturesNames)

LTeamListFeatures = ['LTeamID', 'LScore', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF', 'LShooting', 'LTurnovers', 'LORebounding', 'LDRebounding', 'LFreeThrows', 'LFourFactorsScore']
LTeamListFeaturesNames = dict(zip(LTeamListFeatures, [feature[1:] if feature[0] == 'L' else feature for feature in LTeamListFeatures]))
LTeamDataframe = WrsDetailedResults[WrsDetailedResults.Season <= 2015][LTeamListFeatures].rename(columns = LTeamListFeaturesNames)

RegularSeasonFeatures = pd.concat([WTeamDataframe, LTeamDataframe])
WrsDetailedResults[['WLoc', 'SeedDiff', 'ScoreDiff', 'ExpectedWin', 'WShooting', 'WTurnovers', 'WORebounding', 'WDRebounding', 'WFreeThrows', 'LShooting', 'LTurnovers', 'LORebounding', 'LDRebounding', 'LFreeThrows', 'WFourFactorsScore', 'LFourFactorsScore']].corr()
# --- [NCAA PROCESSING]

WncaaDetailedResults = WncaaDetailedResults\
                        .merge(WncaaSeeds, how = 'inner', left_on = ['WTeamID', 'Season'], right_on = ['TeamID', 'Season'])\
                        .drop('TeamID', axis = 1)\
                        .rename(columns = {'Seed' : 'WSeed'})\
                        .merge(WncaaSeeds, how = 'inner', left_on = ['LTeamID', 'Season'], right_on = ['TeamID', 'Season'])\
                        .drop('TeamID', axis = 1)\
                        .rename(columns = {'Seed' : 'LSeed'})

WncaaDetailedResults['SeedDiff'] = WncaaDetailedResults.apply(lambda row : int(row['WSeed'][-2:]) - int(row['LSeed'][-2:]), axis = 1)
WncaaDetailedResults['ScoreDiff'] = WncaaDetailedResults.apply(lambda row : int(row['WScore']) - int(row['LScore']), axis = 1)
WncaaDetailedResults['ExpectedWin'] = WncaaDetailedResults.SeedDiff.apply(lambda x : x < 0)

# --- Computing the four factors for both teams

WncaaDetailedResults['WShooting'] = WncaaDetailedResults.apply(lambda row : (row['WFGM'] + 0.5 * row['WFGM3'])  / row['WFGA'], axis = 1)
WncaaDetailedResults['WTurnovers'] = WncaaDetailedResults.apply(lambda row : row['WTO'] / (row['WFGA'] + 0.44 * row['WFTA'] + row['WTO']), axis = 1)
WncaaDetailedResults['WORebounding'] = WncaaDetailedResults.apply(lambda row : row['WOR'] / (row['WOR'] + row['LDR']), axis = 1)
WncaaDetailedResults['WDRebounding'] = WncaaDetailedResults.apply(lambda row : row['WDR'] / (row['WDR'] + row['LOR']), axis = 1)
WncaaDetailedResults['WFreeThrows'] = WncaaDetailedResults.apply(lambda row : row['WFTA'] / row['WFGA'], axis = 1)

WncaaDetailedResults['LShooting'] = WncaaDetailedResults.apply(lambda row : (row['LFGM'] + 0.5 * row['LFGM3'])  / row['LFGA'], axis = 1)
WncaaDetailedResults['LTurnovers'] = WncaaDetailedResults.apply(lambda row : row['LTO'] / (row['LFGA'] + 0.44 * row['LFTA'] + row['LTO']), axis = 1)
WncaaDetailedResults['LORebounding'] = WncaaDetailedResults.apply(lambda row : row['LOR'] / (row['LOR'] + row['WDR']), axis = 1)
WncaaDetailedResults['LDRebounding'] = WncaaDetailedResults.apply(lambda row : row['LDR'] / (row['LDR'] + row['WOR']), axis = 1)
WncaaDetailedResults['LFreeThrows'] = WncaaDetailedResults.apply(lambda row : row['LFTA'] / row['LFGA'], axis = 1)

WncaaDetailedResults['WFourFactorsScore'] = 40 * WncaaDetailedResults['WShooting']\
                                            - 25 * WncaaDetailedResults['WTurnovers']\
                                            + 20 * WncaaDetailedResults['WORebounding']\
                                            + 15 * WncaaDetailedResults['WFreeThrows']\
                                            - 40 * WncaaDetailedResults['LShooting']\
                                            + 25 * WncaaDetailedResults['LTurnovers']\
                                            + 20 * WncaaDetailedResults['WDRebounding']\
                                            - 10 * WncaaDetailedResults['LFreeThrows']

WncaaDetailedResults['LFourFactorsScore'] = 40 * WncaaDetailedResults['LShooting']\
                                            - 25 * WncaaDetailedResults['LTurnovers']\
                                            + 20 * WncaaDetailedResults['LORebounding']\
                                            + 15 * WncaaDetailedResults['LFreeThrows']\
                                            - 40 * WncaaDetailedResults['WShooting']\
                                            + 25 * WncaaDetailedResults['WTurnovers']\
                                            + 20 * WncaaDetailedResults['LDRebounding']\
                                            - 10 * WncaaDetailedResults['WFreeThrows']

WncaaDetailedResults = compute_round(WncaaDetailedResults)
WncaaDetailedResults.head()
WncaaDetailedResults[['ExpectedWin', 'WFourFactorsScore', 'LFourFactorsScore']].corr()
TrainData = WrsDetailedResults[WrsDetailedResults.Season <= 2015][['WTeamID', 'LTeamID', 'WScore', 'LScore', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF', 'ScoreDiff', 'SeedDiff', 'WShooting', 'WTurnovers', 'WORebounding', 'WDRebounding', 'WFreeThrows', 'LShooting', 'LTurnovers', 'LORebounding', 'LDRebounding', 'LFreeThrows', 'WFourFactorsScore', 'LFourFactorsScore']]

TrainData['FirstTeamID'] = TrainData[['WTeamID', 'LTeamID']].apply(lambda row : row['WTeamID'] if row['WTeamID'] < row['LTeamID'] else row['LTeamID'], axis = 1)
TrainData['SecondTeamID'] = TrainData[['WTeamID', 'LTeamID']].apply(lambda row : row['WTeamID'] if row['WTeamID'] > row['LTeamID'] else row['LTeamID'], axis = 1)

TrainData
# --- Convert current TrainData into the right format for predictions (we don't want to know who is the winner before the game)

features = list(filter(lambda x : x not in ['FirstTeamID', 'SecondTeamID', 'SeedDiff', 'ScoreDiff', 'TeamID'], list(set(list(map(lambda col : col[1:] if col not in ['FirstTeamID', 'SecondTeamID', 'SeedDiff', 'ScoreDiff'] else col, list(TrainData)))))))

first_team_features = ['FirstTeam' + col for col in features]
second_team_features = ['SecondTeam' + col for col in features]

processed_features = first_team_features + second_team_features + ['SeedDiff', 'ScoreDiff', 'FirstTeamID', 'SecondTeamID', 'Label'] 

train_data_dict = {key : [] for key in processed_features}

for line in TrainData.iterrows():
    if int(line[1].FirstTeamID) == int(line[1].WTeamID) :
        for col in list(TrainData) :
            if (col[0] == 'W' and col != 'WTeamID'):
                train_data_dict['FirstTeam%s'%col[1:]].append(line[1]['%s'%col])
                
            elif (col[0] == 'L' and col != 'LTeamID'):
                train_data_dict['SecondTeam%s'%col[1:]].append(line[1]['%s'%col])
                
        train_data_dict['SeedDiff'].append(line[1].SeedDiff)
        train_data_dict['ScoreDiff'].append(line[1].ScoreDiff)
        train_data_dict['FirstTeamID'].append(line[1].WTeamID)
        train_data_dict['SecondTeamID'].append(line[1].LTeamID)
        train_data_dict['Label'].append(1)
                   
    else :
        for col in list(TrainData) :
            if (col[0] == 'L' and col != 'LTeamID'):
                train_data_dict['SecondTeam%s'%col[1:]].append(line[1]['%s'%col])
                
            elif (col[0] == 'W' and col != 'WTeamID'):
                train_data_dict['FirstTeam%s'%col[1:]].append(line[1]['%s'%col])
                
        train_data_dict['SeedDiff'].append(line[1].SeedDiff)
        train_data_dict['ScoreDiff'].append(line[1].ScoreDiff)
        train_data_dict['FirstTeamID'].append(line[1].LTeamID)
        train_data_dict['SecondTeamID'].append(line[1].WTeamID)
        train_data_dict['Label'].append(0)

CorrectTrainData = pd.DataFrame(train_data_dict)
CorrectTrainData.head()
print('The two classes are relatively well balanced. The first team win with a frequence of %.2f %%'%CorrectTrainData.Label.mean())
basic_features = ['FirstTeamFGM3', 'FirstTeamScore', 'FirstTeamFGA3', 'FirstTeamStl', 'FirstTeamFGM', 'FirstTeamPF', 'FirstTeamFTA', 'FirstTeamAst', 'FirstTeamFGA', 'FirstTeamBlk', 'FirstTeamDR', 'FirstTeamOR', 'FirstTeamTO', 'FirstTeamFTM']
basic_features = basic_features + [feature.replace('First', 'Second') for feature in basic_features] + ['SeedDiff', 'ScoreDiff']
# --- Let's build a model, using a logistic regression, to predict the probability of win for the FirstTeamID

X_basic = CorrectTrainData[basic_features]
X_enhanced = CorrectTrainData[list(CorrectTrainData)[:-3]]

y = CorrectTrainData.Label

basic_RF_model = RandomForestClassifier(n_estimators = 400, random_state = 0).fit(X_basic, y)
enhanced_RF_model = RandomForestClassifier(n_estimators = 400, random_state = 0).fit(X_enhanced, y)
from sklearn.model_selection import RandomizedSearchCV

# --- Performing grid search parameter using K-Fold Cross Validation on different set of paramaters 

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

rf = RandomForestClassifier()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
rf_random.fit(X_basic, y)

rf_random.best_params_
basic_RF_params = rf_random.best_params_
basic_RF_model = RandomForestClassifier(**basic_RF_params).fit(X_basic, y)
basic_preds = [pred[1] for pred in basic_RF_model.predict_proba(X_basic)]
enhanced_preds = [pred[1] for pred in enhanced_RF_model.predict_proba(X_enhanced)]

TrainLabelVersusPreds = CorrectTrainData.copy()[['FirstTeamID', 'SecondTeamID', 'Label']]

TrainLabelVersusPreds['BasicPred'] = basic_preds
TrainLabelVersusPreds['EnhancedPred'] = enhanced_preds

TrainLabelVersusPreds['BasicLogloss'] = TrainLabelVersusPreds.apply(lambda row : logloss(row['Label'], row['BasicPred']), axis = 1)
TrainLabelVersusPreds['EnhancedLogloss'] = TrainLabelVersusPreds.apply(lambda row : logloss(row['Label'], row['EnhancedPred']), axis = 1)

TrainLabelVersusPreds.head()

print('Average logloss of the new basic model on training : %.5f'%TrainLabelVersusPreds.BasicLogloss.mean())
print('\nAverage logloss of the new enhanced model on training : %.5f'%TrainLabelVersusPreds.EnhancedLogloss.mean())
SubmissionsStage1.head()
GroundTruthStage1 = WncaaCompactResults.loc[:, ['Season', 'WTeamID', 'LTeamID']]

GroundTruthStage1['FirstTeamID'] = GroundTruthStage1[['WTeamID', 'LTeamID']].apply(lambda row : row['WTeamID'] if row['WTeamID'] < row['LTeamID'] else row['LTeamID'], axis = 1)
GroundTruthStage1['SecondTeamID'] = GroundTruthStage1[['WTeamID', 'LTeamID']].apply(lambda row : row['WTeamID'] if row['WTeamID'] > row['LTeamID'] else row['LTeamID'], axis = 1)
GroundTruthStage1['ID'] = GroundTruthStage1.apply(lambda row : str(row['Season']) + '_' + str(row['FirstTeamID']) + '_' + str(row['SecondTeamID']), axis = 1)

GroundTruthStage1['Label'] = GroundTruthStage1[['WTeamID', 'FirstTeamID']].apply(lambda row : 1 if row['FirstTeamID'] == row['WTeamID'] else 0, axis = 1)

GroundTruthStage1 = GroundTruthStage1[GroundTruthStage1.Season >= 2010]

GroundTruthStage1.head()
Stage1Seeds = WncaaSeeds[WncaaSeeds.Season > 2014]

SubmissionsStage1 = SubmissionsStage1.drop('Pred', axis = 1)

SubmissionsStage1['Season'] = SubmissionsStage1.ID.apply(lambda x : int(x[:4]))
SubmissionsStage1['FirstTeamID'] = SubmissionsStage1.ID.apply(lambda x : int(x[5:9]))
SubmissionsStage1['SecondTeamID'] = SubmissionsStage1.ID.apply(lambda x : int(x[-4:]))

SubmissionsStage1 = SubmissionsStage1\
                        .merge(Stage1Seeds, how = 'inner', left_on = ['Season', 'FirstTeamID'], right_on = ['Season', 'TeamID'])\
                        .drop('TeamID', axis = 1)\
                        .rename(columns = {'Seed' : 'FirstTeamSeed'})\
                        .merge(Stage1Seeds, how = 'inner', left_on = ['Season', 'SecondTeamID'], right_on = ['Season', 'TeamID'])\
                        .drop('TeamID', axis = 1)\
                        .rename(columns = {'Seed' : 'SecondTeamSeed'})

SubmissionsStage1['SeedDiff'] = SubmissionsStage1.apply(lambda row : int(row['FirstTeamSeed'][-2:]) - int(row['SecondTeamSeed'][-2:]), axis = 1)
SubmissionsStage1 = SubmissionsStage1.merge(GroundTruthStage1[['ID', 'Label']], how = 'inner', on = ['ID'])
TestPresence = SubmissionsStage1.copy()

# --- We will remove all teams from which we don't have any data between 2010 and 2015. This is OF COURSE NOT A THING TO DO as its cheating. We just want to evaluate feature engineering here.

TeamList = TestFeatures.TeamID.tolist()

TestPresence['FirstTeamPresent'] = SubmissionsStage1.FirstTeamID.apply(lambda x : x in TeamList)
TestPresence['SecondTeamPresent'] = SubmissionsStage1.SecondTeamID.apply(lambda x : x in TeamList)

FTeamToRemove = TestPresence[TestPresence.FirstTeamPresent == False].FirstTeamID.unique()
STeamToRemove = TestPresence[TestPresence.SecondTeamPresent == False].SecondTeamID.unique()

TeamToRemove = list(FTeamToRemove) + list(STeamToRemove)

SubmissionsStage1 = SubmissionsStage1[(~SubmissionsStage1.FirstTeamID.isin(TeamToRemove)) & (~SubmissionsStage1.SecondTeamID.isin(TeamToRemove))]
NaiveModelSubmissions = SubmissionsStage1.copy()
NaiveModelSubmissions['Pred'] = NaiveModelSubmissions.SeedDiff.apply(lambda x : 0.5 if x == 0 else 1 if x < -9 else 0.8 if -9 <= x < 0 else 0.2 if 0 < x <= 9 else 0)
NaiveModelSubmissions['LogLoss'] = NaiveModelSubmissions[['Pred', 'Label']].apply(lambda row : logloss(row['Label'], row['Pred']), axis = 1)

NaiveModelSubmissions.head()
print('Average logloss for the naïve model : {}'.format(NaiveModelSubmissions.LogLoss.mean()))
TestFeatures = RegularSeasonFeatures.groupby(['TeamID']).mean().reset_index()
TestFeatures
BetterModelSubmission = SubmissionsStage1.copy()

test_data_dict = {key : [] for key in processed_features[:-1] + ['ID']}

for line in BetterModelSubmission.iterrows():
    
    first_team = line[1].FirstTeamID
    second_team = line[1].SecondTeamID
    
    test_data_dict['FirstTeamID'].append(first_team)
    test_data_dict['SecondTeamID'].append(second_team)
    test_data_dict['ID'].append(line[1].ID)
    test_data_dict['SeedDiff'].append(line[1].SeedDiff)
    
    first_team_data = TestFeatures[TestFeatures.TeamID == first_team]
    second_team_data = TestFeatures[TestFeatures.TeamID == second_team]
    
    for feature in list(TestFeatures)[1:] :
        try :
            test_data_dict['FirstTeam'+feature].append(first_team_data[feature].tolist()[0])
            test_data_dict['SecondTeam'+feature].append(second_team_data[feature].tolist()[0])
        except:
            print(line)

    test_data_dict['ScoreDiff'].append(first_team_data.Score.tolist()[0] - second_team_data.Score.tolist()[0])
    
# Need to compute the ScoreDiff as FirstTeamScore - SecondTeamScore

#for key in test_data_dict.keys():
#    print(key, len(test_data_dict[key]))

TestData = pd.DataFrame(test_data_dict)
TestData = TestData[list(TestData)[:-3]]
basic_test_preds = [pred[1] for pred in basic_RF_model.predict_proba(TestData[basic_features])]
enhanced_test_preds = [pred[1] for pred in enhanced_RF_model.predict_proba(TestData)]

BetterModelSubmission['BasicPred'] = basic_test_preds
BetterModelSubmission['EnhancedPred'] = enhanced_test_preds

BetterModelSubmission['BasicLogLoss'] = BetterModelSubmission.apply(lambda row : logloss(row['Label'], row['BasicPred']), axis = 1)
BetterModelSubmission['EnhancedLogLoss'] = BetterModelSubmission.apply(lambda row : logloss(row['Label'], row['EnhancedPred']), axis = 1)
BetterModelSubmission.BasicLogLoss.mean()
