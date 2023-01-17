import numpy as np
import pandas as pd
import os
import math
import statistics as stats

import sklearn.metrics as mtr
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from tqdm.notebook import trange, tqdm

from keras.layers import Dense,Input,Flatten,concatenate,Dropout,Lambda,BatchNormalization
from keras.models import Model
import keras.backend as K
from keras.callbacks import Callback
from  keras.callbacks import EarlyStopping,ModelCheckpoint
import datetime

import itertools
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial import ConvexHull

from scipy.spatial.distance import euclidean
from scipy.special import expit

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 150)

from kaggle.competitions import nflrush
env = nflrush.make_env()
train = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', dtype={'WindSpeed': 'object'})
#train = train.loc[train.Season != 2017]
    
outcomes = train[['GameId','PlayId','Yards']].drop_duplicates()

train_2 = train.copy()
train_3 = train.copy()
def strtoseconds(txt):
    txt = txt.split(':')
    ans = int(txt[0])*60 + int(txt[1]) + int(txt[2])/60
    return ans

def strtofloat(x):
    try:
        return float(x)
    except:
        return -1

def map_weather(txt):
    ans = 1
    if pd.isna(txt):
        return 0
    if 'partly' in txt:
        ans*=0.5
    if 'climate controlled' in txt or 'indoor' in txt:
        return ans*3
    if 'sunny' in txt or 'sun' in txt:
        return ans*2
    if 'clear' in txt:
        return ans
    if 'cloudy' in txt:
        return -ans
    if 'rain' in txt or 'rainy' in txt:
        return -2*ans
    if 'snow' in txt:
        return -3*ans
    return 0

def OffensePersonnelSplit(x):
    dic = {'DB' : 0, 'DL' : 0, 'LB' : 0, 'OL' : 0, 'QB' : 0, 'RB' : 0, 'TE' : 0, 'WR' : 0}
    for xx in x.split(","):
        xxs = xx.split(" ")
        dic[xxs[-1]] = int(xxs[-2])
    return dic

def DefensePersonnelSplit(x):
    dic = {'DB' : 0, 'DL' : 0, 'LB' : 0, 'OL' : 0}
    for xx in x.split(","):
        xxs = xx.split(" ")
        dic[xxs[-1]] = int(xxs[-2])
    return dic

def orientation_to_cat(x):
    x = np.clip(x, 0, 360 - 1)
    try:
        return str(int(x/15))
    except:
        return "nan"
    
def euclidean_distance(x1,y1,x2,y2):
        x_diff = (x1-x2)**2
        y_diff = (y1-y2)**2
        return np.sqrt(x_diff + y_diff)

def standardize_speed(df):
    #standardize speed by year
    speed_std = np.where(df.Season == '2017', ((df['S'] - 2.4355) / 1.2930),
                           np.where(df.Season == '2018', ((df['S'] - 2.7570) / 1.4551), 
                                   ((df['S'] - 2.7456) / 1.4501)))
    return(speed_std)

#train['S_std'] = standardize_speed(train)
player_schema = {2557850:'De\'Angelo Henderson',2560913:'Tracy Walker',2550565:'Da\'Ron Payne',
                 2556512:'Donte Deayon',2560813:'DJ Moore',2561283:'Zeke Turner',2561544:'Trenton Scott',
                 2506925:'Domata Peko',2539340:'Ricky Wagner',2560828:'Chuks Okorafor'}

team_schema = {'ARZ':'ARI','BLT':'BAL','CLV':'CLE','HST':'HOU'}

def lookup_player_name(player_id, player_name):
    if player_id in player_schema:
        return(player_schema[player_id])
    else:
        return(player_name)

print('Adjusting player names and team abbreviations...')    
def adjust_players_and_teams(df, player_schema, team_schema):
    #adjust player names
    df['DisplayName'] = df.apply(lambda x: lookup_player_name(x['NflId'], x['DisplayName']), axis = 1)
    
    #adjust team abbreviations
    df['VisitorTeamAbbr'] = df['VisitorTeamAbbr'].apply(lambda x: team_schema[x] if x in team_schema.keys() else x)
    df['HomeTeamAbbr'] = df['HomeTeamAbbr'].apply(lambda x: team_schema[x] if x in team_schema.keys() else x)
    df['PossessionTeam'] = df['PossessionTeam'].apply(lambda x: team_schema[x] if x in team_schema.keys() else x)
    
    return(df)

adjusted_df = adjust_players_and_teams(train_2, player_schema, team_schema)

print('Standardizing player coordinates...')
def standardize_coordinates(df):
    #Is the offense moving from left to right?
    df['ToLeft'] = df['PlayDirection'] == 'left'
    
    #Flagging the ball carrier
    df['IsBallCarrier'] = df['NflId'] == df['NflIdRusher']
    
    #Assigning the possession team as home or away
    df['TeamOnOffense'] = np.where(df['PossessionTeam'] == df['HomeTeamAbbr'], 'home', 'away')
    
    #Which team is on offense?
    df['IsOnOffense'] = df['Team'] == df['TeamOnOffense']
    
    #How far away from the offensive teams own goal line?
    df['YardsFromOwnGoal'] = np.where(df['PossessionTeam'] == df['FieldPosition'], 
                                            df['YardLine'], 50 + (50 - df['YardLine']))
    
    
    df['YardsFromOwnGoal'] = np.where(df['YardLine'] == 50, 50, df['YardsFromOwnGoal'])
    
    #Standardizing X & Y Coordinates for each player
    df['X_std'] = np.where(df['ToLeft'], 120 - df['X'], df['X']) - 10
    df['Y_std'] = np.where(df['ToLeft'], 160/3 - df['Y'], df['Y'])
    
    return(df)

standardized_coord_df = standardize_coordinates(adjusted_df)

print('Standardizing player direction...')
def standardize_direction(df):
    # - 0 degrees = straight left
    # - 90 degrees = straight up the middle
    # - 180 degrees = straight right
    # - 270 degrees = straight backwards
    df['Dir_std_1'] = np.where((df['ToLeft']) & (df['Dir'] < 90), df['Dir'] + 360, df['Dir'])
    df['Dir_std_1'] = np.where((df['ToLeft']==False) & (df['Dir'] > 270), df['Dir'] - 360, df['Dir_std_1'])
    
    df['Dir_std_2'] = np.where(df['ToLeft'], df['Dir_std_1'] - 180, df['Dir_std_1'])
    
    df['X_std_end'] = (df['S']*np.cos((90 - df['Dir_std_2'])*math.pi/180) + df['X_std']).fillna(25)
    df['Y_std_end'] = (df['S']*np.cos((90 - df['Dir_std_2'])*math.pi/180) + df['Y_std']).fillna(25)
    
    df[['Dir_std_1','Dir_std_2']] = df[['Dir_std_1','Dir_std_2']].fillna(90)
    
    return(df)

standardized_dir_df = standardize_direction(standardized_coord_df)
#Checking to make sure the standardized Dir is correct
standardized_dir_df.loc[(standardized_dir_df.IsBallCarrier)].Dir_std_2.hist(bins = 50)
def get_o_line_stats(df):
    ol_df = df.loc[(df.Position.isin(['C','G','T','OG','OT','TE'])) & (df.IsOnOffense), 
           ['PlayId','A','Dir_std_1','Dir_std_2','ToLeft', 'Orientation']]
    
    ol_df['Dir_std'] = np.where(ol_df.ToLeft, ol_df['Dir_std_2'], ol_df['Dir_std_1'])
    
    ol_df['OlMidDir'] = abs(ol_df['Dir_std'] - 90)
    
    ol_df['Dir_cat'] = np.where((ol_df.Dir_std < 150) & (ol_df.Dir_std > 30), 10, 
                                np.where(ol_df.Dir_std > 150, 1,
                                        np.where((ol_df.Dir_std > 0) & (ol_df.Dir_std < 30), 1, -1 )))
    
    ol_df['OlMovingDf'] = np.where((ol_df.Dir_std < 150) & (ol_df.Dir_std > 30), 1, 0)
    ol_df['OlATimesDir'] = ol_df['A'] * ol_df['Dir_cat']
    
    ol_df_grouped = ol_df.groupby('PlayId', as_index=False)[['A','OlATimesDir','OlMovingDf', 'OlMidDir', 'Orientation']].mean()
    ol_df_grouped.rename(columns={'A':'OlAMean', 'Orientation': 'OlOrientationMean'}, inplace=True)
    
    return(ol_df_grouped)

ol_agg_df = get_o_line_stats(standardized_dir_df)
def create_features(df, deploy=False):
    def new_X(x_coordinate, play_direction):
        if play_direction == 'left':
            return(120.0 - x_coordinate)
        else:
            return(x_coordinate)

    def new_line(rush_team, field_position, yardline):
        if rush_team == field_position:
            # offense starting at X = 0 plus the 10 yard endzone plus the line of scrimmage
            return(10.0 + yardline)
        else:
            # half the field plus the yards between midfield and the line of scrimmage
            return(60.0 + (50 - yardline))

    def new_orientation(angle, play_direction):
        if play_direction == 'left':
            new_angle = 360.0 - angle
            if new_angle == 360.0:
                new_angle = 0.0
            return(new_angle)
        else:
            return(angle)

    def back_direction(orientation):
        if orientation > 180.0:
            return(1)
        else:
            return(0)
        
    def update_yardline(df):
        new_yardline = df[df['NflId'] == df['NflIdRusher']]
        new_yardline['YardLine'] = new_yardline[['PossessionTeam','FieldPosition','YardLine']].apply(lambda x: new_line(x[0],x[1],x[2]), axis=1)
        new_yardline = new_yardline[['GameId','PlayId','YardLine']]

        return(new_yardline)

    def update_orientation(df, yardline):
        df['X'] = df[['X','PlayDirection']].apply(lambda x: new_X(x[0],x[1]), axis=1)
        df['Orientation'] = df[['Orientation','PlayDirection']].apply(lambda x: new_orientation(x[0],x[1]), axis=1)
        df['Dir'] = df[['Dir','PlayDirection']].apply(lambda x: new_orientation(x[0],x[1]), axis=1)

        df = df.drop('YardLine', axis=1)
        df = pd.merge(df, yardline, on=['GameId','PlayId'], how='inner')

        return(df)

    def back_features(df):
        carriers = df[df['NflId'] == df['NflIdRusher']][['GameId','PlayId','NflIdRusher','X','Y','Orientation','Dir','YardLine']]
        carriers['back_from_scrimmage'] = carriers['YardLine'] - carriers['X']
        carriers['back_oriented_down_field'] = carriers['Orientation'].apply(lambda x: back_direction(x))
        carriers['back_moving_down_field'] = carriers['Dir'].apply(lambda x: back_direction(x))
        carriers = carriers.rename(columns={'X':'back_X',
                                            'Y':'back_Y'})
        carriers = carriers[['GameId','PlayId','NflIdRusher','back_X','back_Y','back_from_scrimmage','back_oriented_down_field','back_moving_down_field']]

        return(carriers)

    def features_relative_to_back(df, carriers):
        player_distance = df[['GameId','PlayId','NflId','X','Y']]
        player_distance = pd.merge(player_distance, carriers, on=['GameId','PlayId'], how='inner')
        player_distance = player_distance[player_distance['NflId'] != player_distance['NflIdRusher']]
        player_distance['dist_to_back'] = player_distance[['X','Y','back_X','back_Y']].apply(lambda x: euclidean_distance(x[0],x[1],x[2],x[3]), axis=1)

        player_distance = player_distance.groupby(['GameId','PlayId','back_from_scrimmage','back_oriented_down_field','back_moving_down_field'])\
                                         .agg({'dist_to_back':['min','max','mean','std']})\
                                         .reset_index()
        player_distance.columns = ['GameId','PlayId','back_from_scrimmage','back_oriented_down_field','back_moving_down_field',
                                   'min_dist','max_dist','mean_dist','std_dist']

        return(player_distance)

    def defense_features(df):
        rusher = df[df['NflId'] == df['NflIdRusher']][['GameId','PlayId','Team','X','Y']]
        rusher.columns = ['GameId','PlayId','RusherTeam','RusherX','RusherY']

        defense = pd.merge(df,rusher,on=['GameId','PlayId'],how='inner')
        defense = defense[defense['Team'] != defense['RusherTeam']][['GameId','PlayId','X','Y','RusherX','RusherY']]
        defense['def_dist_to_back'] = defense[['X','Y','RusherX','RusherY']].apply(lambda x: euclidean_distance(x[0],x[1],x[2],x[3]), axis=1)

        defense = defense.groupby(['GameId','PlayId'])\
                         .agg({'def_dist_to_back':['min','max','mean','std']})\
                         .reset_index()
        defense.columns = ['GameId','PlayId','def_min_dist','def_max_dist','def_mean_dist','def_std_dist']

        return(defense)

    def static_features(df):
        add_new_feas = []
        ## Height
        df['PlayerHeight_dense'] = df['PlayerHeight'].apply(lambda x: 12*int(x.split('-')[0])+int(x.split('-')[1]))
        
        add_new_feas.append('PlayerHeight_dense')

        ## Time
        df['TimeHandoff'] = df['TimeHandoff'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))
        df['TimeSnap'] = df['TimeSnap'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))

        df['TimeDelta'] = df.apply(lambda row: (row['TimeHandoff'] - row['TimeSnap']).total_seconds(), axis=1)
        df['PlayerBirthDate'] =df['PlayerBirthDate'].apply(lambda x: datetime.datetime.strptime(x, "%m/%d/%Y"))

        ## Age
        seconds_in_year = 60*60*24*365.25
        df['PlayerAge'] = df.apply(lambda row: (row['TimeHandoff']-row['PlayerBirthDate']).total_seconds()/seconds_in_year, axis=1)
        add_new_feas.append('PlayerAge')

        ## WindSpeed
        df['WindSpeed_ob'] = df['WindSpeed'].apply(lambda x: x.lower().replace('mph', '').strip() if not pd.isna(x) else x)
        df['WindSpeed_ob'] = df['WindSpeed_ob'].apply(lambda x: (int(x.split('-')[0])+int(x.split('-')[1]))/2 if not pd.isna(x) and '-' in x else x)
        df['WindSpeed_ob'] = df['WindSpeed_ob'].apply(lambda x: (int(x.split()[0])+int(x.split()[-1]))/2 if not pd.isna(x) and type(x)!=float and 'gusts up to' in x else x)
        df['WindSpeed_dense'] = df['WindSpeed_ob'].apply(strtofloat)
        add_new_feas.append('WindSpeed_dense')

        ## Weather
        df['GameWeather_process'] = df['GameWeather'].str.lower()
        df['GameWeather_process'] = df['GameWeather_process'].apply(lambda x: "indoor" if not pd.isna(x) and "indoor" in x else x)
        df['GameWeather_process'] = df['GameWeather_process'].apply(lambda x: x.replace('coudy', 'cloudy').replace('clouidy', 'cloudy').replace('party', 'partly') if not pd.isna(x) else x)
        df['GameWeather_process'] = df['GameWeather_process'].apply(lambda x: x.replace('clear and sunny', 'sunny and clear') if not pd.isna(x) else x)
        df['GameWeather_process'] = df['GameWeather_process'].apply(lambda x: x.replace('skies', '').replace("mostly", "").strip() if not pd.isna(x) else x)
        df['GameWeather_dense'] = df['GameWeather_process'].apply(map_weather)
        add_new_feas.append('GameWeather_dense')

        ## Orientation and Dir
        df["Orientation_ob"] = df["Orientation"].apply(lambda x : orientation_to_cat(x)).astype("object")
        df["Dir_ob"] = df["Dir"].apply(lambda x : orientation_to_cat(x)).astype("object")

        df["Orientation_sin"] = df["Orientation"].apply(lambda x : np.sin(x/360 * 2 * np.pi))
        df["Orientation_cos"] = df["Orientation"].apply(lambda x : np.cos(x/360 * 2 * np.pi))
        df["Dir_sin"] = df["Dir"].apply(lambda x : np.sin(x/360 * 2 * np.pi))
        df["Dir_cos"] = df["Dir"].apply(lambda x : np.cos(x/360 * 2 * np.pi))
        add_new_feas.append("Dir_sin")
        add_new_feas.append("Dir_cos")

        ## diff Score
        df["diffScoreBeforePlay"] = df["HomeScoreBeforePlay"] - df["VisitorScoreBeforePlay"]
        add_new_feas.append("diffScoreBeforePlay")
    
    
        static_features = df[df['NflId'] == df['NflIdRusher']][add_new_feas+['GameId','PlayId','X','Y','S','A','Dis','Orientation','Dir',
                                                            'YardLine','Quarter','Down','Distance','DefendersInTheBox','TimeDelta',]].drop_duplicates()

        static_features.fillna(-999,inplace=True)
            
        return(static_features)

    def combine_features(relative_to_back, defense, static, deploy=deploy):
        df = pd.merge(relative_to_back,defense,on=['GameId','PlayId'],how='inner')
        df = pd.merge(df,static,on=['GameId','PlayId'],how='inner')

        if not deploy:
            df = pd.merge(df, outcomes, on=['GameId','PlayId'], how='inner')

        return(df)
    
    yardline = update_yardline(df)
    df = update_orientation(df, yardline)
    back_feats = back_features(df)
    rel_back = features_relative_to_back(df, back_feats)
    def_feats = defense_features(df)
    static_feats = static_features(df)
    basetable = combine_features(rel_back, def_feats, static_feats, deploy=deploy)
    
    basetable['MaxYards'] = 110 - basetable['YardLine']
    
    return(basetable)
%time train_basetable = create_features(train, False)
pos_dict= {'DefensePersonnel': ['DL','LB','DB'],
         'OffensePersonnel': ['RB','TE','WR','QB','OL']}

def get_personnel(row, position):
    filtered_row = [pos_group for pos_group in row if position in pos_group]
    if len(filtered_row) > 0:
        return(int(filtered_row[0][0]))
    else:
        return(0)
    
def ball_carrier_features(df):
    bc_df = df.loc[df['IsBallCarrier'], ['PlayId','ToLeft','X_std','Y_std','X_std_end','Y_std_end','Dir_std_1','Dir_std_2','Position', 'A']]
    bc_df.rename(columns={'X_std': 'X_std_bc','Y_std':'Y_std_bc',
                          'X_std_end':'X_std_end_bc','Y_std_end':'Y_std_end_bc',
                          'Dir_std_1':'Dir_std_1_bc','Dir_std_2':'Dir_std_2_bc',
                          'Position':'BcPosition',
                         }, inplace=True)
    bc_df['BcRunRight'] = np.where(((bc_df.ToLeft) & (bc_df.Dir_std_2_bc > 90)) | 
                                   ((bc_df.ToLeft == False) & (bc_df.Dir_std_1_bc > 90)),True, False)
    bc_df['BcDir'] = np.where(bc_df.ToLeft, bc_df.Dir_std_2_bc, bc_df.Dir_std_1_bc)
    
    bc_df['BcFast'] = np.where((bc_df.BcPosition == 'WR') | (bc_df.BcPosition == 'CB'), 1, 0)
    bc_df['BcSlow'] = np.where((bc_df.BcPosition == 'QB') | (bc_df.BcPosition == 'FB') | 
                               (bc_df.BcPosition == 'DE') | (bc_df.BcPosition == 'DT'), 1, 0)
    
    bc_df['BcDirCat'] = np.where((bc_df.BcDir < 150) & (bc_df.BcDir > 30), 10, 
                                np.where(bc_df.BcDir > 150, 1,
                                        np.where((bc_df.BcDir > 0) & (bc_df.BcDir < 30), 1, -1 )))
    
    bc_df['BcMovingDf'] = np.where((bc_df.BcDir < 150) & (bc_df.BcDir > 30), 1, 0)
    bc_df['BcATimesDir'] = bc_df['A'] * bc_df['BcDirCat']
    
    bc_df.drop('A', axis= 1, inplace=True)
    
    return(bc_df)

def calculate_defender_loc(df, ball_carrier_df):
    def_df = df.loc[df['IsOnOffense']==False]
    
    dir_df = def_df[['PlayId','DisplayName','JerseyNumber','X_std','Y_std']].merge(ball_carrier_df, on='PlayId')
    
    
    dir_df['DefenderTowardsPlayDir'] = np.where(((dir_df['BcRunRight']) & (dir_df['Y_std'] <= dir_df['Y_std_bc'])) |
                                            ((dir_df['BcRunRight'] == False) & (dir_df['Y_std'] >= dir_df['Y_std_bc'])),
                                            1, 0)
    
    dir_agg_df = dir_df.groupby(['PlayId','X_std_bc','Y_std_bc','X_std_end_bc',
                                 'Y_std_end_bc','BcRunRight','BcDir','BcFast','BcSlow',
                                 'BcMovingDf','BcATimesDir',
                                ], as_index=False)['DefenderTowardsPlayDir'].sum()
    
    spread_x_df = dir_df.groupby('PlayId', as_index = False)['X_std'].agg({'DefenseMaxX': np.max,
                                                                           'DefenseMinX': np.min})
    
    spread_y_df = dir_df.groupby('PlayId', as_index = False)['Y_std'].agg({'DefenseMaxY': np.max,
                                                                           'DefenseMinY': np.min})
    
    spread_x_df['DefenseXSpread'] = spread_x_df.DefenseMaxX - spread_x_df.DefenseMinX
    
    spread_y_df['DefenseYSpread'] = spread_y_df.DefenseMaxY - spread_y_df.DefenseMinY
    
    dir_agg_df = dir_agg_df.merge(spread_x_df[['PlayId','DefenseXSpread']], on = 'PlayId').merge(spread_y_df[['PlayId','DefenseYSpread']], on ='PlayId')
    
    return(dir_agg_df)

print('Creating some more features...')
def more_feature_engineering(df, positions_schema):
    df['GameClockAdj'] = df['GameClock'].apply(strtoseconds)
    
    for k, v in pos_dict.items():
        df[k + 'Split'] = df[k].str.split(', ')
        for pos in v:
            df[pos + 'Count'] = df[k + 'Split'].apply(lambda row: get_personnel(row, pos))
    
    df = df.drop(list(pos_dict.keys()) + [x + 'Split' for x in pos_dict.keys()], axis = 1)
    
    df['Home'] = np.where(df['TeamOnOffense'] == 'home', 1, 0)
    
    df['Shotgun'] = np.where(df['OffenseFormation'] == 'SHOTGUN', 1, 0)
    
    #Which team name is on defense?
    df['TeamAbbrOnDefense'] = np.where(df['TeamOnOffense'] == 'home', df['VisitorTeamAbbr'], df['HomeTeamAbbr'])
    
    #Weather
    df['GameWeatherPro'] = df['GameWeather'].str.lower().apply(lambda x: "indoor" if not pd.isna(x) and "indoor" in x else x)
    df['GameWeatherPro'] = df['GameWeatherPro'].apply(lambda x: x.replace('coudy', 'cloudy').replace('clouidy', 'cloudy').replace('party', 'partly') if not pd.isna(x) else x)
    df['GameWeatherPro'] = df['GameWeatherPro'].apply(lambda x: x.replace('clear and sunny', 'sunny and clear') if not pd.isna(x) else x)
    df['GameWeatherPro'] = df['GameWeatherPro'].apply(lambda x: x.replace('skies', '').replace("mostly", "").strip() if not pd.isna(x) else x)
    df['GameWeatherPro'] = df['GameWeatherPro'].apply(map_weather)
    
    #Diff Score
    df["DiffScoreBeforePlay"] = df["HomeScoreBeforePlay"] - df["VisitorScoreBeforePlay"]
    df["DiffScoreBeforePlayBinary"] = (df["HomeScoreBeforePlay"] > df["VisitorScoreBeforePlay"]).astype("object")
    
    df = df.drop(['GameWeather','Season','TimeHandoff','TimeSnap','PlayerBirthDate','Orientation','Dir','WindSpeed','GameClock'], axis =1)
    
    #Ball Carrier Features
    bc_df = ball_carrier_features(df)
    
    #Defender Locations
    merged_bc_def_df = calculate_defender_loc(df, bc_df)
    
    #Merge DataFrames
    final_fe_df = df.merge(merged_bc_def_df, on ='PlayId')
    
    #Calculate Distance between each Defender & the Ball Carrier
    final_fe_df['DistFromBc'] = np.sqrt((final_fe_df.X_std-final_fe_df.X_std_bc)**2 
                                           + (final_fe_df.Y_std-final_fe_df.Y_std_bc)**2)
    
    final_fe_df['S'] = np.where(final_fe_df['S'] == 0, .1, final_fe_df['S'])
    
    final_fe_df['TimeToBc'] = final_fe_df['DistFromBc'] / final_fe_df['S']
    
    #Calculate Distance between each Defender & the Ball Carrier
    final_fe_df['DistFromBcEnd'] = np.sqrt((final_fe_df.X_std_end-final_fe_df.X_std_end_bc)**2 
                                           + (final_fe_df.Y_std_end-final_fe_df.Y_std_end_bc)**2)
    
    final_fe_df['TimeToBcEnd'] = final_fe_df['DistFromBcEnd'] / final_fe_df['S']
    
    #Ball Carrier Direction minus Direction of Defender
    final_fe_df['Dir_std'] = np.where(final_fe_df.ToLeft, final_fe_df.Dir_std_2, final_fe_df.Dir_std_1)
    #final_fe_df['BcDirDefDir'] = final_fe_df.BcDir - final_fe_df.Dir_std
    
    final_fe_df['DefendersInTheBox'] = final_fe_df['DefendersInTheBox'].fillna(int(final_fe_df.DefendersInTheBox.median()))
    final_fe_df['DefendersInTheBox_vs_Distance'] = final_fe_df['DefendersInTheBox'] / final_fe_df['Distance']
    
    final_fe_df['InsideRun'] = np.where((final_fe_df['BcDir'] > 65) & (final_fe_df['BcDir'] < 115), 1, 0)
    final_fe_df['OffTackle'] = np.where(((final_fe_df['BcDir'] <= 65) & (final_fe_df['BcDir'] > 30)) | \
                                         ((final_fe_df['BcDir'] >= 115) & (final_fe_df['BcDir'] < 150)), 1, 0)
    
    return(final_fe_df)

fe_df = more_feature_engineering(standardized_dir_df, pos_dict)
#Select the play specific features to use
fe_feature_lst = ['GameId','PlayId','GameClockAdj','Shotgun','DefenseXSpread','DefenseYSpread',
                  'X_std_bc','Y_std_bc','BcDir','InsideRun','OffTackle','TeamOnOffense','YardsFromOwnGoal',
                  'BcFast','BcSlow','BcMovingDf','BcATimesDir',#'DBCount','DefendersInTheBox_vs_Distance',
                 ]

fe_feature_df = fe_df[fe_feature_lst].drop_duplicates()
def get_bc_voronoi_area(df, point_type='current'):
    df = df.sort_values(['IsOnOffense','PlayId','IsBallCarrier','Y_std'])
    bc_v_area_lst = []
    
    if point_type == 'current':
        X_col, Y_col = 'X_std', 'Y_std'
    elif point_type == 'end':
        X_col, Y_col = 'X_std_end', 'Y_std_end'
    else:
        print('Invalid point type')
    
    for play_id in df.PlayId.unique():
        try:
            v_fe_df = df.loc[df.PlayId == play_id]
            v_fe_df = v_fe_df.loc[(v_fe_df.IsOnOffense == False) | (v_fe_df.IsBallCarrier)]

            points = np.c_[v_fe_df[X_col], v_fe_df[Y_col]]

            vor = Voronoi(points)

            vertices = vor.vertices
            regions = vor.regions
            point_regions = vor.point_region

            v_perm = [vertices[r] for r in regions[point_regions[-1]]]

            rusher_x = v_fe_df.loc[v_fe_df.IsBallCarrier, X_col].values[0]
            rusher_y = v_fe_df.loc[v_fe_df.IsBallCarrier, Y_col].values[0]

            vor_df = pd.DataFrame(np.vstack(v_perm), columns = [X_col, Y_col])
            vor_df[X_col] = np.where(vor_df[X_col] < rusher_x, rusher_x,
                                  np.where(vor_df[X_col] > rusher_x + 20, rusher_x + 20, vor_df[X_col]))

            vor_df[Y_col] = np.where(vor_df[Y_col] < (rusher_y / 2), rusher_y / 2,
                                  np.where(vor_df[Y_col] > rusher_y * 2, rusher_y * 2, vor_df[Y_col]))
            
            vor_area = ConvexHull(vor_df, qhull_options = 'QJ').area
            
            if vor_area > 1000:
                print(play_id)
            bc_v_area_lst.append(vor_area)

        except:
            print('Voronoi Error for play {}'.format(play_id))
            bc_v_area_lst.append(42)
            
    return(bc_v_area_lst)
%time bc_v_area = get_bc_voronoi_area(fe_df, 'current')
bc_v_area_end = get_bc_voronoi_area(fe_df, 'end')
fe_feature_df['bc_v_area'] = bc_v_area
fe_feature_df['bc_v_area_end'] = bc_v_area_end
def standardize_dataset(train):
    train['ToLeft'] = train.PlayDirection == "left"
    train['IsBallCarrier'] = train.NflId == train.NflIdRusher
    train['TeamOnOffense'] = "home"
    train.loc[train.PossessionTeam != train.HomeTeamAbbr, 'TeamOnOffense'] = "away"
    train['IsOnOffense'] = train.Team == train.TeamOnOffense # Is player on offense?
    train['YardLine_std'] = 100 - train.YardLine
    train.loc[train.FieldPosition.fillna('') == train.PossessionTeam,  
            'YardLine_std'
             ] = train.loc[train.FieldPosition.fillna('') == train.PossessionTeam,  
              'YardLine']
    train['X_std'] = train.X
    train.loc[train.ToLeft, 'X_std'] = 120 - train.loc[train.ToLeft, 'X'] 
    train['Y_std'] = train.Y
    train.loc[train.ToLeft, 'Y_std'] = 53.3 - train.loc[train.ToLeft, 'Y'] 
    train['Orientation_std'] = train.Orientation
    train.loc[train.ToLeft, 'Orientation_std'] = np.mod(180 + train.loc[train.ToLeft, 'Orientation_std'], 360)
    train['Dir_std'] = train.Dir
    train.loc[train.ToLeft, 'Dir_std'] = np.mod(180 + train.loc[train.ToLeft, 'Dir_std'], 360)
    train.loc[train['Season'] == 2017, 'Orientation'] = np.mod(90 + train.loc[train['Season'] == 2017, 'Orientation'], 360)    
    
    return train

dominance_df = standardize_dataset(train_3)
def radius_calc(dist_to_ball):
    ''' I know this function is a bit awkward but there is not the exact formula in the paper,
    so I try to find something polynomial resembling
    Please consider this function as a parameter rather than fixed
    I'm sure experts in NFL could find a way better curve for this'''
    return 4 + 6 * (dist_to_ball >= 15) + (dist_to_ball ** 3) / 560 * (dist_to_ball < 15)

@np.vectorize
def pitch_control(x_point, y_point):
    '''Compute the pitch control over a coordinate (x, y)'''

    offense_ids = my_play[my_play['IsOnOffense']].index
    offense_control = compute_influence(x_point, y_point, offense_ids)
    offense_score = np.sum(offense_control)

    defense_ids = my_play[~my_play['IsOnOffense']].index
    defense_control = compute_influence(x_point, y_point, defense_ids)
    defense_score = np.sum(defense_control)

    return expit(offense_score - defense_score)

class Controller:
    '''This class is a wrapper for the two functions written above'''
    def __init__(self, play):
        self.play = play
        self.vec_influence = np.vectorize(self.compute_influence)
        self.vec_control = np.vectorize(self.pitch_control) 
        
    def compute_influence(self, x_point, y_point, player_id):
        '''Compute the influence of a certain player over a coordinate (x, y) of the pitch
        '''
        point = np.array([x_point, y_point])
        player_row = self.play.loc[player_id]
        theta = math.radians(player_row[56])
        speed = player_row[5]
        player_coords = player_row[54:56].values
        ball_coords = self.play[self.play['IsBallCarrier']].iloc[:, 54:56].values

        dist_to_ball = euclidean(player_coords, ball_coords)

        S_ratio = (speed / 13) ** 2         # we set max_speed to 13 m/s
        RADIUS = radius_calc(dist_to_ball)  # updated

        S_matrix = np.matrix([[RADIUS * (1 + S_ratio), 0], [0, RADIUS * (1 - S_ratio)]])
        R_matrix = np.matrix([[np.cos(theta), - np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        COV_matrix = np.dot(np.dot(np.dot(R_matrix, S_matrix), S_matrix), np.linalg.inv(R_matrix))

        norm_fact = (1 / 2 * np.pi) * (1 / np.sqrt(np.linalg.det(COV_matrix)))    
        mu_play = player_coords + speed * np.array([np.cos(theta), np.sin(theta)]) / 2

        intermed_scalar_player = np.dot(np.dot((player_coords - mu_play),
                                        np.linalg.inv(COV_matrix)),
                                 np.transpose((player_coords - mu_play)))
        player_influence = norm_fact * np.exp(- 0.5 * intermed_scalar_player[0, 0])

        intermed_scalar_point = np.dot(np.dot((point - mu_play), 
                                        np.linalg.inv(COV_matrix)), 
                                 np.transpose((point - mu_play)))
        point_influence = norm_fact * np.exp(- 0.5 * intermed_scalar_point[0, 0])

        return point_influence / player_influence
    
    
    def pitch_control(self, x_point, y_point):
        '''Compute the pitch control over a coordinate (x, y)'''

        offense_ids = self.play[self.play['IsOnOffense']].index
        offense_control = self.vec_influence(x_point, y_point, offense_ids)
        offense_score = np.sum(offense_control)

        defense_ids = self.play[~self.play['IsOnOffense']].index
        defense_control = self.vec_influence(x_point, y_point, defense_ids)
        defense_score = np.sum(defense_control)

        return expit(offense_score - defense_score)
    
    def display_control(self, grid_size=(30, 15), figsize=(11, 7)):
        front, behind = 15, 5
        left, right = 20, 20

        colorm = ['purple'] * 11 + ['orange'] * 11
        colorm[np.where(self.play.Rusher.values)[0][0]] = 'black'
        player_coords = self.play[self.play['Rusher']][['X_std', 'Y_std']].values[0]

        X, Y = np.meshgrid(np.linspace(player_coords[0] - behind, 
                                       player_coords[0] + front, 
                                       grid_size[0]), 
                           np.linspace(player_coords[1] - left, 
                                       player_coords[1] + right, 
                                       grid_size[1]))

        # infl is an array of shape num_points with values in [0,1] accounting for the pitch control
        infl = self.vec_control(X, Y)

        plt.figure(figsize=figsize)
        plt.contourf(X, Y, infl, 12, cmap='bwr')
        plt.scatter(self.play['X'].values, self.play['Y'].values, c=colorm)
        plt.title('Yards gained = {}, play_id = {}'.format(self.play['Yards'].values[0], 
                                                           self.play['PlayId'].unique()[0]))
        plt.show()

control_lst = []
for play_id in tqdm(dominance_df.PlayId.unique()):
    my_play = dominance_df[dominance_df['PlayId']==play_id]
    control = Controller(my_play)
    #Bc influence
    bc_coords = list(*my_play.loc[(my_play.IsBallCarrier), ['X_std','Y_std']].values)
    pitch_control = control.vec_control(*bc_coords)
    control_lst.append(pitch_control)
train_basetable['BcInfl'] = [np.asscalar(i) for i in control_lst]
def centeroidnp(arr):
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return(sum_x/length, sum_y/length)

def get_defense_centroid(df):
    centroid_lst = []
    defense_df = df.loc[df.IsOnOffense == False]
    for play_id in defense_df.PlayId.unique():
        def_points = defense_df.loc[defense_df.PlayId == play_id, ['X_std','Y_std']]
        cp = centeroidnp(np.array(def_points))
        centroid_lst.append(cp)
        
    return(centroid_lst)

%time train_basetable['def_centroid'] = get_defense_centroid(fe_df)
train_basetable['def_centroid_X_std'] = train_basetable.def_centroid.apply(lambda x: x[0])
train_basetable['def_centroid_Y_std'] = train_basetable.def_centroid.apply(lambda x: x[1])
def get_qb_location(df):
    qb_fe_df = df.loc[(df.Position == 'QB')].sort_values(['PlayId','S'])
    #Since some plays have more than 1 QB, I rank them by Speed and choose the slowest
    qb_fe_df['QbRank'] = qb_fe_df.groupby(by=['PlayId'])['S'].transform(lambda x: x.rank())
    qb_fe_df = qb_fe_df.loc[qb_fe_df['QbRank'] == 1, ['PlayId', 'X_std', 'Y_std']]
    qb_fe_df.rename(columns={'X_std':'X_std_qb', 'Y_std': 'Y_std_qb'}, inplace=True)
    return(qb_fe_df)

qb_location_df = get_qb_location(fe_df)
def get_defense_end(df):
    def_df = df.loc[df.IsOnOffense==False]
    #Def yards made up to Bc
    def_df['DefYardsTowardsBc'] =  def_df['DistFromBc'] - def_df['DistFromBcEnd']
    
    def_dist_end_df = def_df.groupby('PlayId', as_index=False)['DistFromBcEnd'] \
        .agg({'MinDistFromBcEnd':np.min,'MeanDistFromBcEnd':np.mean,'MedDistFromBcEnd':np.median,'StdDistFromBcEnd':np.std})
    
    def_toward_bc_df = def_df.groupby('PlayId', as_index = False)['DefYardsTowardsBc'] \
        .agg({'SumDefYardsTowardsBc':sum, 'StdDefYardsTowardsBc':np.std, 'MaxDefYardsTowardsBc': np.max})
    
    return(def_dist_end_df.merge(def_toward_bc_df, on = 'PlayId'))

def_dist_end_df = get_defense_end(fe_df)

#def_dir_df = fe_df.loc[fe_df.IsOnOffense==False].groupby('PlayId', as_index=False)['BcDirDefDir'].agg({'StdBcDirDefDir':np.std})
def_time_to_spot_df = fe_df.loc[fe_df.IsOnOffense == False] \
    .groupby('PlayId', as_index=False)[['TimeToBc','TimeToBcEnd']] \
    .agg([np.min, np.mean, np.median, np.std])

def_time_to_spot_df.columns = def_time_to_spot_df.columns.droplevel(0)

def_time_to_spot_df.columns = ['TimeToBc' + c for c in def_time_to_spot_df.columns[0:4]] \
    + ['TimeToBcEnd' + c for c in def_time_to_spot_df.columns[4:]]

def_time_to_spot_df.reset_index(inplace=True)
# fe_defenders = fe_df.loc[fe_df.IsOnOffense == False]
# fe_defenders['DefTimeToBcRank'] = fe_defenders.groupby('PlayId')['TimeToBc'].transform(lambda x: x.rank())
# three_def_df = fe_defenders.loc[fe_defenders.DefTimeToBcRank <= 3]
# grouped_closest_time_df = three_def_df.groupby('PlayId', as_index=False)['DistFromBc'].mean()
# grouped_closest_time_df.rename(columns={'TimeToBc':'Top3DefMeanTime'}, inplace=True)
fe_defenders = fe_df.loc[fe_df.IsOnOffense == False]
fe_defenders['DistFromBcRank'] = fe_defenders.groupby('PlayId')['DistFromBc'].transform(lambda x: x.rank())
three_def_df = fe_defenders.loc[fe_defenders.DistFromBcRank <= 3]
grouped_closest_dist_df = three_def_df.groupby('PlayId', as_index=False)['DistFromBc'].mean()
grouped_closest_dist_df.rename(columns={'DistFromBc':'Top3DistFromBcMean'}, inplace=True)
merged_basetable = train_basetable.merge(def_dist_end_df, on='PlayId') \
    .merge(ol_agg_df, on = 'PlayId') \
    .merge(fe_feature_df, on=['GameId','PlayId']) \
    .merge(qb_location_df, how = 'left', on = 'PlayId') \
    .merge(def_time_to_spot_df, how = 'left', on = 'PlayId') \
    .merge(grouped_closest_dist_df, how = 'left', on = 'PlayId')

merged_basetable['Shotgun_Down_Distance'] = merged_basetable['Shotgun'] * merged_basetable['Down'] * merged_basetable['Distance']

merged_basetable['BcDistFromDefCent'] = np.sqrt((merged_basetable.X_std_bc - merged_basetable.def_centroid_X_std)**2 
                                           + (merged_basetable.Y_std_bc - merged_basetable.def_centroid_Y_std)**2)

merged_basetable['BcDistFromQb'] = np.sqrt((merged_basetable.X_std_bc - merged_basetable.X_std_qb)**2 
                                           + (merged_basetable.Y_std_bc - merged_basetable.Y_std_qb)**2)

merged_basetable['BcDistFromQb'] = merged_basetable['BcDistFromQb'].fillna(merged_basetable.BcDistFromQb.mean())

merged_basetable['BcDirFromMid'] = abs(merged_basetable['BcDir'] - 90)
# down_dummies = pd.get_dummies(merged_basetable.Down, drop_first=True)
# down_dummies.columns = ['Down' + str(d).upper() for d in down_dummies.columns]

# quarter_dummies = pd.get_dummies(merged_basetable.Quarter, drop_first=True)
# quarter_dummies.columns = ['Quarter' + str(q).upper() for q in quarter_dummies.columns]

# merged_basetable = pd.concat([merged_basetable,quarter_dummies, down_dummies], axis = 1)
merged_basetable['Quarter4'] = np.where(merged_basetable.Quarter == 4, 1, 0)
merged_basetable['Down4'] = np.where(merged_basetable.Down == 4, 1, 0)
cols_to_drop = ['def_centroid','X','Y','X_std_qb','Y_std_qb','def_centroid_X_std','def_centroid_Y_std',
                'Dir','TeamOnOffense','Quarter','Down','YardsFromOwnGoal',
               ]

merged_basetable.drop(cols_to_drop, axis = 1, inplace=True)
merged_basetable.corr()['Yards']
def cleanabv(train):
    #   Clean Abbreviations
    train['ToLeft'] = train.PlayDirection == "left"
    train['IsBallCarrier'] = train.NflId == train.NflIdRusher
    train.loc[train.VisitorTeamAbbr == "ARI", 'VisitorTeamAbbr'] = "ARZ"
    train.loc[train.HomeTeamAbbr == "ARI", 'HomeTeamAbbr'] = "ARZ"
    train.loc[train.VisitorTeamAbbr == "BAL", 'VisitorTeamAbbr'] = "BLT"
    train.loc[train.HomeTeamAbbr == "BAL", 'HomeTeamAbbr'] = "BLT"
    train.loc[train.VisitorTeamAbbr == "CLE", 'VisitorTeamAbbr'] = "CLV"
    train.loc[train.HomeTeamAbbr == "CLE", 'HomeTeamAbbr'] = "CLV"
    train.loc[train.VisitorTeamAbbr == "HOU", 'VisitorTeamAbbr'] = "HST"
    train.loc[train.HomeTeamAbbr == "HOU", 'HomeTeamAbbr'] = "HST"
    #train['Dir_std'] = np.mod(90 - train.Dir, 360)
    train['Dir_std_1'] = np.where((train['ToLeft']) & (train['Dir'] < 90), train['Dir'] + 360, train['Dir'])
    train['Dir_std_1'] = np.where((train['ToLeft']==False) & (train['Dir'] > 270), train['Dir'] - 360, train['Dir_std_1'])
    train['Dir_std_2'] = np.where(train['ToLeft'], train['Dir_std_1'] - 180, train['Dir_std_1'])
    train['Dir_std'] = np.where(train.ToLeft, train.Dir_std_2, train.Dir_std_1)
    train.loc[train.DisplayName == "Bradley Sowell", "Position"] = "T"
    return train

def getolinesix(play):
    oline = play.loc[(play.loc[:,"Position"]=="T") | (play.loc[:,"Position"]=="G") | (play.loc[:,"Position"]=="C") | (play.loc[:,"Position"]=="OT") | (play.loc[:,"Position"]=="OG"),:]
    oline = oline.sort_values(by = 'Y')
    oline = oline.reset_index()
    firstoline = oline.loc[0,"level_0"]
    lastoline = oline.loc[4,"level_0"]
    onetwo = 0
    if firstoline>10:
        offense = play[11:22]
        onetwo = 2
    else:
        offense = play[0:11]
        onetwo = 1
    offense = offense.sort_values(by = 'Y')
    offense = offense.reset_index()
    ox1 = offense.loc[offense.loc[:,"level_0"]==firstoline,:]
    ox1 = ox1.index
    ox5 = offense.loc[offense.loc[:,"level_0"]==lastoline,:]
    ox5 = ox5.index
    
    if len(ox1) == 0:
        ox1 = pd.Series([1])
    
    if len(ox5) == 0:
        ox5 = pd.Series([10])
    
    lh = 0
    li = -1
    rh = 0
    ri = -1
    
    for i in range(ox1[0]-1,-1,-1):
        tempx = offense.loc[i,"Y"]
        tempy = offense.loc[i,"X"]
        oxx = offense.loc[ox1,"Y"]
        oxy = offense.loc[ox1,"X"]
        tempd = distance(tempx,oxx,tempy,oxy)
        if tempd <= 4:
            if offense.loc[i,"Position"]=="TE":
                lh = 1
                #li = offense.loc[i,"index"]
                li = i
                break
        else:
            lh = 0
            li = -1
    
    for i in range(ox5[0]+1,11,1):
        tempx = offense.loc[i,"Y"]
        tempy = offense.loc[i,"X"]
        oxx = offense.loc[ox5,"Y"]
        oxy = offense.loc[ox5,"X"]
        tempd = distance(tempx,oxx,tempy,oxy)
        if tempd <= 4:
            if offense.loc[i,"Position"]=="TE":
                rh = 1
                #print(offense)
                #ri = offense.loc[i,"index"]
                #print(i)
                ri = i
                break
        else:
            rh = 0
            ri = -1
        
    play.reset_index()
    play.sort_values(by = 'Y')

    return oline,ri,rh,li,lh,offense

def encodepos(play,uniqueteams):
    possession = np.zeros(22)
    for i in range(0,22):
        for j in range(0,32):
            if play.loc[i,"PossessionTeam"]==uniqueteams[j]:
                possession[i]=j
    
    return possession

def encodefpos(play,uniqueteams):
    fposition = np.zeros(22)
    for i in range(0,22):
        for j in range(0,32):
            if play.loc[i,"FieldPosition"]==uniqueteams[j]:
                fposition[i]=j
    
    return fposition

def changedirection(play,pos,fpos):
    #ha = home/away
    ha = play.loc[play.loc[:,"NflId"]==play.loc[0,"NflIdRusher"],"Team"]
    bc = np.where(play.loc[:,"NflId"]==play.loc[0,"NflIdRusher"])[0]
    bcy = play.loc[bc,"X"]
    bcy = bcy.reset_index()
    if bc > 10:
        meandy = stats.mean(play.loc[11:,'X'])
    else:
        meandy = stats.mean(play.loc[0:10,'X'])
    
    postemp = pos[0]
    fpostemp = fpos[0]
    
    if bcy.loc[0,"X"]>meandy:
        play.loc[:,"X"] = 100 - play.loc[:,"X"] + 20
    else:
        play.loc[:,"X"] = play.loc[:,"X"]
        play.loc[:,"Y"] = 53.3 - play.loc[:,"Y"]
    
    if postemp != fpostemp:
        play.loc[:,"YardLine"] = 100 - play.loc[0,"YardLine"]
    
    ydline =  play.loc[0,"YardLine"]
    
    return play, ha, ydline, bc
    
def distance(x1,x2,y1,y2):
    dist = math.sqrt((x1-x2)**2+(y1-y2)**2)
    return dist

def linear(x1,x2,y1,y2):
    slope = (y1-y2)/(x1-x2)
    temp = slope*x1
    b = y1-temp
    return slope, b

def hmengagedextra(play,ha,ydline,bc):
    oline,ri,rh,li,lh,offense = getolinesix(play)
    #print(play)
    play = play.reset_index()
    
    if rh == 1:
        rol = offense.loc[[ri]]
        oline = pd.concat([oline,rol])
        
    else:
        temp = oline.loc[[4]]
        temp.loc[4,"Y"]=oline.loc[4,"Y"]+2
        oline = pd.concat([oline,temp])
    
    if lh == 1:
        lol = offense.loc[[li]]
        oline = pd.concat([lol,oline])
    else:
        temp = oline.loc[[0]]
        temp.loc[0,"Y"] = temp.loc[0,"Y"]-2
        oline = pd.concat([temp,oline])
    olx = oline.loc[:,"Y"]
    olx = olx.reset_index()
    oly = oline.loc[:,"X"]
    oly = oly.reset_index()
    #print(oline)
    
    bcy = play.loc[bc,"X"]
    bcy = bcy.reset_index()
    if ha.all() == "away":
        defense = play.loc[11:21,:]
    else:
        defense = play.loc[0:10,:]
        
    defense = defense.drop(columns="level_0")
    defense = defense.reset_index()
    olnum = np.zeros(6)
    olnumeng = np.zeros(6)
    numdefbf = np.zeros(6)
    for i in range(0,11):
        dx = defense.loc[i,"Y"]
        dy = defense.loc[i,"X"]
        for j in range(0,6):
            olx1 = olx.loc[j,"Y"]
            olx2 = olx.loc[j+1,"Y"]
            oly1 = oly.loc[j,"X"]
            oly2 = oly.loc[j+1,"X"]
            if (dx > olx1) & (dx <= olx2):
                dist1 = distance(olx1,dx,oly1,dy)
                dist2 = distance(olx2,dx,oly2,dy)
                if (dy < ydline+15) & (dy > bcy.loc[0,"X"]):
                    slope,b = linear(olx1,olx2,oly1,oly2)
                    tempy = dx*slope+b
                    if dy<=tempy:
                        numdefbf[j] = numdefbf[j] + 1
                    if dist1 <=1.5 or dist2 <=1.5:
                        olnumeng[j] = olnumeng[j]+1
                    olnum[j] = olnum[j]+1
                    
    return olnum,olnumeng,numdefbf,oline

train = pd.read_csv("../input/nfl-big-data-bowl-2020/train.csv", low_memory=False)
#train = train.loc[train.Season != 2017]
train = cleanabv(train)
uniqueplays = np.unique(train.loc[:,"PlayId"])
uniqueteams = np.unique(train.loc[:,"PossessionTeam"])
holes_spread = []
for i in tqdm(range(0, len(uniqueplays))):
    play = train.loc[train.loc[:,"PlayId"]==uniqueplays[i],:]
    play = play.reset_index()
    playid = play.PlayId[0]
    pos = encodepos(play,uniqueteams)
    fpos = encodefpos(play,uniqueteams)
    play,ha,ydline,bc = changedirection(play,pos,fpos)
    olnum,olnumeng,numdefbf,oline = hmengagedextra(play,ha,ydline,bc)
    bc_dir = play.loc[play.IsBallCarrier, 'Dir_std'].values[0]
    oline_y = oline.Y
    holes = np.diff(oline_y)
    holes_spread.append(np.mean(holes))
    
    if i == 0:
        olnums = olnum
        olnumseng = olnumeng
        numdefsbf = numdefbf
    else:
        olnums = np.vstack((olnums,olnum))
        olnumseng = np.vstack((olnumseng,olnumeng))
        numdefsbf = np.vstack((numdefsbf,numdefbf))
los_df = outcomes.copy().drop('Yards', axis = 1)
los_df['olnums'] = [sum(i) for i in olnums]
los_df['olnumseng'] = [sum(i) for i in olnumseng]
los_df['numdefsbf'] = [sum(i) for i in numdefsbf]
los_df['avgholesize'] = holes_spread
merged_basetable = merged_basetable.merge(los_df, on=['GameId', 'PlayId'])
merged_basetable['numdefsb_cube'] = merged_basetable['numdefsbf'] ** 3
merged_basetable['numdefsbf_cube_div_min_dist'] =  merged_basetable['numdefsb_cube'] / merged_basetable['def_min_dist']
merged_basetable['BcInfl'] = merged_basetable['BcInfl'].fillna(.8)
merged_basetable['v_infl'] = (merged_basetable['BcInfl'] * merged_basetable['bc_v_area_end']).fillna(24)
lower_bound = merged_basetable.Yards.quantile(0.002)
upper_bound = merged_basetable.Yards.quantile(0.998)
merged_basetable = merged_basetable.loc[(merged_basetable.Yards > lower_bound) & (merged_basetable.Yards < upper_bound)]
cluster_features = ['back_from_scrimmage','S','A','Shotgun','BcDistFromQb','InsideRun','OffTackle','TimeDelta','BcDirFromMid',]
cluster_df = merged_basetable[cluster_features]
def scale_data(feature_data):
    mms = MinMaxScaler()
    mms.fit(feature_data)
    data_transformed = mms.transform(feature_data)
    return(data_transformed)

def plot_elbow(transformed_data, k_max):
    sum_of_squared_distances = []
    K = range(1, k_max)
    for k in K:
        km = KMeans(n_clusters=k, init = 'k-means++', random_state = 42)
        km = km.fit(transformed_data)
        sum_of_squared_distances.append(km.inertia_)
        
    plt.plot(K, sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()
    
def fit_clusters(feature_df, k):
    kmeans = KMeans(n_clusters = k, init = 'k-means++', random_state = 42, max_iter=100, n_init=1)
    fit = kmeans.fit(feature_df)
    return(fit)
data_transformed = scale_data(cluster_df)
plot_elbow(data_transformed, 15)
cluster_fit = fit_clusters(data_transformed, 5)
cluster_labels = cluster_fit.predict(data_transformed)
merged_basetable['RunCluster'] = cluster_labels
X = merged_basetable.copy()
yards = X.Yards

y = np.zeros((yards.shape[0], 199))
for idx, target in enumerate(list(yards)):
    y[idx][99 + target] = 1

X.drop(['GameId','PlayId','Yards'], axis=1, inplace=True)

#Scale the numeric features
scaler = StandardScaler()

binary_features = ['back_oriented_down_field', 'back_moving_down_field', 'Shotgun', 'InsideRun', 'OffTackle', 
                   'BcFast','BcSlow','BcMovingDf','Quarter4', 'Down4','RunCluster',
#                   'Quarter2', 'Quarter3', 'Quarter5', 'Down2', 'Down3', 
                  ]

binary_array = np.array(X.loc[:, binary_features])

numeric_df = X.drop(binary_features, axis = 1)
numeric_scaled = scaler.fit_transform(numeric_df)
X_transformed = np.hstack((numeric_scaled, binary_array))

X_train, X_val, y_train, y_val = train_test_split(X_transformed, y, test_size=0.25, random_state=12345)

print(X_train.shape, X_val.shape)
print(y_train.shape, y_val.shape)
def get_rf(x_tr, y_tr, x_val, y_val, shape):
    model = RandomForestRegressor(bootstrap=False, max_features=0.3, min_samples_leaf=15, 
                                  min_samples_split=8, n_estimators=25, n_jobs=-1, random_state=42)
    model.fit(x_tr, y_tr)
    
    y_pred = model.predict(x_val)
    y_valid = y_val
    crps = crps_score(y_pred, y_valid, shape=shape)
    
    return model, crps, y_pred

# Calculate CRPS score
def crps_score(y_prediction, y_valid, shape=X.shape[0]):
    y_true = np.clip(np.cumsum(y_valid, axis=1), 0, 1)
    y_pred = np.clip(np.cumsum(y_prediction, axis=1), 0, 1)
    val_s = ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * shape)
    crps = np.round(val_s, 6)
    return crps

from sklearn.ensemble import RandomForestRegressor
#Train RF
#rf, crps_rf = get_rf(tr_x, tr_y, val_x, val_y, shape=val_x.shape[0])
rf, crps_rf, y_pred = get_rf(X_train, y_train, X_val, y_val, shape=X_val.shape[0])
#models_rf.append(rf)
print("the crps (RF) is %f"%(crps_rf))
pd.DataFrame(list(zip(X.columns, rf.feature_importances_)),
             columns = ['Feature', 'importances']).sort_values('importances', ascending = False)
from keras.layers import Dense,Input,Flatten,concatenate,Dropout,Lambda
from keras.models import Model
import keras.backend as K
import re
from keras.losses import binary_crossentropy
from  keras.callbacks import EarlyStopping,ModelCheckpoint
import codecs

from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from sklearn.metrics import f1_score

class CRPSCallback(Callback):
    
    def __init__(self,validation, predict_batch_size=20, include_on_batch=False):
        super(CRPSCallback, self).__init__()
        self.validation = validation
        self.predict_batch_size = predict_batch_size
        self.include_on_batch = include_on_batch
        
        print('validation shape',len(self.validation))

    def on_batch_begin(self, batch, logs={}):
        pass

    def on_train_begin(self, logs={}):
        if not ('CRPS_score_val' in self.params['metrics']):
            self.params['metrics'].append('CRPS_score_val')

    def on_batch_end(self, batch, logs={}):
        if (self.include_on_batch):
            logs['CRPS_score_val'] = float('-inf')

    def on_epoch_end(self, epoch, logs={}):
        logs['CRPS_score_val'] = float('-inf')
            
        if (self.validation):
            X_valid, y_valid = self.validation[0], self.validation[1]
            y_pred = self.model.predict(X_valid)
            y_true = np.clip(np.cumsum(y_valid, axis=1), 0, 1)
            y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1)
            val_s = ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * X_valid.shape[0])
            val_s = np.round(val_s, 6)
            logs['CRPS_score_val'] = val_s
def get_model(x_tr,y_tr,x_val,y_val):
    inp = Input(shape = (x_tr.shape[1],))
    x = Dense(1024, input_dim=X.shape[1], activation='relu')(inp)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
#     x = Dense(128, activation='relu')(x)
#     x = Dropout(0.5)(x)
#     x = BatchNormalization()(x)
    
    out = Dense(199, activation='softmax')(x)
    model = Model(inp,out)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[])
    #add lookahead
#     lookahead = Lookahead(k=5, alpha=0.5) # Initialize Lookahead
#     lookahead.inject(model) # add into model

    
    es = EarlyStopping(monitor='CRPS_score_val', 
                       mode='min',
                       restore_best_weights=True, 
                       verbose=1, 
                       patience=15)

    mc = ModelCheckpoint('best_model.h5',monitor='CRPS_score_val',mode='min',
                                   save_best_only=True, verbose=1, save_weights_only=True)
    
    bsz = 1024
    steps = x_tr.shape[0]/bsz
    
    model.fit(x_tr, y_tr,callbacks=[CRPSCallback(validation = (x_val,y_val)),es,mc], epochs=20, batch_size=bsz,verbose=0)
    model.load_weights("best_model.h5")
    
    y_pred = model.predict(x_val)
    y_valid = y_val
    y_true = np.clip(np.cumsum(y_valid, axis=1), 0, 1)
    y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1)
    val_s = ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * x_val.shape[0])
    crps = np.round(val_s, 6)

    return model,crps, y_true, y_pred
from sklearn.model_selection import train_test_split, KFold
import time

losses = []
models = []
crps_csv = []

s_time = time.time()

for k in range(1):
    kfold = KFold(5, random_state = 42 + k, shuffle = True)
    for k_fold, (tr_inds, val_inds) in enumerate(kfold.split(yards)):
        print("-----------")
        print("-----------")
        tr_x,tr_y = X_transformed[tr_inds],y[tr_inds]
        val_x,val_y = X_transformed[val_inds],y[val_inds]
        model,crps,y_true,y_preds = get_model(tr_x,tr_y,val_x,val_y)
        models.append(model)
        print("the %d fold crps is %f"%((k_fold+1),crps))
        crps_csv.append(crps)
        
print("mean crps is %f"%np.mean(crps_csv))
print('mean crps is {} and std is {}'.format(np.mean(crps_csv), np.std(crps_csv)))
def nn_predict(x_te):
    model_num = len(models)
    for k,m in enumerate(models):
        if k==0:
            y_pred = m.predict(x_te,batch_size=1024)
        else:
            y_pred+=m.predict(x_te,batch_size=1024)
            
    y_pred = y_pred / model_num
    
    return y_pred
#if  TRAIN_OFFLINE==False:
from kaggle.competitions import nflrush
iter_test = env.iter_test()
%%time
test_x_lst = []
#for (test_df, sample_prediction_df) in iter_test:
#     try:
test_df_2 = test_df.copy()
test_df_3 = test_df.copy()
test_df_4 = test_df.copy()

#test_df['S_std'] = standardize_speed(test_df)
test_basetable = create_features(test_df, deploy=True)
test_adjusted_df = adjust_players_and_teams(test_df_2, player_schema, team_schema)
test_standardized_coor_df = standardize_coordinates(test_adjusted_df)
test_standardized_dir_df = standardize_direction(test_standardized_coor_df)
test_ol_agg_df = get_o_line_stats(test_standardized_dir_df)
test_merged_basetable = test_basetable.merge(test_ol_agg_df, on='PlayId')

test_fe_df = more_feature_engineering(test_standardized_dir_df, pos_dict)
test_fe_feature_df = test_fe_df[fe_feature_lst].drop_duplicates()

#Voronoi
test_bc_v_area = get_bc_voronoi_area(test_fe_df, 'current')
test_bc_v_area_end = get_bc_voronoi_area(test_fe_df, 'end')

test_fe_feature_df['bc_v_area'] = test_bc_v_area[0]
test_fe_feature_df['bc_v_area_end'] = test_bc_v_area_end[0]

test_def_dist_end_df = get_defense_end(test_fe_df)

test_fe_df[['X_std_end','Y_std_end']] = test_fe_df[['X_std_end','Y_std_end']].fillna(25)
#QB Location
test_qb_location_df = get_qb_location(test_fe_df)

#Time to Spot
test_def_time_to_spot_df = test_fe_df.loc[test_fe_df.IsOnOffense == False] \
.groupby('PlayId', as_index=False)[['TimeToBc','TimeToBcEnd']] \
. agg([np.min, np.mean, np.median, np.std])

test_def_time_to_spot_df.columns = test_def_time_to_spot_df.columns.droplevel(0)

test_def_time_to_spot_df.columns = ['TimeToBc' + c for c in test_def_time_to_spot_df.columns[0:4]] \
+ ['TimeToBcEnd' + c for c in test_def_time_to_spot_df.columns[4:]]

test_def_time_to_spot_df.reset_index(inplace=True)

#Top 3 Defenders Distance
test_fe_defenders = test_fe_df.loc[test_fe_df.IsOnOffense == False]
test_fe_defenders['DistFromBcRank'] = test_fe_defenders.groupby('PlayId')['DistFromBc'].transform(lambda x: x.rank())
test_three_def_df = test_fe_defenders.loc[test_fe_defenders.DistFromBcRank <= 3]
test_grouped_closest_dist_df = test_three_def_df.groupby('PlayId', as_index=False)['DistFromBc'].mean()
test_grouped_closest_dist_df.rename(columns={'DistFromBc':'Top3DistFromBcMean'}, inplace=True)

#Merge Tables
test_merged_basetable = test_merged_basetable.merge(test_def_dist_end_df, how = 'left', on='PlayId') \
    .merge(test_fe_feature_df, how = 'left', on=['GameId','PlayId']) \
    .merge(test_qb_location_df, how = 'left', on = 'PlayId') \
    .merge(test_def_time_to_spot_df, how = 'left', on = 'PlayId') \
    .merge(test_grouped_closest_dist_df, how = 'left', on = 'PlayId').fillna(25)

test_merged_basetable['Shotgun_Down_Distance'] = test_merged_basetable['Shotgun'] * test_merged_basetable['Down'] * test_merged_basetable['Distance']

#Influence
# test_dominance_df = standardize_dataset(test_df_3)

# test_control = Controller(test_dominance_df)
# #Bc influence
# test_bc_coords = list(*test_dominance_df.loc[(test_dominance_df.IsBallCarrier), ['X_std','Y_std']].values)
# test_pitch_control = test_control.vec_control(*test_bc_coords)
# test_merged_basetable['BcInfl'] = np.asscalar(test_pitch_control)

#Centroid
test_merged_basetable['def_centroid'] = get_defense_centroid(test_fe_df)
test_merged_basetable['def_centroid_X_std'] = test_merged_basetable.def_centroid.apply(lambda x: x[0])
test_merged_basetable['def_centroid_Y_std'] = test_merged_basetable.def_centroid.apply(lambda x: x[1])

test_merged_basetable['BcDistFromDefCent'] = np.sqrt((test_merged_basetable.X_std_bc - test_merged_basetable.def_centroid_X_std)**2 
                           + (test_merged_basetable.Y_std_bc - test_merged_basetable.def_centroid_Y_std)**2)

test_merged_basetable['BcDistFromQb'] = np.sqrt((test_merged_basetable.X_std_bc - test_merged_basetable.X_std_qb)**2 
                           + (test_merged_basetable.Y_std_bc - test_merged_basetable.Y_std_qb)**2)

test_merged_basetable['BcDirFromMid'] = abs(test_merged_basetable['BcDir'] - 90)

#One hot
#     test_merged_basetable['Down2'] = np.where(test_merged_basetable.Down == 2, 1, 0)
#     test_merged_basetable['Down3'] = np.where(test_merged_basetable.Down == 3, 1, 0)
test_merged_basetable['Down4'] = np.where(test_merged_basetable.Down == 4, 1, 0)
#     test_merged_basetable['Quarter2'] = np.where(test_merged_basetable.Quarter == 2, 1, 0)
#     test_merged_basetable['Quarter3'] = np.where(test_merged_basetable.Quarter == 3, 1, 0)
test_merged_basetable['Quarter4'] = np.where(test_merged_basetable.Quarter == 4, 1, 0)
#     test_merged_basetable['Quarter5'] = np.where(test_merged_basetable.Quarter == 5, 1, 0)

test_merged_basetable.drop(['GameId','PlayId'] + cols_to_drop, axis=1, inplace=True)

#Clustering
test_cluster_df = test_merged_basetable[cluster_features]
test_data_transformed = scale_data(test_cluster_df)
test_merged_basetable['RunCluster'] = cluster_fit.predict(test_data_transformed)

#LOS
#         play = test_df_4.reset_index()
#         playid = play.PlayId[0]
#         pos = encodepos(play,uniqueteams)
#         fpos = encodefpos(play,uniqueteams)
#         play,ha,ydline,bc = changedirection(play,pos,fpos)
#         olnum,olnumeng,numdefbf,oline = hmengagedextra(play,ha,ydline,bc)
#         holes = np.diff(oline.Y)

#         test_merged_basetable['olnums'] = sum(olnum)
#         test_merged_basetable['olnumseng'] = sum(olnumeng)
#         test_merged_basetable['numdefsbf'] = sum(numdefbf)
#         test_merged_basetable['avgholesize'] = np.mean(holes)

#         test_merged_basetable['numdefsb_cube'] = test_merged_basetable['numdefsbf'] ** 3
#         test_merged_basetable['numdefsbf_cube_div_min_dist'] =  test_merged_basetable['numdefsb_cube'] / test_merged_basetable['def_min_dist']

# test_merged_basetable['BcInfl'] = test_merged_basetable['BcInfl'].fillna(.5)
# test_merged_basetable['v_infl'] = (test_merged_basetable['BcInfl'] * test_merged_basetable['bc_v_area_end']).fillna(25)

test_merged_basetable = test_merged_basetable[list(merged_basetable.drop(['GameId','PlayId','Yards'], axis=1).columns)].fillna(1)
test_x_lst.append(test_merged_basetable)

test_binary_array = np.array(test_merged_basetable.loc[:, binary_features])

test_numeric_df = test_merged_basetable.drop(binary_features, axis = 1)
test_numeric_scaled = scaler.transform(test_numeric_df)
test_scaled_basetable = np.hstack((test_numeric_scaled, test_binary_array))

nn_y_pred = nn_predict(test_scaled_basetable)
nn_y_pred = np.clip(np.cumsum(nn_y_pred, axis=1), 0, 1).tolist()[0]

rf_y_pred = rf.predict(test_scaled_basetable)
rf_y_pred = np.clip(np.cumsum(rf_y_pred, axis=1), 0, 1)

blended_y_pred = (nn_y_pred + rf_y_pred) / 2

preds_df = pd.DataFrame(data=[nn_y_pred], columns=sample_prediction_df.columns)

env.predict(preds_df)
#     except:
#         print('Actual Prediction Failed')
#         zero_array = np.zeros(199)
#         preds_df = pd.DataFrame(data=[zero_array], columns=sample_prediction_df.columns)
#         preds_df.iloc[:,102] = .5
#         preds_df.iloc[:,103:] = 1

#         env.predict(preds_df)

# env.write_submission_file()
test_merged_basetable
preds_df
