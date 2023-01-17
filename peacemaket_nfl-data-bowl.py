# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import pickle
import pandas as pd
import numpy as np
import datetime as dt
import time
pd.options.display.max_columns = 100

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

import math
from kaggle.competitions import nflrush
env = nflrush.make_env()
df = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv', sep=',', low_memory=False)
df.head()
#from game clock to seconds
def from_gameClock_to_sec(x):
    x = time.strptime(x[:5],'%M:%S')
    return dt.timedelta(hours=x.tm_hour,minutes=x.tm_min,seconds=x.tm_sec).total_seconds()

#transform str feet-inches to inches
def from_str_height_to_int(x):
    x = x.split('-')
    return int(x[0])*12 + int(x[1])


# eg - from '5 DL, 3 LB, 2 DB, 1 OL' to different columns with counts
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

def sin(x):
    return np.sin((x + 90) * math.pi / 180)

def cos(x):
    return -np.cos((x + 90) * math.pi / 180)

def tan(x):
    return np.tan((x + 90) * math.pi / 180)
    
def preproc(df, possesion_def_groupby=[], drop_cols=[], train=True):
    map_abbr = {'ARI': 'ARZ', 'BAL': 'BLT', 'CLE': 'CLV', 'HOU': 'HST'}
    for abb in df['PossessionTeam'].unique():
        map_abbr[abb] = abb
    for abb in df['HomeTeamAbbr'].unique():
        map_abbr[abb] = abb
    for abb in df['VisitorTeamAbbr'].unique():
        map_abbr[abb] = abb

    df['PossessionTeam'] = df['PossessionTeam'].map(map_abbr)
    df['HomeTeamAbbr'] = df['HomeTeamAbbr'].map(map_abbr)
    df['VisitorTeamAbbr'] = df['VisitorTeamAbbr'].map(map_abbr)
    df['GameClockSec'] = df['GameClock'].apply(from_gameClock_to_sec)


    #inpute nan values with mode values
    df.OffenseFormation = df.OffenseFormation.fillna('SINGLEBACK')
    df.DefendersInTheBox = df.DefendersInTheBox.fillna(7)

    df['DefendersInTheBox_vs_Distance'] = df['DefendersInTheBox'] / df['Distance']

    df.PlayerHeight = df.PlayerHeight.apply(from_str_height_to_int)

    # impute nan FieldPosition
    nanfieldpos = df[df['FieldPosition'].isna()]['PlayId'].unique()
    for i in nanfieldpos:
        tmp = df[(df['PlayId']==i) & (df['NflIdRusher']==df['NflId'])]
        if tmp['PlayDirection'].item() == 'left':
            if tmp['X'].item()>=60:
                df.loc[df[(df['PlayId']==i)].index, 'FieldPosition'] = tmp['PossessionTeam'].item()
            elif tmp['PossessionTeam'].item() == tmp['VisitorTeamAbbr'].item():
                df.loc[df[(df['PlayId']==i)].index, 'FieldPosition'] = tmp['HomeTeamAbbr'].item()
            else:
                df.loc[df[(df['PlayId']==i)].index, 'FieldPosition'] = tmp['VisitorTeamAbbr'].item()
        else:
            if tmp['X'].item()<=60:
                df.loc[df[(df['PlayId']==i)].index, 'FieldPosition'] = tmp['PossessionTeam'].item()
            elif tmp['PossessionTeam'].item() == tmp['VisitorTeamAbbr'].item():
                df.loc[df[(df['PlayId']==i)].index, 'FieldPosition'] = tmp['HomeTeamAbbr'].item()
            else:
                df.loc[df[(df['PlayId']==i)].index, 'FieldPosition'] = tmp['VisitorTeamAbbr'].item()

    df['HomePossesion'] = df['PossessionTeam'] == df['HomeTeamAbbr']
    df['isHome'] = df.Team == 'home'

    df['score_dif'] = df['HomeScoreBeforePlay'].values - df['VisitorScoreBeforePlay'].values
    df.loc[~df['HomePossesion'],  'score_dif'] = -df.loc[~df['HomePossesion'],  'score_dif'].values

    df['possesion_win'] = 0
    df.loc[df.score_dif < 0, 'possesion_win'] = -1
    df.loc[df.score_dif > 0, 'possesion_win'] = 1

    df['ToLeft'] = df['PlayDirection'].values == "left"
    df['IsBallCarrier'] = df['NflId'].values == df.NflIdRusher

    df.loc[df['Dir'].isna(), 'Dir'] = 0
    df.loc[df['Orientation'].isna(), 'Orientation'] = 0
    # df['Dir_std'] = np.mod(90 - df['Dir'].values, 360)

    # time features
    df['TimeHandoff'] = df['TimeHandoff'].apply(lambda x: dt.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))
    df['TimeSnap'] = df['TimeSnap'].apply(lambda x: dt.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))

    df.PlayerBirthDate = pd.to_datetime(df.PlayerBirthDate, infer_datetime_format=True)
    #Player Age
    df['PlayerAge'] = (df.TimeHandoff - df.PlayerBirthDate).dt.days / 365

    # dif in seconds between Handoff and Snap
    df['time_dif'] = (df['TimeHandoff'] - df['TimeSnap']).dt.seconds


    ## OffensePersonnel
    temp = df["OffensePersonnel"].iloc[np.arange(0, len(df), 22)].apply(lambda x : pd.Series(OffensePersonnelSplit(x)))
    temp.columns = ["Offense_" + c for c in temp.columns]
    temp["PlayId"] = df["PlayId"].iloc[np.arange(0, len(df), 22)]
    df = df.merge(temp, on = "PlayId")

    ## DefensePersonnel
    temp = df["DefensePersonnel"].iloc[np.arange(0, len(df), 22)].apply(lambda x : pd.Series(DefensePersonnelSplit(x)))
    temp.columns = ["Defense_" + c for c in temp.columns]
    temp["PlayId"] = df["PlayId"].iloc[np.arange(0, len(df), 22)]
    df = df.merge(temp, on = "PlayId")

    # determine defense team for further grouping about loosing and gaining yards for offense and defense separetly   
    df['DefenseTeam'] = df.HomeTeamAbbr.values
    df.loc[df.PossessionTeam == df.HomeTeamAbbr, 'DefenseTeam'] = df.loc[df.PossessionTeam == df.HomeTeamAbbr, 'VisitorTeamAbbr'].values

    cat_features = ['OffenseFormation', 'Position', 'PossessionTeam', 'DefenseTeam', ]
#     print(df.shape)
    #one hot encoding for cat_features
    if train == True:
        ohe = OneHotEncoder(sparse=False, categories='auto', dtype=np.int8)

        values = ohe.fit_transform(df[cat_features])

        filename = 'ohe.preproc'
        pickle.dump(ohe, open(filename, 'wb'))
    else:
        filename = 'ohe.preproc'
        ohe = pickle.load(open(filename, 'rb'))
        values = ohe.transform(df[cat_features])
        

    for ind, col in enumerate(ohe.get_feature_names()):
        local_ind = int(col[1])
        col = col.replace('x%s'%local_ind, cat_features[local_ind])
        df[col] = values[:, ind]

        
    #####
    ## normalize X
    #####
    
    
    df.loc[df.FieldPosition != df.PossessionTeam, 'YardLine'] = 100 - df.loc[df.FieldPosition != df.PossessionTeam, 'YardLine'].values
    df['yards_for_touchdown'] = 100 - df.YardLine.values

    df.loc[df.ToLeft, 'X'] = 120 - df.loc[df.ToLeft, 'X'].values
    df.loc[df.ToLeft, 'Y'] = 160/3 - df.loc[df.ToLeft, 'Y'].values

    df.loc[df.ToLeft, 'Orientation'] = np.mod(180 + df.loc[df.ToLeft, 'Orientation'].values, 360)
    df.loc[df.ToLeft, 'Dir'] = np.mod(180 + df.loc[df.ToLeft, 'Dir'].values, 360)

    # remove bias from Orientation during Season 2017 - https://www.kaggle.com/peterhurford/something-is-the-matter-with-orientation
    # althought the author states that it's getting worse on LB score
    df.loc[df['Season'] == 2017, 'Orientation'] = np.mod(90 + df.loc[df['Season'] == 2017, 'Orientation'].values, 360)


    # cos - going right - positive, going left - negative, going up or down - neutral
    df['DirCos'] = cos(df.Dir.values)
    df['OrientationCos'] = cos(df.Orientation.values)

    if 'PlayDirection' in df.columns:
        # flip values when change Left PlayDirection to Right PlayDirection
        df.loc[df.ToLeft, 'DirCos'] = - df.loc[df.ToLeft, 'DirCos'].values
        df.loc[df.ToLeft, 'OrientationCos'] = - df.loc[df.ToLeft, 'OrientationCos'].values

    # sin - going up - positive, going down - negative, going right or left - neutral
    df['DirSin'] = sin(df.Dir.values)
    df['OrientationSin'] = sin(df.Orientation.values)
    #### ------

    # # YardLine - X = 0
    df['X'] = df['X'].values - df['YardLine'].values - 10

    df['Distance_to_YardLine'] = abs(df['X'].values)
    
    df.drop(['DisplayName', 'JerseyNumber', 'Stadium', 
         'Location', 'StadiumType', 'Turf', 'GameWeather', 
         'Temperature', 'Humidity', 'WindSpeed', 'WindDirection',
         'PlayDirection', 'GameClock',
        'OffensePersonnel', 'DefensePersonnel', 'ToLeft', 'PlayDirection', 
        'FieldPosition', 'HomeScoreBeforePlay', 'VisitorScoreBeforePlay',
        'PlayerBirthDate', 'PlayerCollegeName', 'GameId'], axis=1, inplace=True)

    df = df.merge(df[df.IsBallCarrier][['PlayId', 'X', 'Y', 'isHome', 'Dir', 'Orientation']].rename(columns={'isHome': 'isHome_Carrier', 
                                                                                                             'X': 'X_carrier', 
                                                                                                             'Y': 'Y_carrier',
                                                                                                             'Dir': 'Dir_carrier',
                                                                                                             'Orientation': 'Orientation_carrier'}), 
                  on='PlayId')

    df['isPossession'] = df['isHome_Carrier'] == df['isHome']
    df = df.sort_values(['PlayId', 'IsBallCarrier', 'isPossession', 'X'], ascending=[True, False, False, True]).reset_index(drop=True)
    
    df['Distance_to_BallCarrier'] = np.power(np.power(df.Y_carrier - df.Y, 2) + np.power(df.X_carrier - df.X, 2), 0.5)

    #### ----
    for window in [10, 20, 30, 40]:
        df['Dir_window_%s'%window] = (df.X > df.X_carrier) & (df.Y < tan(df.Dir_carrier + window) * (df.X - df.X_carrier) + df.Y_carrier) & (df.Y < tan(df.Dir_carrier - window) * (df.X - df.X_carrier) + df.Y_carrier)
        df['Orientation_window_%s'%window] = (df.X > df.X_carrier) & (df.Y < tan(df.Orientation_carrier + window) * (df.X - df.X_carrier) + df.Y_carrier) & (df.Y < tan(df.Orientation_carrier - window) * (df.X - df.X_carrier) + df.Y_carrier)

    df['upper_members'] = df.Y_carrier > df.Y
    df['lower_members'] = df.Y_carrier < df.Y
    #### ----

    only_carrier = df[df.IsBallCarrier].reset_index(drop=True)
    
    #### ------
    #Label Encoding for cat_features
    for col in cat_features:
        if train == True:
            le = LabelEncoder()
            values = le.fit_transform(only_carrier[col])

            filename = '%s_le.preproc'%col
            pickle.dump(le, open(filename, 'wb'))
        else:
            filename = '%s_le.preproc'%col
            le = pickle.load(open(filename, 'rb'))
            
            values = le.transform(only_carrier[col])

        only_carrier[col] = values
    #### ------
    
    #### ------
    clmns = ['upper_members', 'lower_members'] + ['%s_window_%s'%(j, window) for window in [10, 20, 30, 40] for j in ['Dir', 'Orientation']]
    tmp = df[~df.IsBallCarrier].groupby(['PlayId', 'isPossession'])[clmns].agg('sum').reset_index()

    only_carrier = only_carrier.merge(tmp.loc[np.arange(0, tmp.shape[0], 2)].rename(columns={i: i+'_count_offense' for i in clmns}).iloc[:, [i for i in range(tmp.shape[1]) if i!=1]],
            on = 'PlayId')
    only_carrier = only_carrier.merge(tmp.loc[np.arange(1, tmp.shape[0], 2)].rename(columns={i: i+'_count_defense' for i in clmns}).iloc[:, [i for i in range(tmp.shape[1]) if i!=1]],
            on = 'PlayId')
    #### ------

    only_carrier.drop(clmns, axis=1, inplace=True)
    df.drop(clmns, axis=1, inplace=True)
    
    #### ------
    tmp = df[~df.IsBallCarrier].groupby(['PlayId', 'isPossession'])['Distance_to_YardLine'].agg(['min', 'max', 'median', 'mean', 'std']).reset_index()

    only_carrier = only_carrier.merge(tmp.loc[np.arange(0, tmp.shape[0], 2)].rename(columns={i: 'Distance_to_YardLine_' + i + '_offense' for i in tmp.columns[2:]}).iloc[:, [i for i in range(tmp.shape[1]) if i!=1]],
            on = 'PlayId')
    only_carrier = only_carrier.merge(tmp.loc[np.arange(1, tmp.shape[0], 2)].rename(columns={i: 'Distance_to_YardLine_' + i + '_defense' for i in tmp.columns[2:]}).iloc[:, [i for i in range(tmp.shape[1]) if i!=1]],
            on = 'PlayId')
    #### ------

    #### ------
    tmp = df[~df.IsBallCarrier].groupby(['PlayId', 'isPossession'])['Distance_to_BallCarrier'].agg(['min', 'max', 'median', 'mean', 'std']).reset_index()

    only_carrier = only_carrier.merge(tmp.loc[np.arange(0, tmp.shape[0], 2)].rename(columns={i: 'Distance_to_BallCarrier_' + i + '_offense' for i in tmp.columns[2:]}).iloc[:, [i for i in range(tmp.shape[1]) if i!=1]],
            on = 'PlayId')
    only_carrier = only_carrier.merge(tmp.loc[np.arange(1, tmp.shape[0], 2)].rename(columns={i: 'Distance_to_BallCarrier_' + i + '_defense' for i in tmp.columns[2:]}).iloc[:, [i for i in range(tmp.shape[1]) if i!=1]],
            on = 'PlayId')
    #### ------

    #### ------
    # tm = time.time()
    anal_results = []
    for i in np.arange(0, df.shape[0], 22):
        values = df.iloc[i:i+22][['X', 'Y']].values

        vls = np.sqrt(np.power(values[:11, 0].reshape(-1, 1) - values[11:, 0], 2) + np.power(values[:11, 1].reshape(-1, 1) - values[11:, 1], 2))
        vls[values[:11, 0].reshape(-1, 1)>=values[11:, 0]] = 100000

        distance = []
        for _ in range(11):
            if (vls == 100000).all():
                break
            i, j = np.unravel_index(vls.argmin(), vls.shape)
            distance.append(vls[i, j])
            vls[i, :] = 100000
            vls[:, j] = 100000

        if distance:
            anal_results.append((len(distance), min(distance), max(distance), np.mean(distance), np.median(distance), np.std(distance)))
        else:
            anal_results.append((0, 0, 0, 0, 0, 0))
    # print((time.time() - tm))# * 23171
    anal_results = pd.DataFrame(anal_results, columns=['count_dist', 'min_dist', 'max_dist', 'mean_dist', 'median_dist', 'std_dist'])
    only_carrier = only_carrier.merge(anal_results, left_index=True, right_index=True)
    #### ------
#     print(only_carrier.head())
    #### ------
    for ind, j in enumerate(['Possession', 'Defense']):
        if train == True:
#             tmp = only_carrier.groupby(['%sTeam'%j, 'Season'])['Yards'].agg({
#                                                                                               'mean': np.mean, 
#                                                                                               'perc50': lambda x: np.percentile(x, 50), 
#                                                                                               'perc25': lambda x: np.percentile(x, 25),
#                                                                                               'perc75': lambda x: np.percentile(x, 75),
#                                                                                               'perc95': lambda x: np.percentile(x, 95),
#                                                                                               'perc05': lambda x: np.percentile(x, 5),
#                                                                                               'std': np.std,
#                                                                                              }).reset_index()#.head()


#             if '%sTeamYards_2017_mean'%j not in only_carrier.columns:
#                 only_carrier = only_carrier.merge(tmp.loc[tmp.Season == 2017, ['%sTeam'%j] + list(tmp.columns[2:])].rename(columns={i: '%sTeamYards_2017_%s'%(j, i) for i in tmp.columns[2:]}), on='%sTeam'%j)
#                 only_carrier = only_carrier.merge(tmp.loc[tmp.Season == 2018, ['%sTeam'%j] + list(tmp.columns[2:])].rename(columns={i: '%sTeamYards_2018_%s'%(j, i) for i in tmp.columns[2:]}), on='%sTeam'%j)
        
            tmp = only_carrier.groupby('%sTeam'%j)['Yards'].agg({
                                                                                              'mean': np.mean, 
                                                                                              'perc50': lambda x: np.percentile(x, 50), 
                                                                                              'perc25': lambda x: np.percentile(x, 25),
                                                                                              'perc75': lambda x: np.percentile(x, 75),
                                                                                              'perc95': lambda x: np.percentile(x, 95),
                                                                                              'perc05': lambda x: np.percentile(x, 5),
                                                                                              'std': np.std,
                                                                                             }).reset_index()
            
            tmp = tmp.rename(columns={i: '%sTeamYards_%s'%(j, i) for i in tmp.columns[1:]})
            possesion_def_groupby.append(tmp)
            tmp.to_csv('%sTeam.csv'%j, sep=';', index=False)
            
            if '%sTeamYards_mean'%j not in only_carrier.columns:
                only_carrier = only_carrier.merge(tmp, on='%sTeam'%j)
        else:#eval mode
            if len(possesion_def_groupby) != 2:
                tmp = pd.read_csv('%sTeam.csv'%j, sep=';')
                possesion_def_groupby.append(tmp)
            else:
                tmp = possesion_def_groupby[ind]
                
            tmp = tmp[tmp['%sTeam'%j] == only_carrier.iloc[0]['%sTeam'%j]]
            
            for col in tmp.columns[1:]:
                only_carrier[col] = tmp[col].values

            
    #### ------
    if not drop_cols:
        drop_cols = [i for i in only_carrier.columns if only_carrier[i].unique().shape[0]==1 or i in ['HomePossesion', 
                                                                                             'isHome_Carrier', 
                                                                                             'Dir_carrier',
                                                                                             'Orientation_carrier',]]
    only_carrier.drop(drop_cols, axis=1, inplace=True)
    if train == True:
        return df, only_carrier, possesion_def_groupby, drop_cols, cat_features
    return df, only_carrier, possesion_def_groupby, cat_features

possesion_def_groupby = []
df, only_carrier, possesion_def_groupby, drop_cols, cat_features = preproc(df, possesion_def_groupby, drop_cols=[], train=True)
only_carrier.shape
pickle.dump(cat_features, open('cat_featues', 'wb'))
pickle.dump(drop_cols, open('drop_cols', 'wb'))
ex_columns = ['PlayId', 'Team', 'X_carrier', 'Y_carrier', 'NflId',
             'Season', 'NflIdRusher', 'TimeHandoff', 'TimeSnap',
              'HomeTeamAbbr', 'VisitorTeamAbbr', 'YardLine',
             ]

# tmp = [i + '_' for i in cat_features]
# ohe_columns = [i for i in only_carrier.columns if any(j in i for j in tmp)]

# another_cat_features = ['Quarter', 'Down', 'Week']

# nn_with_embed_columns = [i for i in only_carrier.columns if i not in ex_columns + ohe_columns and '201' not in i]
nn_with_ohe_columns = [i for i in only_carrier.columns if i not in ex_columns + cat_features and '201' not in i]
# print('nn_with_embed_columns', len(nn_with_embed_columns) - 1, 'nn_with_ohe_columns', len(nn_with_ohe_columns) - 1)

# cat_embedings_in = [only_carrier[i].unique().shape[0] for i in cat_features]

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
class Net(nn.Module):

    def __init__(self, 
                 architecture, 
                 cat_embedings_in=[],
                 embed_in_scale_param=0.75, 
                 is_regression=True, 
                 is_restnet=False,
                 batch_norm=True,
                 dropout=0,
                ):
        
        super(Net, self).__init__()
        
        self.is_regression = is_regression
        self.is_restnet = is_restnet
        self.dropout = dropout
        self.batch_norm = batch_norm
        
        self.cat_embedings_in = cat_embedings_in
        if cat_embedings_in:
            self.embeding_dims = [max(1, int(round(embed_in_scale_param * i))) for i in cat_embedings_in]
            architecture[0] = architecture[0] - len(self.embeding_dims) + sum(self.embeding_dims)

            self.embeds = nn.ModuleList([nn.Embedding(num_embeddings = cat_embedings_in[i], 
                                        embedding_dim = self.embeding_dims[i]) 
                           for i in range(len(cat_embedings_in))])
        
        self.fc = []
        if batch_norm:
            self.bn = []
        if dropout>0:
            self.dp = []
            
        for ind in range(len(architecture[:-1])):
            input_dim = architecture[ind]
            if is_restnet and ind>0:
                input_dim += architecture[ind - 1]
            
            self.fc.append(nn.Linear(input_dim, architecture[ind + 1]))
            
            if batch_norm:
                self.bn.append(nn.BatchNorm1d(architecture[ind + 1]))
            if dropout>0:
                self.dp.append(nn.Dropout(dropout))
                
        self.fc = nn.ModuleList(self.fc)
#         print([i for i in self.fc])
        if batch_norm:
            self.bn.pop()
            if self.bn:
                self.bn = nn.ModuleList(self.bn)
#                 print([i for i in self.bn])
        if dropout>0:
            self.dp.pop()
            if self.dp:
                self.dp = nn.ModuleList(self.dp)
#                 print([i for i in self.dp])
                
            

    def forward(self, x):
        x = self.feature_extract(x)
        x = self.fc[-1](x)
        if self.is_regression:
            return x
        return F.softmax(x)
    

    def feature_extract(self, x):
        
        if self.cat_embedings_in:
            y = None
            for ind in range(len(self.embeds)):
                if y is None:
                    y = self.embeds[ind](x[:, ind].type(torch.long))
                else:
                    y = torch.cat((self.embeds[ind](x[:, ind].type(torch.long)), y), dim = 1)

            x = torch.cat((y.float(), x[:, len(self.embeds):].float()), dim=1)
        else:
            x = x.float()
            
        if self.is_restnet:
            last_x = x
            
        for ind in range(len(self.fc)):
            
            if self.is_restnet and ind>0:
                x = torch.cat((last_x.float(), x), dim=1)
                last_x = x[:, x.shape[1] - self.fc[ind-1].out_features:]
                
            if ind == len(self.fc) - 1:
                return x
            
            x = self.fc[ind](x)
            if self.bn:
                x = self.bn[ind](x)
            if self.dp:
                x = self.dp[ind](x)
                
        return x
    

class CustomsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, X, Y):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.X = torch.tensor((X * 1).astype(np.float), dtype=torch.float16)
        self.Y = torch.tensor(Y.reshape(-1, 1), dtype=torch.float)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return [self.X[idx], self.Y[idx]]


def torch_train_eval(model, dataloader, criterion, optimizer, device='cuda:0', isTrain = True):
    model.train(isTrain)
    cum_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data
        cnt = inputs.shape[0]
        if 'cuda' in device:
            inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        outputs = net(inputs)
        if 'cuda' in device:
            outputs = outputs.to(device)
        loss = criterion(outputs, labels)
        
        if isTrain:
            loss.backward()
            optimizer.step()

        cum_loss += loss.item() * cnt
        
    return cum_loss

def save_model_params():
    print('Saving model parameters...')
    pickle.dump({'architecture': [X_train.shape[1], 256, 128, 1], 
                 'dropout': 0.5}, open('model.params', 'wb'))

def load_model_params():
    print('Loading model parameters...')
    return pickle.load(open('model.params', 'rb'))

def save_model(model, path='model.model', states_only=False):
    if states_only:
        torch.save(model.state_dict(), path + '_states')
    else:
        torch.save(model, path)
    print('Model has been saved...')
    
def load_model(path='model.model', states_only=False):
    print('Loading the model...')
    if states_only:
        model = Net(**load_model_params())
        model.load_state_dict(torch.load(path + '_states'))
    else:
        model = torch.load(path)
    model.eval()
    return model

def scale(only_carrier, scaler=None, non_bin_features=[]):

    # scaled_only_carrier = only_carrier[only_carrier['Yards']<=30][nn_with_embed_columns].reset_index(drop=True).copy()
    scaled_only_carrier = only_carrier.reset_index(drop=True).copy()

    if scaler is None:
        non_bin_features = [i for i in nn_with_ohe_columns if i not in cat_features + ['Yards'] and only_carrier[i].unique().shape[0]>2]
        pickle.dump(non_bin_features, open('non_bin_features', 'wb'))
        
        scaler = StandardScaler()
        scaled_only_carrier[non_bin_features] = scaler.fit_transform(scaled_only_carrier[non_bin_features])

        pickle.dump(scaler, open('StandardScaler.scaler', 'wb'))
    else: #eval mode
#         non_bin_features = pickle.load(open('non_bin_features', 'rb'))
#         scaler = pickle.load(open('StandardScaler.scaler', 'rb'))
        scaled_only_carrier[non_bin_features] = scaler.transform(scaled_only_carrier[non_bin_features])

    return scaled_only_carrier

def predict(model, X, device='cuda:0'):
    if len(X.shape) == 1:
        X = torch.tensor((X.reshape(1, -1) * 1).astype(np.float), dtype=torch.float16)
    else:
        X = torch.tensor((X * 1).astype(np.float), dtype=torch.float16)
    if 'cuda' in device:
        X = X.to(device)
    return model(X).cpu().detach().numpy().flatten()
scaled_only_carrier = scale(only_carrier[nn_with_ohe_columns])

train_ratio = 0.8
y_col = 'Yards'
other_columns = [i for i in scaled_only_carrier.columns if i not in [y_col] + cat_features]
X_columns = other_columns

X_train, Y_train = scaled_only_carrier.loc[:int(scaled_only_carrier.shape[0]*train_ratio), X_columns].values, scaled_only_carrier.loc[:int(scaled_only_carrier.shape[0]*train_ratio), y_col].values
X_test, Y_test = scaled_only_carrier.loc[int(scaled_only_carrier.shape[0]*train_ratio):, X_columns].values, scaled_only_carrier.loc[int(scaled_only_carrier.shape[0]*train_ratio):, y_col].values

device = "cpu" # cpu cuda:0

print(device)
net = Net([X_train.shape[1], 256, 128, 1], cat_embedings_in=[], is_restnet=False, batch_norm=True, dropout=0.5,
         embed_in_scale_param=0.3
         )
net.to(device)

torch_trainset = CustomsDataset(X_train, Y_train)
torch_validset = CustomsDataset(X_test, Y_test)

batch_size = 32
train_dataloader = DataLoader(torch_trainset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(torch_validset, batch_size=batch_size, shuffle=True)

criterion = nn.L1Loss()
optimizer = optim.Adam(net.parameters(), lr=0.0005)

n_epochs = 5
# early_stop_epoch = 2

for epoch in range(n_epochs):  # loop over the dataset multiple times
    
    train_loss = torch_train_eval(net, train_dataloader, criterion, optimizer, device=device, isTrain=True)
    
    valid_loss = torch_train_eval(net, valid_dataloader, criterion, optimizer, device=device, isTrain=False)
        
    print(f'epoch: {epoch}, trainloss: {train_loss/Y_train.shape[0]}, validloss: {valid_loss/Y_test.shape[0]}')
#     break
    
print('Finished Training')

save_model(net)
predict(net, X_train[0], device)
def make_prediction(prediction, sample_pred):
    prediction = round(prediction[0], 6)
    ind = 99 + int(max(-99, min(98, prediction)))
    sample_pred.iloc[0, np.arange(0, ind, 1)] = 0
    sample_pred.iloc[0, ind] = 1 - (prediction - int(prediction)) if prediction - int(prediction) >= 0 else abs(prediction - int(prediction))
    sample_pred.iloc[0,  np.arange(ind+1, 199, 1)] = 1
    return sample_pred
drop_cols = pickle.load(open('drop_cols', 'rb'))

cat_features = ['OffenseFormation', 'Position', 'PossessionTeam', 'DefenseTeam', ]
ex_columns = ['PlayId', 'Team', 'X_carrier', 'Y_carrier', 'NflId',
         'Season', 'NflIdRusher', 'TimeHandoff', 'TimeSnap',
          'HomeTeamAbbr', 'VisitorTeamAbbr', 'YardLine',
         ]

# tmp = [i + '_' for i in cat_features]
# ohe_columns = [i for i in only_carrier.columns if any(j in i for j in tmp)]

nn_with_ohe_columns = [i for i in only_carrier.columns if i != 'Yards' and i not in ex_columns + cat_features and '201' not in i]
X_columns = nn_with_ohe_columns

scaler = pickle.load(open('StandardScaler.scaler', 'rb'))
non_bin_features = pickle.load(open('non_bin_features', 'rb'))

model = load_model()
device = 'cpu'

isTrain = False

for test_df, sample_pred in env.iter_test():
    ttt  = test_df.copy()
    test_df, only_carrier, possesion_def_groupby = preproc(test_df, possesion_def_groupby, drop_cols, train=isTrain)[:3]
    
    scaled_only_carrier = scale(only_carrier[nn_with_ohe_columns], scaler=scaler, non_bin_features=non_bin_features)
    
    prediction = predict(model, scaled_only_carrier.values, device=device)
    
    sample_pred = make_prediction(prediction, sample_pred)
    env.predict(sample_pred)


env.write_submission_file()
