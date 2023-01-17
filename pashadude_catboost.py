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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.patches as patches

import math

import pprint

import os

import gc

import datetime

from time import time

import seaborn as sns

import datetime, tqdm

from kaggle.competitions import nflrush

from sklearn.model_selection import GroupKFold, StratifiedKFold, train_test_split

from catboost import Pool, CatBoostRegressor

from sklearn.preprocessing import LabelEncoder, StandardScaler



print(os.listdir("../input"))

env = nflrush.make_env()
offline = True



if offline:

    train = pd.read_csv("../input/nfl-big-data-bowl-2020/train.csv", low_memory=False, dtype={'WindSpeed': 'object'})

    #test = pd.read_csv('test.csv', low_memory=False,  dtype={'WindSpeed': 'object'})

else:

    train = pd.read_csv("../input/nfl-big-data-bowl-2020/train.csv", low_memory=False, dtype={'WindSpeed': 'object'})

    test = pd.read_csv('../input/nfl-big-data-bowl-2020/test.csv', low_memory=False,  dtype={'WindSpeed': 'object'})
def strtoint(x):

    try:

        return int(x)

    except:

        return 0

    

def switch_speed_direction(x):

    if x == 'calm':

        x = 0

    elif x == 'e':

        x = 8

    elif x == 'se':

        x = 1

    elif x == 'ssw':

        x = 13

    return x



def switch_direction_speed(x):

    if x == '8':

        x = 'E'

    elif x == '1':

        x = 'SE'

    elif x == '13':

        x = 'SSW'

    return x



def strtoseconds(txt):

    txt = txt.split(':')

    ans = int(txt[0])*60 + int(txt[1]) + int(txt[2])/60

    return ans



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



def transform_StadiumType(txt):

    if pd.isna(txt):

        return np.nan

    if 'outdoor' in txt or 'open' in txt:

        return True

    if 'indoor' in txt or 'closed' in txt:

        return False

    

    return np.nan



def get_offense_personel(offense_scheme):

    '''

    Get's the number of persons from the OffensePersonnel column

    '''

    list_of_values = offense_scheme.split()

    counter = 0

    for val in list_of_values:

        try :

            counter += int(val)

        except:

            pass

    return counter



def OffensePersonnelSplit(x):

    dic = {'DB' : 0, 'DL' : 0, 'LB' : 0, 'OL' : 0, 'QB' : 0, 'RB' : 0, 'TE' : 0, 'WR' : 0}

    #dic = {'OL' : 0, 'TE' : 0, 'WR' : 0}

    for xx in x.split(","):

        xxs = xx.split(" ")

        dic[xxs[-1]] = int(xxs[-2])

    return dic



def DefensePersonnelSplit(x):

    dic = {'DB' : 0, 'DL' : 0, 'LB' : 0, 'OL' : 0}

    #dic = {'DB' : 0, 'DL' : 0, 'LB' : 0}

    for xx in x.split(","):

        xxs = xx.split(" ")

        dic[xxs[-1]] = int(xxs[-2])

    return dic



outdoor = ['Outdoor', 'Outdoors', 'Cloudy', 'Heinz Field', 'Outdor', 'Ourdoor', 

           'Outside', 'Outddors','Outdoor Retr Roof-Open', 'Oudoor', 'Bowl']

indoor_closed = ['Indoors', 'Indoor', 'Indoor, Roof Closed', 'Indoor, Roof Closed', 'Retractable Roof',

                 'Retr. Roof-Closed', 'Retr. Roof - Closed', 'Retr. Roof Closed']

retractable = ['Outdoor Retr Roof-Open', 'Retractable Roof', 'Retr. Roof-Closed', 'Retr. Roof - Closed', 'Retr. Roof Closed',

              'Retr. Roof-Open', 'Retr. Roof - Open']

indoor_open   = ['Indoor, Open Roof', 'Open', 'Retr. Roof-Open', 'Retr. Roof - Open']

dome_closed   = ['Dome', 'Domed, closed', 'Closed Dome', 'Domed', 'Dome, closed']

dome_open     = ['Domed, Open', 'Domed, open']

rain = ['Rainy', 'Rain Chance 40%', 'Showers','Cloudy with periods of rain, thunder possible. Winds shifting to WNW, 10-20 mph.',

        'Scattered Showers', 'Cloudy, Rain', 'Rain shower', 'Light Rain', 'Rain']



overcast = ['Cloudy, light snow accumulating 1-3"', 'Party Cloudy', 'Cloudy, chance of rain',

            'Coudy', 'Cloudy, 50% change of rain', 'Rain likely, temps in low 40s.',

            'Cloudy and cold', 'Cloudy, fog started developing in 2nd quarter',

            'Partly Clouidy', '30% Chance of Rain', 'Mostly Coudy', 'Cloudy and Cool',

            'cloudy', 'Partly cloudy', 'Overcast', 'Hazy', 'Mostly cloudy', 'Mostly Cloudy',

            'Partly Cloudy', 'Cloudy']



clear = ['Partly clear', 'Sunny and clear', 'Sun & clouds', 'Clear and Sunny',

        'Sunny and cold', 'Sunny Skies', 'Clear and Cool', 'Clear and sunny',

        'Sunny, highs to upper 80s', 'Mostly Sunny Skies', 'Cold',

        'Clear and warm', 'Sunny and warm', 'Clear and cold', 'Mostly sunny',

        'T: 51; H: 55; W: NW 10 mph', 'Clear Skies', 'Clear skies', 'Partly sunny',

        'Fair', 'Partly Sunny', 'Mostly Sunny', 'Clear', 'Sunny', 'Sunny, Windy']



snow  = ['Heavy lake effect snow', 'Snow']



none  = ['N/A Indoor', 'Indoors', 'Indoor', 'N/A (Indoors)', 'Controlled Climate']



north = ['N','From S','North']

south = ['S','From N','South','s']

west = ['W','From E','West']

east = ['E','From W','from W','EAST','East']

north_east = ['FROM SW','FROM SSW','FROM WSW','NE','NORTH EAST','North East','East North East','NorthEast','Northeast','ENE','From WSW','From SW']

north_west = ['E','From ESE','NW','NORTHWEST','N-NE','NNE','North/Northwest','W-NW','WNW','West Northwest','Northwest','NNW','From SSE']

south_east = ['E','From WNW','SE','SOUTHEAST','South Southeast','East Southeast','Southeast','SSE','From SSW','ESE','From NNW']

south_west = ['E','From ENE','SW','SOUTHWEST','W-SW','South Southwest','West-Southwest','WSW','SouthWest','Southwest','SSW','From NNE']

no_wind = ['clear','Calm']

natural_grass = ['natural grass','Naturall Grass','Natural Grass']

grass = ['Grass']

fieldturf = ['FieldTurf','Field turf','FieldTurf360','Field Turf']

artificial = ['Artificial','Artifical']

train['ToLeft'] = train.PlayDirection == "left"

train['IsBallCarrier'] = train.NflId == train.NflIdRusher

train['HomeTeamAbbr'] = train['HomeTeamAbbr'].replace({'CLE':'CLV', 'ARI':'ARZ', 'HOU':'HST', 'BAL':'BLT'})

train['Dir_rad'] = np.mod(90 - train.Dir, 360) * math.pi/180.0

train['TeamOnOffense'] = "home"

train.loc[train.PossessionTeam != train.HomeTeamAbbr, 'TeamOnOffense'] = "away"

train['IsOnOffense'] = train.Team == train.TeamOnOffense

train['YardLine_std'] = 100 - train.YardLine

train.loc[train.FieldPosition.fillna('') == train.PossessionTeam,  

          'YardLine_std'

         ] = train.loc[train.FieldPosition.fillna('') == train.PossessionTeam,  

          'YardLine']

train['X_std'] = train.X

train.loc[train.ToLeft, 'X_std'] = 120 - train.loc[train.ToLeft, 'X'] 

train['Y_std'] = train.Y

train.loc[train.ToLeft, 'Y_std'] = 160/3 - train.loc[train.ToLeft, 'Y'] 

train['Dir_std'] = train['Dir_rad']

train.loc[train.ToLeft, 'Dir_std'] = np.mod(np.pi + train.loc[train.ToLeft, 'Dir_rad'], 2*np.pi)

train.loc[train.Season >= 2018, 'Orientation_rad'] = np.mod(train.loc[train.Season >= 2018, 'Orientation'] - 90, 360) * math.pi/180.0

train['Orientation_rad'] = np.mod(train.Orientation, 360) * math.pi/180.0

train.loc[train.Season >= 2018, 'Orientation_rad'

         ] = np.mod(train.loc[train.Season >= 2018, 'Orientation'] - 90, 360) * math.pi/180.0

train['Orientation_std'] = train.Orientation_rad

train.loc[train.ToLeft, 'Orientation_std'] = np.mod(np.pi + train.loc[train.ToLeft, 'Orientation_rad'], 2*np.pi)

train['WindSpeed'] = train['WindSpeed'].apply(lambda x: x.lower().replace('mph', '').strip() if not pd.isna(x) else x)

train['WindSpeed_n'] = train['WindSpeed'].apply(lambda x: x.lower().replace('mph', '').strip() if not pd.isna(x) else x)

train['WindSpeed_n'] = train['WindSpeed_n'].apply(lambda x: (-int(x.split('-')[0])+int(x.split('-')[1]))*np.random.random_sample()+int(x.split('-')[0]) if not pd.isna(x) and '-' in x else x)

train['WindSpeed_n'] = train['WindSpeed_n'].apply(lambda x: (-int(x.split()[-1])+int(x.split()[0]))*np.random.random_sample()+int(x.split()[-1]) if not pd.isna(x) and type(x)!=float and 'gusts up to' in x else x)

train['WindSpeed_n'] = train['WindSpeed_n'].apply(lambda x: switch_speed_direction(x))

train['WindSpeed_n']=train['WindSpeed_n'].apply(lambda x: strtoint(x))

train['WindDirection'] = train['WindDirection'].apply(lambda x: switch_direction_speed(x))

train['WindDirection'] = train['WindDirection'].replace(north,'north')

train['WindDirection'] = train['WindDirection'].replace(south,'south')

train['WindDirection'] = train['WindDirection'].replace(west,'west')

train['WindDirection'] = train['WindDirection'].replace(east,'east')

train['WindDirection'] = train['WindDirection'].replace(north_east,'north_east')

train['WindDirection'] = train['WindDirection'].replace(north_west,'north_west')

train['WindDirection'] = train['WindDirection'].replace(south_east,'clear')

train['WindDirection'] = train['WindDirection'].replace(south_west,'south_west')

train['WindDirection'] = train['WindDirection'].replace(no_wind,'no_wind')

train['Turf'] = train['Turf'].replace(natural_grass,'natural_grass')

train['Turf'] = train['Turf'].replace(grass,'grass')

train['Turf'] = train['Turf'].replace(fieldturf,'fieldturf')

train['Turf'] = train['Turf'].replace(artificial,'artificial')

train['PlayerHeight'] = train['PlayerHeight'].apply(lambda x: 12*int(x.split('-')[0])+int(x.split('-')[1]))

train['Team'] = train['Team'].apply(lambda x: x.strip()=='home')

train['GameWeather'] = train['GameWeather'].replace(rain,'rain')

train['GameWeather'] = train['GameWeather'].replace(overcast,'overcast')

train['GameWeather'] = train['GameWeather'].replace(clear,'clear')

train['GameWeather'] = train['GameWeather'].replace(snow,'snow')

train['GameWeather'] = train['GameWeather'].replace(none,'none')

train['GameClock_sec'] = train['GameClock'].apply(strtoseconds)

train["GameClock_minute"] = train["GameClock"].apply(lambda x : x.split(":")[0]).astype("object")

train['TimeHandoff'] = train['TimeHandoff'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))

train['TimeSnap'] = train['TimeSnap'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))

train['TimeDelta'] = train.apply(lambda row: (row['TimeHandoff'] - row['TimeSnap']).total_seconds(), axis=1)

train['PlayerBirthDate'] = train['PlayerBirthDate'].apply(lambda x: datetime.datetime.strptime(x, "%m/%d/%Y"))

seconds_in_year = 60*60*24*365.25

train['PlayerAge'] = train.apply(lambda row: (row['TimeHandoff']-row['PlayerBirthDate']).total_seconds()/seconds_in_year, axis=1)

train["PlayerAge_ob"] = train['PlayerAge'].astype(np.int).astype("object")

train['IsRusher'] = (train['NflId'] == train['NflIdRusher'])

train['IsRusher_ob'] = (train['NflId'] == train['NflIdRusher']).astype("object")

temp = train[train["IsRusher"]][["Team", "PlayId"]].rename(columns={"Team":"RusherTeam"})

train = train.merge(temp, on = "PlayId")

train["IsRusherTeam"] = train["Team"] == train["RusherTeam"]

train["Quarter"] = train["Quarter"].astype("object")

train["Down"] = train["Down"].astype("object")

train["JerseyNumber"] = train["JerseyNumber"].astype("object")

train["YardLine"] = train["YardLine"].astype("object")

train["diffScoreBeforePlay"] = train["HomeScoreBeforePlay"] - train["VisitorScoreBeforePlay"]

train["diffScoreBeforePlay_binary_ob"] = (train["HomeScoreBeforePlay"] > train["VisitorScoreBeforePlay"]).astype("object")

train['RetractableRoof'] = train['StadiumType'].apply(lambda x: x in retractable)

train['DomeRoof'] = train['StadiumType'].apply(lambda x: x in dome_closed+dome_open)

train['StadiumType'] = train['StadiumType'].replace(outdoor,'outdoor')

train['StadiumType'] = train['StadiumType'].replace(indoor_closed,'indoor_closed')

train['StadiumType'] = train['StadiumType'].replace(indoor_open,'indoor_open')

train['StadiumType'] = train['StadiumType'].replace(dome_closed,'dome_closed')

train['StadiumType'] = train['StadiumType'].replace(dome_open,'dome_open')

train['Outdoor'] = train['StadiumType'].apply(transform_StadiumType)

train['Position_Team'] = train['Position'] + '_' +train['Team'].astype(str)

train.loc[train.FieldPosition == train.PossessionTeam,'YardsFromOwnGoal'] = train.loc[train.FieldPosition == train.PossessionTeam,'YardLine']

train.loc[train.FieldPosition != train.PossessionTeam,'YardsFromOwnGoal'] = 50 - train.loc[train.FieldPosition != train.PossessionTeam,'YardLine']

train["OffenseInTheBox"] = train["OffensePersonnel"].apply(get_offense_personel)

train["MoreOffense"] = train["OffenseInTheBox"] > train["DefendersInTheBox"]

temp = train["OffensePersonnel"].iloc[np.arange(0, len(train), 22)].apply(lambda x : pd.Series(OffensePersonnelSplit(x)))

temp.columns = ["Offense" + c for c in temp.columns]

temp["PlayId"] = train["PlayId"].iloc[np.arange(0, len(train), 22)]

train = train.merge(temp, on = "PlayId")

temp = train["DefensePersonnel"].iloc[np.arange(0, len(train), 22)].apply(lambda x : pd.Series(DefensePersonnelSplit(x)))

temp.columns = ["Defense" + c for c in temp.columns]

temp["PlayId"] = train["PlayId"].iloc[np.arange(0, len(train), 22)]

train = train.merge(temp, on = "PlayId")

train = train.sort_values(by = ['X']).sort_values(by = ['Dis']).sort_values(by=['PlayId', 'IsRusherTeam', 'IsRusher']).reset_index(drop = True)

columns = list(train.columns)

columns.remove("Yards")

cat_columns = list(train.select_dtypes(object))

cat_columns.extend(list(train.select_dtypes(bool)))

dcols = ['WindSpeed','GameClock','Orientation','StadiumType','PlayerAge','DisplayName', 'JerseyNumber','GameId']

train.drop(columns = dcols, inplace = True)

cat_num_columns = ['GameClock_minute','PlayerAge_ob','YardsFromOwnGoal','GameClock_sec']

cat_columns = list(set(cat_columns) - set(cat_num_columns) - set(dcols))

train[cat_columns] = train[cat_columns].astype('str')

train[cat_columns].fillna('nan')

train[cat_num_columns] = train[cat_num_columns].astype('int')

train[cat_num_columns].fillna(method="ffill")



#print(train.info())
X_train = train[list(set(columns)-set(dcols))]

y_train = train["Yards"]

scaler = StandardScaler()

scaler.fit(y_train.values.reshape(-1, 1))

y_train = scaler.transform(y_train.values.reshape(-1, 1)).flatten()

label_encoder = LabelEncoder()

for i in cat_columns:

    X_train[i] = label_encoder.fit_transform(X_train[i].values)

cat_features_idx = np.where(X_train.columns.isin(cat_columns))[0]

print(X_train.info())


NFOLDS = 17

split_groups = train["PlayId"]

SEED = 47



folds = GroupKFold(n_splits=NFOLDS)



models, y_valid_pred = [], np.zeros(len(X_train))



cat_params = {'bagging_temperature': 0.11303262970608133,

 'border_count': 143,

 'depth': 8,

 'learning_rate': 0.0007,

 'iterations': 777,

 'l2_leaf_reg': 22.2,

 'learning_rate': 0.09718416557498283,

 'random_strength': 0.48627940822903243,

 'eval_metric':"MAE",

             }



for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train, groups=split_groups)):

    print('Fold:',fold_)    

    estimator = CatBoostRegressor(**cat_params)  

    x_val, y_val = X_train.iloc[val_idx, :], y_train[val_idx]    

    

    train_data = Pool(X_train.iloc[trn_idx, :], y_train[trn_idx])

    valid_data = Pool(x_val, y_val)

    

    regressor = estimator.fit(

            train_data, 

            early_stopping_rounds=70,

            eval_set=valid_data,

            use_best_model=True,

            plot = True)

    

    y_valid_pred[val_idx] += regressor.predict(x_val)

    models.append(regressor)
feature_importance = pd.DataFrame({'feature': X_train.columns, 'importance':estimator.get_feature_importance()}).sort_values('importance', ascending=False)[:50]

plt.figure(figsize=(14,25))

sns.barplot(x=feature_importance.importance, y=feature_importance.feature);
def prepare(train):

    outdoor = ['Outdoor', 'Outdoors', 'Cloudy', 'Heinz Field', 'Outdor', 'Ourdoor', 

               'Outside', 'Outddors','Outdoor Retr Roof-Open', 'Oudoor', 'Bowl']

    indoor_closed = ['Indoors', 'Indoor', 'Indoor, Roof Closed', 'Indoor, Roof Closed', 'Retractable Roof',

                     'Retr. Roof-Closed', 'Retr. Roof - Closed', 'Retr. Roof Closed']

    retractable = ['Outdoor Retr Roof-Open', 'Retractable Roof', 'Retr. Roof-Closed', 'Retr. Roof - Closed', 'Retr. Roof Closed',

                  'Retr. Roof-Open', 'Retr. Roof - Open']

    indoor_open   = ['Indoor, Open Roof', 'Open', 'Retr. Roof-Open', 'Retr. Roof - Open']

    dome_closed   = ['Dome', 'Domed, closed', 'Closed Dome', 'Domed', 'Dome, closed']

    dome_open     = ['Domed, Open', 'Domed, open']

    rain = ['Rainy', 'Rain Chance 40%', 'Showers','Cloudy with periods of rain, thunder possible. Winds shifting to WNW, 10-20 mph.',

            'Scattered Showers', 'Cloudy, Rain', 'Rain shower', 'Light Rain', 'Rain']



    overcast = ['Cloudy, light snow accumulating 1-3"', 'Party Cloudy', 'Cloudy, chance of rain',

                'Coudy', 'Cloudy, 50% change of rain', 'Rain likely, temps in low 40s.',

                'Cloudy and cold', 'Cloudy, fog started developing in 2nd quarter',

                'Partly Clouidy', '30% Chance of Rain', 'Mostly Coudy', 'Cloudy and Cool',

                'cloudy', 'Partly cloudy', 'Overcast', 'Hazy', 'Mostly cloudy', 'Mostly Cloudy',

                'Partly Cloudy', 'Cloudy']



    clear = ['Partly clear', 'Sunny and clear', 'Sun & clouds', 'Clear and Sunny',

            'Sunny and cold', 'Sunny Skies', 'Clear and Cool', 'Clear and sunny',

            'Sunny, highs to upper 80s', 'Mostly Sunny Skies', 'Cold',

            'Clear and warm', 'Sunny and warm', 'Clear and cold', 'Mostly sunny',

            'T: 51; H: 55; W: NW 10 mph', 'Clear Skies', 'Clear skies', 'Partly sunny',

            'Fair', 'Partly Sunny', 'Mostly Sunny', 'Clear', 'Sunny', 'Sunny, Windy']



    snow  = ['Heavy lake effect snow', 'Snow']



    none  = ['N/A Indoor', 'Indoors', 'Indoor', 'N/A (Indoors)', 'Controlled Climate']



    north = ['N','From S','North']

    south = ['S','From N','South','s']

    west = ['W','From E','West']

    east = ['E','From W','from W','EAST','East']

    north_east = ['FROM SW','FROM SSW','FROM WSW','NE','NORTH EAST','North East','East North East','NorthEast','Northeast','ENE','From WSW','From SW']

    north_west = ['E','From ESE','NW','NORTHWEST','N-NE','NNE','North/Northwest','W-NW','WNW','West Northwest','Northwest','NNW','From SSE']

    south_east = ['E','From WNW','SE','SOUTHEAST','South Southeast','East Southeast','Southeast','SSE','From SSW','ESE','From NNW']

    south_west = ['E','From ENE','SW','SOUTHWEST','W-SW','South Southwest','West-Southwest','WSW','SouthWest','Southwest','SSW','From NNE']

    no_wind = ['clear','Calm']

    natural_grass = ['natural grass','Naturall Grass','Natural Grass']

    grass = ['Grass']

    fieldturf = ['FieldTurf','Field turf','FieldTurf360','Field Turf']

    artificial = ['Artificial','Artifical']

    

    train['ToLeft'] = train.PlayDirection == "left"

    train['IsBallCarrier'] = train.NflId == train.NflIdRusher

    train['HomeTeamAbbr'] = train['HomeTeamAbbr'].replace({'CLE':'CLV', 'ARI':'ARZ', 'HOU':'HST', 'BAL':'BLT'})

    train['Dir_rad'] = np.mod(90 - train.Dir, 360) * math.pi/180.0

    train['TeamOnOffense'] = "home"

    train.loc[train.PossessionTeam != train.HomeTeamAbbr, 'TeamOnOffense'] = "away"

    train['IsOnOffense'] = train.Team == train.TeamOnOffense

    train['YardLine_std'] = 100 - train.YardLine

    train.loc[train.FieldPosition.fillna('') == train.PossessionTeam,  

              'YardLine_std'

             ] = train.loc[train.FieldPosition.fillna('') == train.PossessionTeam,  

              'YardLine']

    train['X_std'] = train.X

    train.loc[train.ToLeft, 'X_std'] = 120 - train.loc[train.ToLeft, 'X'] 

    train['Y_std'] = train.Y

    train.loc[train.ToLeft, 'Y_std'] = 160/3 - train.loc[train.ToLeft, 'Y'] 

    train['Dir_std'] = train['Dir_rad']

    train.loc[train.ToLeft, 'Dir_std'] = np.mod(np.pi + train.loc[train.ToLeft, 'Dir_rad'], 2*np.pi)

    train.loc[train.Season >= 2018, 'Orientation_rad'] = np.mod(train.loc[train.Season >= 2018, 'Orientation'] - 90, 360) * math.pi/180.0

    train['Orientation_rad'] = np.mod(train.Orientation, 360) * math.pi/180.0

    train.loc[train.Season >= 2018, 'Orientation_rad'

             ] = np.mod(train.loc[train.Season >= 2018, 'Orientation'] - 90, 360) * math.pi/180.0

    train['Orientation_std'] = train.Orientation_rad

    train.loc[train.ToLeft, 'Orientation_std'] = np.mod(np.pi + train.loc[train.ToLeft, 'Orientation_rad'], 2*np.pi)

    train['WindSpeed'] = train['WindSpeed'].apply(lambda x: x.lower().replace('mph', '').strip() if not pd.isna(x) else x)

    train['WindSpeed_n'] = train['WindSpeed'].apply(lambda x: x.lower().replace('mph', '').strip() if not pd.isna(x) else x)

    train['WindSpeed_n'] = train['WindSpeed_n'].apply(lambda x: (-int(x.split('-')[0])+int(x.split('-')[1]))*np.random.random_sample()+int(x.split('-')[0]) if not pd.isna(x) and '-' in x else x)

    train['WindSpeed_n'] = train['WindSpeed_n'].apply(lambda x: (-int(x.split()[-1])+int(x.split()[0]))*np.random.random_sample()+int(x.split()[-1]) if not pd.isna(x) and type(x)!=float and 'gusts up to' in x else x)

    #train['WindSpeed_n'] = train['WindSpeed_n'].apply(lambda x: switch_speed_direction(x))

    train['WindSpeed_n']=train['WindSpeed_n'].apply(lambda x: strtoint(x))

    #train['WindDirection'] = train['WindDirection'].apply(lambda x: switch_direction_speed(x))

    train['WindDirection'] = train['WindDirection'].replace(north,'north')

    train['WindDirection'] = train['WindDirection'].replace(south,'south')

    train['WindDirection'] = train['WindDirection'].replace(west,'west')

    train['WindDirection'] = train['WindDirection'].replace(east,'east')

    train['WindDirection'] = train['WindDirection'].replace(north_east,'north_east')

    train['WindDirection'] = train['WindDirection'].replace(north_west,'north_west')

    train['WindDirection'] = train['WindDirection'].replace(south_east,'clear')

    train['WindDirection'] = train['WindDirection'].replace(south_west,'south_west')

    train['WindDirection'] = train['WindDirection'].replace(no_wind,'no_wind')

    train['Turf'] = train['Turf'].replace(natural_grass,'natural_grass')

    train['Turf'] = train['Turf'].replace(grass,'grass')

    train['Turf'] = train['Turf'].replace(fieldturf,'fieldturf')

    train['Turf'] = train['Turf'].replace(artificial,'artificial')

    train['PlayerHeight'] = train['PlayerHeight'].apply(lambda x: 12*int(x.split('-')[0])+int(x.split('-')[1]))

    train['Team'] = train['Team'].apply(lambda x: x.strip()=='home')

    train['GameWeather'] = train['GameWeather'].replace(rain,'rain')

    train['GameWeather'] = train['GameWeather'].replace(overcast,'overcast')

    train['GameWeather'] = train['GameWeather'].replace(clear,'clear')

    train['GameWeather'] = train['GameWeather'].replace(snow,'snow')

    train['GameWeather'] = train['GameWeather'].replace(none,'none')

    train['GameClock_sec'] = train['GameClock'].apply(strtoseconds)

    train["GameClock_minute"] = train["GameClock"].apply(lambda x : x.split(":")[0]).astype("object")

    train['TimeHandoff'] = train['TimeHandoff'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))

    train['TimeSnap'] = train['TimeSnap'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))

    train['TimeDelta'] = train.apply(lambda row: (row['TimeHandoff'] - row['TimeSnap']).total_seconds(), axis=1)

    train['PlayerBirthDate'] = train['PlayerBirthDate'].apply(lambda x: datetime.datetime.strptime(x, "%m/%d/%Y"))

    seconds_in_year = 60*60*24*365.25

    train['PlayerAge'] = train.apply(lambda row: (row['TimeHandoff']-row['PlayerBirthDate']).total_seconds()/seconds_in_year, axis=1)

    train["PlayerAge_ob"] = train['PlayerAge'].astype(np.int).astype("object")

    train['IsRusher'] = (train['NflId'] == train['NflIdRusher'])

    train['IsRusher_ob'] = (train['NflId'] == train['NflIdRusher']).astype("object")

    temp = train[train["IsRusher"]][["Team", "PlayId"]].rename(columns={"Team":"RusherTeam"})

    train = train.merge(temp, on = "PlayId")

    train["IsRusherTeam"] = train["Team"] == train["RusherTeam"]

    train["Quarter"] = train["Quarter"].astype("object")

    train["Down"] = train["Down"].astype("object")

    train["JerseyNumber"] = train["JerseyNumber"].astype("object")

    train["YardLine"] = train["YardLine"].astype("object")

    train["diffScoreBeforePlay"] = train["HomeScoreBeforePlay"] - train["VisitorScoreBeforePlay"]

    train["diffScoreBeforePlay_binary_ob"] = (train["HomeScoreBeforePlay"] > train["VisitorScoreBeforePlay"]).astype("object")

    train['RetractableRoof'] = train['StadiumType'].apply(lambda x: x in retractable)

    train['DomeRoof'] = train['StadiumType'].apply(lambda x: x in dome_closed+dome_open)

    train['StadiumType'] = train['StadiumType'].replace(outdoor,'outdoor')

    train['StadiumType'] = train['StadiumType'].replace(indoor_closed,'indoor_closed')

    train['StadiumType'] = train['StadiumType'].replace(indoor_open,'indoor_open')

    train['StadiumType'] = train['StadiumType'].replace(dome_closed,'dome_closed')

    train['StadiumType'] = train['StadiumType'].replace(dome_open,'dome_open')

    train['Outdoor'] = train['StadiumType'].apply(transform_StadiumType)

    train['Position_Team'] = train['Position'] + '_' +train['Team'].astype(str)

    train.loc[train.FieldPosition == train.PossessionTeam,'YardsFromOwnGoal'] = train.loc[train.FieldPosition == train.PossessionTeam,'YardLine']

    train.loc[train.FieldPosition != train.PossessionTeam,'YardsFromOwnGoal'] = 50 - train.loc[train.FieldPosition != train.PossessionTeam,'YardLine']

    train["OffenseInTheBox"] = train["OffensePersonnel"].apply(get_offense_personel)

    train["MoreOffense"] = train["OffenseInTheBox"] > train["DefendersInTheBox"]

    temp = train["OffensePersonnel"].iloc[np.arange(0, len(train), 22)].apply(lambda x : pd.Series(OffensePersonnelSplit(x)))

    temp.columns = ["Offense" + c for c in temp.columns]

    temp["PlayId"] = train["PlayId"].iloc[np.arange(0, len(train), 22)]

    train = train.merge(temp, on = "PlayId")

    temp = train["DefensePersonnel"].iloc[np.arange(0, len(train), 22)].apply(lambda x : pd.Series(DefensePersonnelSplit(x)))

    temp.columns = ["Defense" + c for c in temp.columns]

    temp["PlayId"] = train["PlayId"].iloc[np.arange(0, len(train), 22)]

    train = train.merge(temp, on = "PlayId")

    train = train.sort_values(by = ['X']).sort_values(by = ['Dis']).sort_values(by=['PlayId', 'IsRusherTeam', 'IsRusher']).reset_index(drop = True)

    

    #print(train.info())

    return train

pd.options.mode.chained_assignment = None

index = 21





for (test, sample_prediction) in tqdm.tqdm(env.iter_test()):

    test = prepare(test)

    

    columns = list(test.columns)

    cat_columns = list(test.select_dtypes(object))

    cat_columns.extend(list(test.select_dtypes(bool)))

    dcols = ['WindSpeed','GameClock','Orientation','StadiumType','PlayerAge','DisplayName', 'JerseyNumber','GameId']

    

    cat_num_columns = ['GameClock_minute','PlayerAge_ob','YardsFromOwnGoal','GameClock_sec']

    cat_columns = list(set(cat_columns)-set(cat_num_columns) - set(dcols))

    test[cat_columns] = test[cat_columns].astype('str')

    test[cat_columns].fillna('nan')

    test[cat_num_columns] = test[cat_num_columns].astype('int')

    test[cat_num_columns].fillna(method="ffill")

    float_nums = ['DefendersInTheBox','Humidity']

    test[float_nums] = test[float_nums].astype('float')

    features = list(set(columns) - set(dcols))

    test = test[features]

    label_encoder = LabelEncoder()

    for i in cat_columns:

        test[i] = label_encoder.fit_transform(test[i].values)

    #cat_features_idx = np.where(test.columns.isin(cat_columns))[0]

    

    

    y_pred = np.zeros(199)        

    y_pred_p = np.sum(np.round(scaler.inverse_transform([model.predict(test)[0] for model in models])))/NFOLDS

    y_pred_p += 99

    for j in range(199):

        if j>=y_pred_p+10:

            y_pred[j]=1.0

        elif j>=y_pred_p-10:

            y_pred[j]=(j+10-y_pred_p)*0.05

    env.predict(pd.DataFrame(data=[y_pred],columns=sample_prediction.columns))    

env.write_submission_file()