from kaggle.competitions import nflrush

import numpy as np

import pandas as pd



env = nflrush.make_env()

dev_df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv')

dev_df.shape
from sklearn.metrics.pairwise import euclidean_distances





def get_time(x):

    x = x.split(":")

    return int(x[0])*60 + int(x[1])



def get_height(x):

    x = x.split("-")

    return int(x[0])*12 + int(x[1])





def process_windspeed(txt):

    txt = str(txt).lower().replace('mph', '').strip()

    if '-' in txt:

        txt = (int(txt.split('-')[0]) + int(txt.split('-')[1])) / 2

    try:

        return float(txt)

    except:

        return -1.0



def new_X(x_coordinate, play_direction):

    if play_direction == 'left':

        return 120.0 - x_coordinate

    else:

        return x_coordinate



def new_line(rush_team, field_position, yardline):

    if rush_team == field_position:

        # offense starting at X = 0 plus the 10 yard endzone plus the line of scrimmage

        return 10.0 + yardline

    else:

        # half the field plus the yards between midfield and the line of scrimmage

        return 60.0 + (50 - yardline)



def new_orientation(angle, play_direction):

    if play_direction == 'left':

        new_angle = 360.0 - angle

        if new_angle == 360.0:

            new_angle = 0.0

        return new_angle

    else:

        return angle



def update_yardline(df):

    new_yardline = df[df['NflId'] == df['NflIdRusher']]

    new_yardline['YardLine'] = new_yardline[['PossessionTeam','FieldPosition','YardLine']].apply(lambda x: new_line(x[0],x[1],x[2]), axis=1)

    new_yardline = new_yardline[['GameId','PlayId','YardLine']]



    return new_yardline



def update_orientation(df, yardline):

    df['X'] = df[['X','PlayDirection']].apply(lambda x: new_X(x[0],x[1]), axis=1)

    df['Orientation'] = df[['Orientation','PlayDirection']].apply(lambda x: new_orientation(x[0],x[1]), axis=1)

    df['Dir'] = df[['Dir','PlayDirection']].apply(lambda x: new_orientation(x[0],x[1]), axis=1)



    df = df.drop('YardLine', axis=1)

    df = pd.merge(df, yardline, on=['GameId','PlayId'], how='inner')

    

    return df



def process_df(df):

    df["is_rusher"] = 1.0*(df["NflId"] == df["NflIdRusher"])

    df["is_home"] = df["Team"] == "home"

    df["is_possession_team"] = 1.0*(df["PossessionTeam"] == df["HomeTeamAbbr"]) - 1.0*(df["PossessionTeam"] == df["VisitorTeamAbbr"])

    df["is_field_team"] = 1.0*(df["FieldPosition"] == df["HomeTeamAbbr"]) - 1.0*(df["FieldPosition"] == df["VisitorTeamAbbr"])

    df["is_left"] = df["PlayDirection"] == "left"

    df["player_height"] = df["PlayerHeight"].apply(get_height)



    df["WindSpeed"] = df["WindSpeed"].apply(process_windspeed)

    df["TimeHandoff"] = pd.to_datetime(df["TimeHandoff"])

    df["TimeSnap"] = pd.to_datetime(df["TimeSnap"])

    df["duration"] = (df["TimeHandoff"] - df["TimeSnap"]).dt.total_seconds()

    

    df["player_age"] = (df["TimeSnap"].dt.date - pd.to_datetime(df["PlayerBirthDate"]).dt.date)/np.timedelta64(1, 'D') / 365

    

    df["game_time"] = df["GameClock"].apply(get_time)

    df["old_data"] = df["Season"] == 2017

    

    yardline = update_yardline(df)

    df = update_orientation(df, yardline)

    

    return df.fillna(-1)



dev_df = process_df(dev_df)
game_cols = []

play_cols = []

player_cols = []



for c in dev_df.columns:

    if dev_df.groupby("GameId")[c].nunique().max() == 1:

        game_cols.append(c)

    elif dev_df.groupby("PlayId")[c].nunique().max() == 1:

        play_cols.append(c)

    else:

        player_cols.append(c)



print("Game cols:", game_cols)

print("Play cols:", play_cols)

print("Player cols:", player_cols)
dev_df = dev_df[dev_df["is_rusher"] == 1]

dev_df.shape
from sklearn.preprocessing import LabelEncoder



le_dict = {}

categoricals = ["Team_le", "NflIdRusher_le", "DisplayName_le", "PlayerCollegeName_le", "Position_le",

                'OffenseFormation_le', 'OffensePersonnel_le', 'DefensePersonnel_le', 'PlayDirection_le',

                'HomeTeamAbbr_le', 'VisitorTeamAbbr_le', 'Stadium_le', 'Location_le', 'StadiumType_le', 'Turf_le', 'GameWeather_le', "WindDirection_le"]



for cat in categoricals:

    le_dict[cat] = LabelEncoder()

    dev_df[cat] = le_dict[cat].fit_transform(dev_df[cat[:-3]].apply(str))
!pip install lofo-importance
from sklearn.model_selection import KFold

from lofo import LOFOImportance, Dataset, plot_importance

import lightgbm as lgb



NUM_FOLDS = 4



#time split

dev_df.sort_values("TimeSnap", inplace=True)

kfold = KFold(NUM_FOLDS, shuffle=False, random_state=0)



features = ['X', 'Y', 'S', 'A', 'Dis', 'Orientation', 'Dir', 'JerseyNumber', 'PlayerWeight', 'is_rusher', 'is_home', 'player_height', 'player_age',

           'YardLine', 'Quarter', 'Down', 'Distance', 'HomeScoreBeforePlay', 'VisitorScoreBeforePlay', 'DefendersInTheBox', 'is_possession_team', 'is_field_team', 'is_left', 'duration', 'game_time',

           'Temperature', 'Humidity', 'WindSpeed', 'old_data'] + categoricals



params = {'num_leaves': 15,

          'objective': 'mae',

          'learning_rate': 0.1,

          "boosting": "gbdt",

          "num_rounds": 100

          }



model = lgb.LGBMRegressor(**params)



dataset = Dataset(df=dev_df, target="Yards", features=features)



lofo_imp = LOFOImportance(dataset, model=model, cv=kfold, scoring="neg_mean_absolute_error", fit_params={"categorical_feature": categoricals})



importance_df = lofo_imp.get_importance()
plot_importance(importance_df, figsize=(12, 18))