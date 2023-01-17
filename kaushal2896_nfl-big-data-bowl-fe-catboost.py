# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from kaggle.competitions import nflrush

from string import punctuation

from tqdm import tqdm

import gc, re

import pickle



from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import KFold

import lightgbm as lgb

from catboost import Pool, CatBoostRegressor

from sklearn.preprocessing import StandardScaler



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
%%time

train_df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)
print (f'Shape of training dataset: {train_df.shape}')
train_df.head()
train_df.columns
train_df.Turf.unique()
## Function to reduce the memory usage

def reduce_mem_usage(df, verbose=True):

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

    if verbose: print(f'Mem. usage decreased to {end_mem} Mb ({100 * (start_mem - end_mem) / start_mem}% reduction)')

    return df
def get_player_specific_cols(col_names):

    cols, total_players = [], 22

    for col in col_names:

        for player in range(total_players):

            cols.append(f'{col}_player{player}')

    return cols
def mean_without_overflow_fast(col):

    col /= len(col)

    return col.mean() * len(col)
def encode_cyclic_feature(df, col, max_vals):

    df[col + '_sin'] = np.sin(2 * np.pi * df[col]/max_vals)

    df[col + '_cos'] = np.cos(2 * np.pi * df[col]/max_vals)

    del df[col]

    return df
def extract_timestamp(df, timestamp_col):

    df[f'{timestamp_col}Hour'] = np.uint8(df[timestamp_col].dt.hour)

    df[f'{timestamp_col}Minute'] = np.uint8(df[timestamp_col].dt.minute)

    df[f'{timestamp_col}Second'] = np.uint8(df[timestamp_col].dt.second)

    return df
def get_player_specific_cols(col_names):

    cols, total_players = [], 22

    for col in col_names:

        for player in range(total_players):

            cols.append(f'{col}_player{player}')

    return cols
def height_to_inches(player_height):

    return int(player_height.split('-')[0]) * 12 + int(player_height.split('-')[1])
def bdate_to_age(bdate):

    now = pd.to_datetime('now')

    return (now.year - bdate.dt.year) - ((now.month - bdate.dt.month) < 0)
def get_grouping_dict(df, key):

    dicts = []

    for _, row in df.iterrows():

        dicts.append(dict([(pos.split()[1], pos.split()[0]) for (pos) in row[key].split(',')]))

    return dicts
def groupby_playid(df, is_training=True):

    

    total_players = 22

    non_player_features = ['GameId', 'PlayId', 'Season', 'YardLine', 'Quarter', 'GameClock',

       'PossessionTeam', 'Down', 'Distance', 'FieldPosition',

       'HomeScoreBeforePlay', 'VisitorScoreBeforePlay',

       'OffenseFormation', 'OffensePersonnel', 'DefendersInTheBox',

       'DefensePersonnel', 'PlayDirection', 'TimeHandoff', 'TimeSnap',

       'Yards', 'HomeTeamAbbr', 'VisitorTeamAbbr', 'Week', 'Stadium',

       'Location', 'StadiumType', 'Turf', 'GameWeather', 'Temperature',

       'Humidity', 'WindSpeed', 'WindDirection', 'NflId']

    

    if not is_training:

        non_player_features.remove('Yards')

    

    player_features = ['Team', 'X', 'Y', 'S', 'A', 'Dis', 'Orientation', 'Dir',

       'DisplayName', 'JerseyNumber', 'PlayerHeight', 'PlayerWeight',

       'PlayerBirthDate', 'PlayerCollegeName', 'Position', 'NflIdRusher']

    

    playids_groups = df.groupby('PlayId').size().keys()

    

    player_features_columns = []

    for feature in player_features:

        for player in range(total_players):

            player_features_columns.append(f'{feature}_player{player}')

    

    # first assign non_player features which are common for a single game playid

    final_df = pd.DataFrame()

    final_df[non_player_features] = df.groupby('PlayId')[non_player_features].first().reset_index(drop=True)

    final_df = final_df.reindex(final_df.columns.tolist() + player_features_columns, axis=1)

    temp_cols = []

    if is_training:

        for group in tqdm(playids_groups, position=0, leave=True):

            temp_cols.append(df[df['PlayId'] == group][player_features].melt()['value'])

    else:

        for group in playids_groups:

            temp_cols.append(df[df['PlayId'] == group][player_features].melt()['value'])

    final_df[player_features_columns] = pd.DataFrame(temp_cols).values

    

    return final_df
def feature_engineering(df, is_training=True, label_encoders={}):

    

    if is_training:

        label_encoders['NflId'] = LabelEncoder()

        label_encoders['NflId'].fit(df['NflId'])

    try:

        df['NflId'] = label_encoders['NflId'].transform(df['NflId'])

    except:

        df['NflId'] = np.nan

       

    team_dict = {

        'away': 0,

        'home': 1

    }

    df['Team'] = df['Team'].map(team_dict)

    season_dict = {

        2017: 0,

        2018: 1

    }

    df['Season'] = df['Season'].map(season_dict)

    df = groupby_playid(df, is_training)

    

    df = df.drop(['Season', 'Temperature', 'Humidity'], axis = 1)

    

    if is_training:

        df = df.apply(lambda group: group.interpolate(limit_direction='both'))

    

    df['WindDirection'] = df['WindDirection'].fillna(method='backfill')

    df['WindSpeed'] = df['WindSpeed'].fillna(method='backfill')

    df['GameWeather'] = df['GameWeather'].fillna(method='backfill')

    df['StadiumType'] = df['StadiumType'].fillna(method='backfill')

    df['FieldPosition'] = df['FieldPosition'].fillna(method='backfill')

    df['OffenseFormation'] = df['OffenseFormation'].fillna(method='backfill')

    

    df['GameClock'] = pd.to_datetime(df['GameClock'])

    df['TimeHandoff'] = pd.to_datetime(df['TimeHandoff'])

    df['TimeSnap'] = pd.to_datetime(df['TimeSnap'])

    

    df = extract_timestamp(df, 'GameClock')

    df = extract_timestamp(df, 'TimeHandoff')

    df = extract_timestamp(df, 'TimeSnap')

    df = df.drop(['GameClock', 'TimeHandoff', 'TimeSnap'], axis=1)

    

    df = encode_cyclic_feature(df, 'GameClockHour', 24)

    df = encode_cyclic_feature(df, 'GameClockMinute', 60)

    df = encode_cyclic_feature(df, 'GameClockSecond', 60)

    

    df = encode_cyclic_feature(df, 'TimeHandoffHour', 24)

    df = encode_cyclic_feature(df, 'TimeHandoffMinute', 60)

    df = encode_cyclic_feature(df, 'TimeHandoffSecond', 60)

    

    df = encode_cyclic_feature(df, 'TimeSnapHour', 24)

    df = encode_cyclic_feature(df, 'TimeSnapMinute', 60)

    df = encode_cyclic_feature(df, 'TimeSnapSecond', 60)

    

    def transform_game_weather(x):

        x = str(x).lower()

        if 'indoor' in x:

            return  'indoor'

        elif 'cloud' in x or 'coudy' in x or 'clouidy' in x:

            return 'cloudy'

        elif 'rain' in x or 'shower' in x:

            return 'rain'

        elif 'sunny' in x:

            return 'sunny'

        elif 'clear' in x:

            return 'clear'

        elif 'cold' in x or 'cool' in x:

            return 'cool'

        elif 'snow' in x:

            return 'snow'

        return x

    

    df['GameWeather'] = df['GameWeather'].apply(lambda row: transform_game_weather(row))

    

    categorical_features = ['PossessionTeam', 'FieldPosition', 'OffenseFormation', 'PlayDirection', 'HomeTeamAbbr', 

                        'VisitorTeamAbbr', 'NflId','Stadium', 'Location', 'GameWeather'] + get_player_specific_cols(['Position', 'PlayerCollegeName', 'NflIdRusher'])

    

    for col in get_player_specific_cols(['PlayerHeight']):

        df[col] = df[col].apply(lambda x: height_to_inches(x))

    

    for col in get_player_specific_cols(['PlayerBirthDate']):

        df[col] = pd.to_datetime(df[col])

        df[col] = bdate_to_age(df[col])

    

    for cat in categorical_features:

        if is_training:

            label_encoders[cat] = LabelEncoder()

            label_encoders[cat].fit(df[cat])

        try:

            df[cat] = label_encoders[cat].transform(df[cat])

        except Exception as e:

            df[cat] = np.nan # Put NaN in case when any unseen label is found in testing dataset.

            

        

#     offense_groups = ['QB', 'RB', 'OL', 'FB', 'WR', 'TE']

#     defense_groups = ['DL', 'LB', 'CB', 'S']

    

#     offense_dicts = get_grouping_dict(df, 'OffensePersonnel')

#     defense_dicts = get_grouping_dict(df, 'DefensePersonnel')

    

#     offense_grps_df = pd.DataFrame(offense_dicts).rename(columns={'OL': 'OL_offense', 'DL': 'DL_offense', 'LB': 'LB_offense', 'DB': 'DB_offense'}).fillna(0).astype(int)

#     defense_grps_df = pd.DataFrame(defense_dicts).rename(columns={'OL': 'OL_defense', 'DL': 'DL_defense', 'LB': 'LB_defense', 'DB': 'DB_defense'}).fillna(0).astype(int)

    

#     df = pd.concat([df, offense_grps_df, defense_grps_df], axis=1)

    df = df.drop(['OffensePersonnel', 'DefensePersonnel'], axis=1)

    

    try:

        df['NflIdRusher'] = label_encoders['NflId'].transform(df['NflIdRusher'])

    except:

        df['NflIdRusher'] = np.nan

        

    wind_directions = ['N', 'E', 'S', 'W', 'NE', 'SE', 'SW', 'NW', 'NNE', 'ENE', 'ESE', 'SSE', 'SSW', 'WSW', 'WNW', 'NNW']  # https://www.quora.com/What-is-the-definition-of-SSW-wind-direction

    

    df.loc[df['WindSpeed'].isin(wind_directions), 'WindSpeed'] = np.nan

    df.loc[~df['WindDirection'].isin(wind_directions), 'WindDirection'] = np.nan

    

    df['WindDirection'] = df['WindDirection'].fillna(method='backfill')

    df['WindSpeed'] = df['WindSpeed'].fillna(method='backfill')

    

    if is_training:

        label_encoders['WindDirection'] = LabelEncoder()

        label_encoders['WindDirection'].fit(df['WindDirection'])

    try:

        df['WindDirection'] = label_encoders['WindDirection'].transform(df['WindDirection'])

    except Exception as e:

        df['WindDirection'] = np.nan

    

    def transform_windspeed(speed):

        speed = str(speed)

        if 'MPH' in speed or 'mph' in speed or 'MPh' in speed:

            speed = speed.replace('MPH', '').strip()

            speed = speed.replace('MPH', '').strip()

            speed = speed.replace('MPh', '').strip()

        if '-' in speed:

            return (float(speed.split('-')[0]) + float(speed.split('-')[1]))/2

        try:

            return float(speed)

        except:

            return 10 # https://sciencing.com/average-daily-wind-speed-24011.html

        

    df['WindSpeed'] = df['WindSpeed'].apply(lambda speed: transform_windspeed(speed))

    

    beaufort = [(0, 0, 0.3), (1, 0.3, 1.6), (2, 1.6, 3.4), (3, 3.4, 5.5), (4, 5.5, 8), 

                (5, 8, 10.8), (6, 10.8, 13.9), (7, 13.9, 17.2), (8, 17.2, 20.8), 

                (9, 20.8, 24.5), (10, 24.5, 28.5), (11, 28.5, 33), (12, 33, 200)]



    for item in beaufort:

        df.loc[(df['WindSpeed']>=item[1]) & (df['WindSpeed']<item[2]), 'beaufort_scale'] = item[0]

    

    df['DefendersInTheBox_vs_Distance'] = df['DefendersInTheBox'] / df['Distance']

    df['Field_eq_Possession'] = df['FieldPosition'] == df['PossessionTeam']

    

    # Add BMI as a feature: formula for BMI: kg/m^2

    total_players = 22

    

    def get_bmi(height, weight):

        return weight / (height ** 2) * 755

    

    def is_rusher(x, y):

        return x == y

    

    for player in range(total_players):

        df[f'BMI_player{player}'] = np.vectorize(get_bmi)(df[f'PlayerHeight_player{player}'], df[f'PlayerWeight_player{player}'])

        df[f'is_rusher_player{player}'] = np.vectorize(is_rusher)(df['NflId'], df[f'NflIdRusher_player{player}'])



    # Cleaning the Turf to Natural and artificial

    # from https://www.kaggle.com/c/nfl-big-data-bowl-2020/discussion/112681#latest-649087

    Turf = {'Field Turf':'Artificial', 'A-Turf Titan':'Artificial', 'Grass':'Natural', 'UBU Sports Speed S5-M':'Artificial', 

            'Artificial':'Artificial', 'DD GrassMaster':'Artificial', 'Natural Grass':'Natural', 

            'UBU Speed Series-S5-M':'Artificial', 'FieldTurf':'Artificial', 'FieldTurf 360':'Artificial', 'Natural grass':'Natural', 'grass':'Natural', 

            'Natural':'Natural', 'Artifical':'Artificial', 'FieldTurf360':'Artificial', 'Naturall Grass':'Natural', 'Field turf':'Artificial', 

            'SISGrass':'Artificial', 'Twenty-Four/Seven Turf':'Artificial', 'natural grass':'Natural'} 



    df['Turf'] = df['Turf'].map(Turf)

    df['Turf'] = df['Turf'] == 'Natural'

    

    def clean_StadiumType(txt):

        if pd.isna(txt):

            return np.nan

        txt = txt.lower()

        txt = ''.join([c for c in txt if c not in punctuation])

        txt = re.sub(' +', ' ', txt)

        txt = txt.strip()

        txt = txt.replace('outside', 'outdoor')

        txt = txt.replace('outdor', 'outdoor')

        txt = txt.replace('outddors', 'outdoor')

        txt = txt.replace('outdoors', 'outdoor')

        txt = txt.replace('oudoor', 'outdoor')

        txt = txt.replace('indoors', 'indoor')

        txt = txt.replace('ourdoor', 'outdoor')

        txt = txt.replace('retractable', 'rtr.')

        return txt

        

    df['StadiumType'] = df['StadiumType'].apply(clean_StadiumType)

    

    def transform_StadiumType(txt):

        if pd.isna(txt):

            return np.nan

        if 'outdoor' in txt or 'open' in txt:

            return 1

        if 'indoor' in txt or 'closed' in txt:

            return 0



        return np.nan

    

    df['StadiumType'] = df['StadiumType'].apply(transform_StadiumType)

    

    # from https://www.kaggle.com/c/nfl-big-data-bowl-2020/discussion/112173#latest-647309

#     df['JerseyNumberGrouped'] = df['JerseyNumber'] // 10

    

    if is_training:

        return df, label_encoders

    return df
%%time

train_df, label_encoders = feature_engineering(train_df)
train_df = reduce_mem_usage(train_df)

gc.collect()
non_feature_cols = ['GameId', 'PlayId'] + get_player_specific_cols(['DisplayName', 'JerseyNumber'])

target_col = ['Yards']
Y_train = train_df[target_col]
X_train = train_df.drop(non_feature_cols+target_col, axis=1)
scaler = StandardScaler()

scaler.fit(Y_train.values.reshape(-1, 1))

Y_train = scaler.transform(Y_train.values.reshape(-1, 1)).flatten()
# seed = 666

# n_folds = 10

# models, y_valid_pred = [], np.zeros(len(X_train))

# lgb_params={

#     'learning_rate': 0.01,

#     'objective': 'regression',

#     'n_estimators': 1000,

#     'num_leaves': 20,

#     'metric': 'rmse',

#     'bagging_fraction': 0.7,

#     'feature_fraction': 0.7

# }



# kf = KFold(n_splits = n_folds, shuffle=False, random_state=seed)



# for train_idx, val_idx in kf.split(X_train, Y_train):

#     x_train, y_train = X_train.iloc[train_idx, :], Y_train[train_idx]

#     x_val, y_val = X_train.iloc[val_idx, :], Y_train[val_idx]

    

#     training_data = lgb.Dataset(x_train, label=y_train)

#     val_data = lgb.Dataset(x_val, label=y_val)

    

#     regressor = lgb.LGBMRegressor(**lgb_params)

#     regressor.fit(x_train, y_train, eval_set=[(x_val, y_val)], early_stopping_rounds=100, verbose=100)

    

#     y_valid_pred[val_idx] += regressor.predict(x_val, num_iteration=regressor.best_iteration_)

#     models.append(regressor)
seed = 666

n_folds = 12

models, y_valid_pred = [], np.zeros(len(X_train))



kf = KFold(n_splits = n_folds, shuffle=False, random_state=seed)



for train_idx, val_idx in kf.split(X_train, Y_train):

    x_train, y_train = X_train.iloc[train_idx, :], Y_train[train_idx]

    x_val, y_val = X_train.iloc[val_idx, :], Y_train[val_idx]



    model = CatBoostRegressor(loss_function="RMSE",

                               eval_metric="RMSE",

                               task_type="CPU",

                               learning_rate=0.02,

                               iterations=2000,

                               l2_leaf_reg=5,

                               random_seed=42,

                               od_type="Iter",

                               depth=6,

                               early_stopping_rounds=150,

                               border_count=32

                              )



    train_data = Pool(x_train, y_train)

    valid_data = Pool(x_val, y_val)



    regressor = model.fit(train_data,

                        eval_set=valid_data,

                        use_best_model=True,

                        verbose=100)

    

    y_valid_pred[val_idx] += regressor.predict(x_val)

    models.append(regressor)
# Reference: https://www.kaggle.com/hukuda222/nfl-simple-model-using-lightgbm

y_pred = np.zeros((len(X_train),199))

y_ans = np.zeros((len(X_train),199))



for i,p in enumerate(np.round(scaler.inverse_transform(y_valid_pred))):

    p+=99

    for j in range(199):

        if j>=p+10:

            y_pred[i][j]=1.0

        elif j>=p-10:

            y_pred[i][j]=(j+10-p)*0.05



for i,p in enumerate(scaler.inverse_transform(Y_train)):

    p+=99

    for j in range(199):

        if j>=p:

            y_ans[i][j]=1.0



print("validation score:",np.sum(np.power(y_pred-y_ans,2))/(199*((len(X_train)))))
with open('models.pickle', 'wb') as handle:

    pickle.dump(models, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('label_encoders.pickle', 'wb') as handle:

    pickle.dump(label_encoders, handle, protocol=pickle.HIGHEST_PROTOCOL)
from IPython.display import FileLink, FileLinks

FileLink('models.pickle')
FileLink('label_encoders.pickle')
with open('models.pickle', 'rb') as handle:

    models = pickle.load(handle)
with open('label_encoders.pickle', 'rb') as handle:

    label_encoders = pickle.load(handle)
env = nflrush.make_env()
non_feature_cols = ['GameId', 'PlayId'] + get_player_specific_cols(['DisplayName', 'JerseyNumber'])

for (test_df, sample_prediction_df) in tqdm(env.iter_test(), position=0, leave=True):

    test_df = feature_engineering(test_df, False, label_encoders)

    test_df = test_df.drop(non_feature_cols, axis=1)

    y_pred = np.zeros(199)        

    y_pred_p = np.sum(np.round(scaler.inverse_transform([model.predict(test_df)[0] for model in models])))/n_folds

    y_pred_p += 99

    for j in range(199):

        if j>=y_pred_p+10:

            y_pred[j]=1.0

        elif j>=y_pred_p-10:

            y_pred[j]=(j+10-y_pred_p)*0.05

    env.predict(pd.DataFrame(data=[y_pred],columns=sample_prediction_df.columns))

env.write_submission_file()