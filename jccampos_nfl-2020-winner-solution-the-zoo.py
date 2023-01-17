import os

import math

import datetime

import re

from string import punctuation



from kaggle.competitions import nflrush

import tqdm

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



import pandas as pd

import numpy as np



from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

import lightgbm as lgb

import random



from sklearn.preprocessing import StandardScaler



sns.set_style('whitegrid')

sns.set_context('talk')



pd.set_option("display.max_columns", 100)
env = nflrush.make_env()

train = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv', dtype={'WindSpeed': 'object'})



# Original dataframe:

#train = pd.read_csv('train.csv', dtype={'WindSpeed': 'object'})

len(train)
def split_play_and_player_cols(df,predicting=False):

    df['IsRusher'] = df['NflId'] == df['NflIdRusher']

    

    play_ids = df["PlayId"].unique()

    #play_ids_filter = np.random.choice(play_ids,int(len(play_ids)*0.01),replace=False)

    #df = df.loc[df.PlayId.isin(play_ids_filter)]

    

    df['PlayId'] = df['PlayId'].astype(str)

    

    # We must assume here that the first 22 rows correspond to the same player:

    player_cols = [

        'PlayId', # This is the link between them

        'Season',

        'Team',

        'X',

        'Y',

        'S',

        'A',

        'Dis',

        'Dir',

        'NflId',

        'IsRusher',

    ]



    df_players = df[player_cols]

    

    play_cols = [

        'PlayId',

        'Season',

        'PossessionTeam',

        'HomeTeamAbbr',

        'VisitorTeamAbbr',

        'PlayDirection', 

        'FieldPosition',

        'YardLine',

    ]

    if not predicting:

        play_cols.append('Yards')

        

    df_play = df[play_cols].copy()



    ## Fillna in FieldPosition attribute

    #df['FieldPosition'] = df.groupby(['PlayId'], sort=False)['FieldPosition'].apply(lambda x: x.ffill().bfill())

    

    # Get first 

    df_play = df_play.groupby('PlayId').first().reset_index()



    #print('rows/plays in df: ', len(df_play))

    assert df_play.PlayId.nunique() == df.PlayId.nunique(), "Play/player split failed?"  # Boom

    

    return df_play, df_players
play_ids = train["PlayId"].unique()



df_play, df_players = split_play_and_player_cols(train)
def process_team_abbr(df):



    #These are only problems:

    map_abbr = {'ARI': 'ARZ', 'BAL': 'BLT', 'CLE': 'CLV', 'HOU': 'HST'}

    for abb in df['PossessionTeam'].unique():

        map_abbr[abb] = abb



    df['PossessionTeam'] = df['PossessionTeam'].map(map_abbr)

    df['HomeTeamAbbr'] = df['HomeTeamAbbr'].map(map_abbr)

    df['VisitorTeamAbbr'] = df['VisitorTeamAbbr'].map(map_abbr)



    df['HomePossession'] = df['PossessionTeam'] == df['HomeTeamAbbr']

    

    return



process_team_abbr(df_play)
def process_play_direction(df):

    df['IsPlayLeftToRight'] = df['PlayDirection'].apply(lambda val: True if val.strip() == 'right' else False)

    return

process_play_direction(df_play)
def process_yard_til_end_zone(df):

    def convert_to_yardline100(row):

        return (100 - row['YardLine']) if (row['PossessionTeam'] == row['FieldPosition']) else row['YardLine']

    df['Yardline100'] = df.apply(convert_to_yardline100, axis=1)

    return



process_yard_til_end_zone(df_play)
df_players = df_players.merge(

    df_play[['PlayId', 'PossessionTeam', 'HomeTeamAbbr', 'PlayDirection', 'Yardline100']], 

    how='left', on='PlayId')



df_players.loc[df_players.Season == 2017, 'S'] = 10*df_players.loc[df_players.Season == 2017,'Dis']
def standarize_direction(df):

    # adjusted the data to always be from left to right

    df['HomePossesion'] = df['PossessionTeam'] == df['HomeTeamAbbr']



    df['Dir_rad'] = np.mod(90 - df.Dir, 360) * math.pi/180.0



    df['ToLeft'] = df.PlayDirection == "left"

    df['TeamOnOffense'] = "home"

    df.loc[df.PossessionTeam != df.HomeTeamAbbr, 'TeamOnOffense'] = "away"

    df['IsOnOffense'] = df.Team == df.TeamOnOffense # Is player on offense?

    df['X_std'] = df.X

    df.loc[df.ToLeft, 'X_std'] = 120 - df.loc[df.ToLeft, 'X']

    df['Y_std'] = df.Y

    df.loc[df.ToLeft, 'Y_std'] = 160/3 - df.loc[df.ToLeft, 'Y']

    df['Dir_std'] = df.Dir_rad

    df.loc[df.ToLeft, 'Dir_std'] = np.mod(np.pi + df.loc[df.ToLeft, 'Dir_rad'], 2*np.pi)

   

    #Replace Null in Dir_rad

    df.loc[(df.IsOnOffense) & df['Dir_std'].isna(),'Dir_std'] = 0.0

    df.loc[~(df.IsOnOffense) & df['Dir_std'].isna(),'Dir_std'] = np.pi



standarize_direction(df_players)
def data_augmentation(df, sample_ids):

    df_sample = df.loc[df.PlayId.isin(sample_ids)].copy()

    df_sample['Y_std'] = 160/3  - df_sample['Y_std']

    df_sample['Dir_std'] = df_sample['Dir_std'].apply(lambda x: 2*np.pi - x)

    df_sample['PlayId'] = df_sample['PlayId'].apply(lambda x: x+'_aug')

    return df_sample



def process_tracking_data(df):

    # More feature engineering for all:

    df['Sx'] = df['S']*df['Dir_std'].apply(math.cos)

    df['Sy'] = df['S']*df['Dir_std'].apply(math.sin)

    

    # ball carrier position

    rushers = df[df['IsRusher']].copy()

    rushers.set_index('PlayId', inplace=True, drop=True)

    playId_rusher_map = rushers[['X_std', 'Y_std', 'Sx', 'Sy']].to_dict(orient='index')

    rusher_x = df['PlayId'].apply(lambda val: playId_rusher_map[val]['X_std'])

    rusher_y = df['PlayId'].apply(lambda val: playId_rusher_map[val]['Y_std'])

    rusher_Sx = df['PlayId'].apply(lambda val: playId_rusher_map[val]['Sx'])

    rusher_Sy = df['PlayId'].apply(lambda val: playId_rusher_map[val]['Sy'])

    

    # Calculate differences between the rusher and the players:

    df['player_minus_rusher_x'] = rusher_x - df['X_std']

    df['player_minus_rusher_y'] = rusher_y - df['Y_std']



    # Velocity parallel to direction of rusher:

    df['player_minus_rusher_Sx'] = rusher_Sx - df['Sx']

    df['player_minus_rusher_Sy'] = rusher_Sy - df['Sy']



    return



sample_ids = np.random.choice(df_play.PlayId.unique(), int(0.5*len(df_play.PlayId.unique())))

#sample_ids = df_play.PlayId.unique()



df_players_aug = data_augmentation(df_players, sample_ids)

df_players = pd.concat([df_players, df_players_aug])

df_players.reset_index()



df_play_aug = df_play.loc[df_play.PlayId.isin(sample_ids)].copy()

df_play_aug['PlayId'] = df_play_aug['PlayId'].apply(lambda x: x+'_aug')

df_play = pd.concat([df_play, df_play_aug])

df_play.reset_index()



# This is necessary to maintain the order when in the next cell we use groupby

df_players.sort_values(by=['PlayId'],inplace=True)

df_play.sort_values(by=['PlayId'],inplace=True)



process_tracking_data(df_players)
tracking_level_features = [

    'PlayId',

    'IsOnOffense',

    'X_std',

    'Y_std',

    'Sx',

    'Sy',

    'player_minus_rusher_x',

    'player_minus_rusher_y',

    'player_minus_rusher_Sx',

    'player_minus_rusher_Sy',

    'IsRusher'

]



df_all_feats = df_players[tracking_level_features]



print('Any null values: ', df_all_feats.isnull().sum().sum())
%%time



grouped = df_all_feats.groupby('PlayId')

train_x = np.zeros([len(grouped.size()),11,10,10])

i = 0

play_ids = df_play.PlayId.values

for name, group in grouped:

    if name!=play_ids[i]:

        print("Error")



    [[rusher_x, rusher_y, rusher_Sx, rusher_Sy]] = group.loc[group.IsRusher==1,['X_std', 'Y_std','Sx','Sy']].values



    offense_ids = group[group.IsOnOffense & ~group.IsRusher].index

    defense_ids = group[~group.IsOnOffense].index



    for j, defense_id in enumerate(defense_ids):

        [def_x, def_y, def_Sx, def_Sy] = group.loc[defense_id,['X_std', 'Y_std','Sx','Sy']].values

        [def_rusher_x, def_rusher_y] = group.loc[defense_id,['player_minus_rusher_x', 'player_minus_rusher_y']].values

        [def_rusher_Sx, def_rusher_Sy] =  group.loc[defense_id,['player_minus_rusher_Sx', 'player_minus_rusher_Sy']].values

        

        train_x[i,j,:,:4] = group.loc[offense_ids,['Sx','Sy','X_std', 'Y_std']].values - np.array([def_Sx, def_Sy, def_x,def_y])

        train_x[i,j,:,-6:] = [def_rusher_Sx, def_rusher_Sy, def_rusher_x, def_rusher_y, def_Sx, def_Sy]

    

    i+=1
# Transform Y into indexed-classes:

train_y = df_play[['PlayId', 'Yards']].copy()



train_y['YardIndex'] = train_y['Yards'].apply(lambda val: val + 99)



min_idx_y = 71

max_idx_y = 150



train_y['YardIndexClipped'] = train_y['YardIndex'].apply(

    lambda val: min_idx_y if val < min_idx_y else max_idx_y if val > max_idx_y else val)



print('max yardIndex: ', train_y.YardIndex.max())

print('max yardIndexClipped: ', train_y.YardIndexClipped.max())

print('min yardIndex: ', train_y.YardIndex.min())

print('min yardIndexClipped: ', train_y.YardIndexClipped.min())
df_season = df_play[['PlayId', 'Season']].copy()
from tensorflow.keras.models import Model



from tensorflow.keras.layers import (

    Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, AvgPool1D, AvgPool2D, Reshape,

    Input, Activation, BatchNormalization, Dense, Add, Lambda, Dropout, LayerNormalization)



from tensorflow.keras.optimizers import Adam

from tensorflow.keras import backend as K

from tensorflow.keras.callbacks import Callback, EarlyStopping



import tensorflow as tf 

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold



def crps(y_true, y_pred):

    loss = K.mean(K.sum((K.cumsum(y_pred, axis = 1) - K.cumsum(y_true, axis=1))**2, axis=1))/199

    return loss
min_idx_y = 71

max_idx_y = 150

num_classes_y = max_idx_y - min_idx_y + 1
def get_conv_net(num_classes_y):

    #_, x, y, z = train_x.shape

    inputdense_players = Input(shape=(11,10,10), name = "playersfeatures_input")

    

    X = Conv2D(128, kernel_size=(1,1), strides=(1,1), activation='relu')(inputdense_players)

    X = Conv2D(160, kernel_size=(1,1), strides=(1,1), activation='relu')(X)

    X = Conv2D(128, kernel_size=(1,1), strides=(1,1), activation='relu')(X)

    

    # The second block of convolutions learns the necessary information per defense player before the aggregation.

    # For this reason the pool_size should be (1, 10). If you want to learn per off player the pool_size must be 

    # (11, 1)

    Xmax = MaxPooling2D(pool_size=(1,10))(X)

    Xmax = Lambda(lambda x1 : x1*0.3)(Xmax)



    Xavg = AvgPool2D(pool_size=(1,10))(X)

    Xavg = Lambda(lambda x1 : x1*0.7)(Xavg)



    X = Add()([Xmax, Xavg])

    X = Lambda(lambda y : K.squeeze(y,2))(X)

    X = BatchNormalization()(X)

    

    X = Conv1D(160, kernel_size=1, strides=1, activation='relu')(X)

    X = BatchNormalization()(X)

    X = Conv1D(96, kernel_size=1, strides=1, activation='relu')(X)

    X = BatchNormalization()(X)

    X = Conv1D(96, kernel_size=1, strides=1, activation='relu')(X)

    X = BatchNormalization()(X)

    

    Xmax = MaxPooling1D(pool_size=11)(X)

    Xmax = Lambda(lambda x1 : x1*0.3)(Xmax)



    Xavg = AvgPool1D(pool_size=11)(X)

    Xavg = Lambda(lambda x1 : x1*0.7)(Xavg)



    X = Add()([Xmax, Xavg])

    X = Lambda(lambda y : K.squeeze(y,1))(X)

    

    X = Dense(96, activation="relu")(X)

    X = BatchNormalization()(X)



    X = Dense(256, activation="relu")(X)

    X = LayerNormalization()(X)

    X = Dropout(0.3)(X)



    outsoft = Dense(num_classes_y, activation='softmax', name = "output")(X)



    model = Model(inputs = [inputdense_players], outputs = outsoft)

    return model
class Metric(Callback):

    def __init__(self, model, callbacks, data):

        super().__init__()

        self.model = model

        self.callbacks = callbacks

        self.data = data



    def on_train_begin(self, logs=None):

        for callback in self.callbacks:

            callback.on_train_begin(logs)



    def on_train_end(self, logs=None):

        for callback in self.callbacks:

            callback.on_train_end(logs)



    def on_epoch_end(self, batch, logs=None):

        X_valid, y_valid = self.data[0], self.data[1]



        y_pred = self.model.predict(X_valid)

        y_true = np.clip(np.cumsum(y_valid, axis=1), 0, 1)

        y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1)

        val_s = ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * X_valid.shape[0])

        logs['val_CRPS'] = val_s

        

        for callback in self.callbacks:

            callback.on_epoch_end(batch, logs)
%%time



models = []

kf = KFold(n_splits=6, shuffle=True)

score = []



for i, (tdx, vdx) in enumerate(kf.split(train_x, train_y)):

    print(f'Fold : {i}')

    X_train, X_val = train_x[tdx], train_x[vdx],

    y_train, y_val = train_y.iloc[tdx]['YardIndexClipped'].values, train_y.iloc[vdx]['YardIndexClipped'].values

    season_val = df_season.iloc[vdx]['Season'].values



    y_train_values = np.zeros((len(y_train), num_classes_y), np.int32)

    for irow, row in enumerate(y_train):

        y_train_values[(irow, row - min_idx_y)] = 1

        

    y_val_values = np.zeros((len(y_val), num_classes_y), np.int32)

    for irow, row in enumerate(y_val - min_idx_y):

        y_val_values[(irow, row)] = 1



    val_idx = np.where(season_val!=2017)

    

    X_val = X_val[val_idx]

    y_val_values = y_val_values[val_idx]



    y_train_values = y_train_values.astype('float32')

    y_val_values = y_val_values.astype('float32')

    

    model = get_conv_net(num_classes_y)



    es = EarlyStopping(monitor='val_CRPS',

                        mode='min',

                        restore_best_weights=True,

                        verbose=0,

                        patience=10)

    

    es.set_model(model)

    metric = Metric(model, [es], [X_val, y_val_values])



    lr_i = 1e-3

    lr_f = 5e-4

    n_epochs = 30 



    decay = (1-lr_f/lr_i)/((lr_f/lr_i)* n_epochs - 1)  #Time-based decay formula

    alpha = (lr_i*(1+decay))

    

    opt = Adam(learning_rate=1e-3)

    model.compile(loss=crps,

                  optimizer=opt)

    

    model.fit(X_train,

              y_train_values, 

              epochs=n_epochs,

              batch_size=64,

              verbose=0,

              callbacks=[metric],

              validation_data=(X_val, y_val_values))



    val_crps_score = min(model.history.history['val_CRPS'])

    print("Val loss: {}".format(val_crps_score))

    

    score.append(val_crps_score)



    models.append(model)

    

print(np.mean(score))
print(np.mean(score))
def get_cdf_prediction_model(predict_x, n_classes=None, model=None, min_idx=None, max_idx=None, yardline100=None):

    '''

    predict_x - array-like of shape [nsamples, n_features]

    min_idx - minimum index considered in training for target var

    max_idx - maximum index considered in training for target var

    '''

    #now = time()

    prediction = model.predict(predict_x)

    

    # Convert data to array of pdfs indexed by training example

    predict_pdfs = np.zeros((len(predict_x), n_classes))



    predict_pdfs[:, min_idx:max_idx+1] = prediction

    

    # can't predict probability of gaining more yards than end zone,

    # so instead: drop and re-normalize?

    max_target_cls_idx = yardline100 + 99

    for idx, predict_row in enumerate(predict_pdfs):

        max_idx = max_target_cls_idx[idx]

        #predict_pdfs[idx, max_idx] = np.sum(predict_row[max_idx:])

        predict_pdfs[idx, max_idx+1:] = 0.0

        # Now renormalize:

        predict_pdfs[idx, :] = predict_pdfs[idx, :]/predict_pdfs[idx, :].sum()

    

    # convert to cdfs:

    predict_cdfs = np.cumsum(predict_pdfs, axis=1)

    return predict_cdfs, predict_pdfs
def split_play_and_player_cols(df,predicting=False):

    df['IsRusher'] = df['NflId'] == df['NflIdRusher']

    

    play_ids = df["PlayId"].unique()

    

    # We must assume here that the first 22 rows correspond to the same player:

    player_cols = [

        'PlayId', # This is the link between them

        'Season',

        'Team',

        'X',

        'Y',

        'S',

        'A',

        'Dis',

        'Dir',

        'NflId',

        'IsRusher',

    ]



    df_players = df[player_cols]

    

    play_cols = [

        'PlayId',

        'Season',

        'PossessionTeam',

        'HomeTeamAbbr',

        'VisitorTeamAbbr',

        'PlayDirection', 

        'FieldPosition',

        'YardLine',

    ]

    if not predicting:

        play_cols.append('Yards')

        

    df_play = df[play_cols].copy()



    ## Fillna in FieldPosition attribute

    #df['FieldPosition'] = df.groupby(['PlayId'], sort=False)['FieldPosition'].apply(lambda x: x.ffill().bfill())

    

    # Get first 

    df_play = df_play.groupby('PlayId').first().reset_index()



    #print('rows/plays in df: ', len(df_play))

    assert df_play.PlayId.nunique() == df.PlayId.nunique(), "Play/player split failed?"  # Boom

    

    return df_play, df_players
def extract_features_one_play(df_test):

    df_play, df_players_split = split_play_and_player_cols(df_test, predicting=True)

    

    process_team_abbr(df_play)

    process_play_direction(df_play)

    process_yard_til_end_zone(df_play)

    

    df_players = df_players_split.merge(

        df_play[['PlayId','PossessionTeam','HomeTeamAbbr','PlayDirection', 'Yardline100']], 

        how='left', on='PlayId')

    

    standarize_direction(df_players)

    process_tracking_data(df_players)

    df_all_feats = df_players[tracking_level_features]



  

    train_x = np.zeros([1,11,10,10])

    [[rusher_x, rusher_y, rusher_Sx, rusher_Sy]] = df_all_feats.loc[df_all_feats.IsRusher==1,['X_std', 'Y_std','Sx','Sy']].values



    offense_ids = df_all_feats[df_all_feats.IsOnOffense & ~df_all_feats.IsRusher].index

    defense_ids = df_all_feats[~df_all_feats.IsOnOffense].index



    for j, defense_id in enumerate(defense_ids):

        [def_x, def_y, def_Sx, def_Sy] = df_all_feats.loc[defense_id,['X_std', 'Y_std','Sx','Sy']].values

        [def_rusher_x, def_rusher_y] = df_all_feats.loc[defense_id,['player_minus_rusher_x', 'player_minus_rusher_y']].values

        [def_rusher_Sx, def_rusher_Sy] =  df_all_feats.loc[defense_id,['player_minus_rusher_Sx', 'player_minus_rusher_Sy']].values



        train_x[0,j,:,:4] = df_all_feats.loc[offense_ids,['Sx','Sy','X_std', 'Y_std']].values - np.array([def_Sx, def_Sy, def_x,def_y])

        train_x[0,j,:,-6:] = [def_rusher_Sx, def_rusher_Sy, def_rusher_x, def_rusher_y, def_Sx, def_Sy]





    yardline100 = df_play['Yardline100'].values

    return [train_x, yardline100]
iter_test = env.iter_test()

for iplay, (test_df, sample_prediction_df) in enumerate(iter_test):

    if iplay % 10 == 0:

        print('Processing test play {}...'.format(iplay))

    [inference_row, yardline100] = extract_features_one_play(test_df)

    

    cdfs_arr = []

    for model in models:

        cdfs_, pdfs_ = get_cdf_prediction_model(

            inference_row, 

            n_classes=199, 

            model=model, 

            min_idx=min_idx_y, 

            max_idx=max_idx_y,

            yardline100=yardline100)

        cdfs_arr.append(cdfs_)

    cdfs = np.mean(cdfs_arr, axis=0)



    # To avoid rounding error:

    cdf_val = cdfs[0].clip(0.0,1.0)

    

    for icol, col in enumerate(sample_prediction_df.columns):

        sample_prediction_df.loc[:, col] = cdf_val[icol]

    env.predict(sample_prediction_df)
env.write_submission_file()

print([filename for filename in os.listdir('/kaggle/working') if '.csv' in filename])