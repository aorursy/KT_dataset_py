import pandas as pd

import numpy as np

import warnings

warnings.simplefilter('ignore')

from catboost import CatBoostRegressor

from sklearn.linear_model import LinearRegression 

# from sklearn.ensemble import RandomForestRegressor

# from sklearn.ensemble import RandomForestClassifier

import lightgbm as lgb

from sklearn.ensemble import ExtraTreesRegressor

#https://www.tensorflow.org/guide/migrate

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

from tqdm import tqdm_notebook

from string import punctuation

import re

import datetime

from scipy.stats import norm

import gc

import random



np.random.seed(42)

random.seed(42)
def convert_pred(yards):

    p = 99 + yards # 99 + Yards

    gap = 10 # -10..10



    y_pred = np.zeros(199)

    y_pred[p-gap:p+gap] = np.cumsum(norm.pdf(np.arange(-gap,gap,1),0,3))

    y_pred[p+gap:] = 1

    return pd.Series(y_pred)
def convert_y(yards):

    p = 99 + yards # 99 + Yards



    y_ans = np.zeros(199)

    y_ans[p:] = 1

    return pd.Series(y_ans)
def predict_models(models, df):

    preds = []

    for model in models:        

        pred = model.predict(df)            

        preds.append(pred)

         

    return np.column_stack(preds)
class GenerateData:

    def __init__(self, cat_features, drop_cols=[]):

        self.cat_features = list(set(cat_features) - set(drop_cols))

        self.drop_cols = drop_cols

        self.players_score = None

        self.category_dict = None

                

        

    def make_train_score(self, df):

        df['MaxHomeScore'] = df.groupby('GameId')['HomeScoreBeforePlay'].transform('max')

        df['MaxVisitorScore'] = df.groupby('GameId')['VisitorScoreBeforePlay'].transform('max')    



        df['Score'] = 0

        df.loc[df['Team'] == 'home', 'Score'] = df[df['Team'] == 'home']['MaxHomeScore'] - df[df['Team'] == 'home']['MaxVisitorScore']

        df.loc[df['Team'] == 'away', 'Score'] = df[df['Team'] == 'away']['MaxVisitorScore'] - df[df['Team'] == 'away']['MaxHomeScore']

        df['Score'] = df['Score'].map(lambda x: 1 if x>0 else 0 if x<0 else 0.5)



        players_score = df[['GameId', 'NflId', 'Score']].drop_duplicates()



        players_score['GeneralScore'] = players_score.groupby('NflId')['Score'].transform('sum')

        players_score = players_score[['NflId', 'GeneralScore']].drop_duplicates()



        df = df.merge(players_score, on='NflId', how='left')



        df.loc[df['Team'] == 'away', 'GeneralScore'] = df[df['Team'] == 'away'].groupby('GameId')['GeneralScore'].transform(lambda x: np.round(x.mode().mean()))

        df.loc[df['Team'] == 'home', 'GeneralScore'] = df[df['Team'] == 'home'].groupby('GameId')['GeneralScore'].transform(lambda x: np.round(x.mode().mean()))



        df.drop(['MaxHomeScore', 'MaxVisitorScore', 'Score'], axis=1, inplace=True)

    

        return df, players_score

    

    

    def make_test_score(self, df, players_score):

        df = df.merge(players_score, on='NflId', how='left')



        df.loc[df['Team'] == 'away', 'GeneralScore'] = df[df['Team'] == 'away'].groupby('GameId')['GeneralScore'].transform(lambda x: np.round(x.mode().mean()))

        df.loc[df['Team'] == 'home', 'GeneralScore'] = df[df['Team'] == 'home'].groupby('GameId')['GeneralScore'].transform(lambda x: np.round(x.mode().mean()))



        return df

    

    

    # convert to category train and save categories for test

    def to_category_train(self, df):

        df = df.fillna(-999)

        category_dict = {}

        for col in self.cat_features:

            f = df[col].factorize()[0]

            d = dict(zip(df[col], f))

            category_dict[col] = d



            df[col] = df[col].map(d)



        return df, category_dict



    

    # convert to categories test, add new categories if need

    def to_category_test(self, df, category_dict):

        df = df.fillna(-999)

        for col, d in category_dict.items():

            df[col + '_orig'] = df[col]

            df[col] = df[col].map(d)



            mask = df[col].isna()

            if mask.any():

                increment = max(d.values()) + 1

                df.loc[mask, col] = df.loc[mask, col + '_orig'].factorize()[0] + increment

                df[col] = df[col].astype('int64')



        del_cols = [i for i in df.columns if '_orig' in i] 

        df.drop(del_cols, axis=1, inplace=True)



        return df

    

    

    def convert_train(self, df):

        df, self.players_score = self.make_train_score(df)

        df = self.make_new_features(df)

        

        # Clearly: Yards<=YardsLeft and YardsLeft-100<=Yards, thus we are going to drop those wrong lines.

        df.drop(df.index[(df['YardsLeft']<df['Yards']) | (df['YardsLeft']-100>df['Yards'])], inplace=True)

        df, self.category_dict = self.to_category_train(df)

        

        return df

    

    def convert_test(self, df):

        df = self.make_test_score(df, self.players_score)

        df = self.make_new_features(df)

        df = self.to_category_test(df, self.category_dict)

        

        return df



    

    # make common features

    def make_new_features(self, df):

        ## convert GameClock to milliseconds

        def convert_to_milliseconds(t):

            t = t.split(':')

            t = list(map(int, t))

            t = t[0]*60*60 + t[1]*60 + t[2]

            return t        

        

        df['GameClock'] = df['GameClock'].apply(convert_to_milliseconds)

        

        

        ## from https://www.kaggle.com/prashantkikani/nfl-starter-lgb-feature-engg

        df['DefendersInTheBox_vs_Distance'] = df['DefendersInTheBox'] / df['Distance']

        

        ## clean and transform StadiumType

        outdoor = ['Outdoor', 'Outdoors', 'Cloudy', 'Heinz Field', 'Outdor', 'Ourdoor', 

                   'Outside', 'Outddors','Outdoor Retr Roof-Open', 'Oudoor', 'Bowl']



        indoor_closed = ['Indoors', 'Indoor', 'Indoor, Roof Closed', 'Indoor, Roof Closed', 'Retractable Roof',

                         'Retr. Roof-Closed', 'Retr. Roof - Closed', 'Retr. Roof Closed']



        indoor_open   = ['Indoor, Open Roof', 'Open', 'Retr. Roof-Open', 'Retr. Roof - Open']

        dome_closed   = ['Dome', 'Domed, closed', 'Closed Dome', 'Domed', 'Dome, closed']

        dome_open     = ['Domed, Open', 'Domed, open']

        

        df['StadiumType'] = df['StadiumType'].replace(outdoor,'outdoor')

        df['StadiumType'] = df['StadiumType'].replace(indoor_closed,'indoor_closed')

        df['StadiumType'] = df['StadiumType'].replace(indoor_open,'indoor_open')

        df['StadiumType'] = df['StadiumType'].replace(dome_closed,'dome_closed')

        df['StadiumType'] = df['StadiumType'].replace(dome_open,'dome_open')

        

        

        ## clean and transform Turf

        #from https://www.kaggle.com/c/nfl-big-data-bowl-2020/discussion/112681#latest-649087

        natural_grass = ['natural grass','Naturall Grass','Natural Grass']

        grass = ['Grass']

        fieldturf = ['FieldTurf','Field turf','FieldTurf360','Field Turf']

        artificial = ['Artificial','Artifical']



        df['Turf'] = df['Turf'].replace(natural_grass,'natural_grass')

        df['Turf'] = df['Turf'].replace(grass,'grass')

        df['Turf'] = df['Turf'].replace(fieldturf,'fieldturf')

        df['Turf'] = df['Turf'].replace(artificial,'artificial')



        

        # deal with Possession Team

        map_abbr = {'ARI': 'ARZ', 'BAL': 'BLT', 'CLE': 'CLV', 'HOU': 'HST'}

        for abb in df['PossessionTeam'].unique():

            map_abbr[abb] = abb

        

        df['PossessionTeam'] = df['PossessionTeam'].map(map_abbr)

        df['HomeTeamAbbr'] = df['HomeTeamAbbr'].map(map_abbr)

        df['VisitorTeamAbbr'] = df['VisitorTeamAbbr'].map(map_abbr)

        

        df['HomePossesion'] = df['PossessionTeam'] == df['HomeTeamAbbr']

        

        df['Field_eq_Possession'] = df['FieldPosition'] == df['PossessionTeam']

        df['HomeField'] = df['FieldPosition'] == df['HomeTeamAbbr']

        

        ## convert PlayerHeight to inches

        df['PlayerHeight'] = df['PlayerHeight'].apply(lambda x: 12*int(x.split('-')[0])+int(x.split('-')[1]))

        

        ## calculate BMI

        df['PlayerBMI'] = (703*df['PlayerWeight'])/(df['PlayerHeight']**2)

                

        # convert to and calculate unix time

        df['TimeHandoff'] = df['TimeHandoff'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ")).astype(int)// 10**9

        df['TimeSnap'] = df['TimeSnap'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ")).astype(int)// 10**9

        df['TimeDelta'] = df['TimeHandoff'] - df['TimeSnap']

        df['PlayerBirthDate'] = df['PlayerBirthDate'].apply(lambda x: datetime.datetime.strptime(x, "%m/%d/%Y")).astype(int)// 10**9

        

        seconds_in_year = 60*60*24*365.25

        df['PlayerAge'] = df['PlayerBirthDate'] / seconds_in_year

        

        #df.drop(['TimeHandoff', 'TimeSnap', 'PlayerBirthDate'], axis=1, inplace=True)

        

        ## WindSpeed and Direction

        df['WindSpeed'] = df['WindSpeed'].apply(lambda x: x.lower().replace('mph', '').strip() if not pd.isna(x) else x)

        

        # let's replace the ones that has x-y by (x+y)/2

        # and also the ones with x gusts up to y

        df['WindSpeed'] = df['WindSpeed'].apply(lambda x: (int(x.split('-')[0])+int(x.split('-')[1]))/2 if not pd.isna(x) and '-' in x else x)

        df['WindSpeed'] = df['WindSpeed'].apply(lambda x: (int(x.split()[0])+int(x.split()[-1]))/2 if not pd.isna(x) and type(x)!=float and 'gusts up to' in x else x)

        

        def str_to_float(txt):

            try:

                return float(txt)

            except:

                return -999

            

        df['WindSpeed'] = df['WindSpeed'].apply(str_to_float)

        

    

        north = ['N','From S','North']

        south = ['S','From N','South','s']

        west = ['W','From E','West']

        east = ['E','From W','from W','EAST','East']

        north_east = ['FROM SW','FROM SSW','FROM WSW','NE','NORTH EAST','North East','East North East','NorthEast','Northeast','ENE','From WSW','From SW']

        north_west = ['E','From ESE','NW','NORTHWEST','N-NE','NNE','North/Northwest','W-NW','WNW','West Northwest','Northwest','NNW','From SSE']

        south_east = ['E','From WNW','SE','SOUTHEAST','South Southeast','East Southeast','Southeast','SSE','From SSW','ESE','From NNW']

        south_west = ['E','From ENE','SW','SOUTHWEST','W-SW','South Southwest','West-Southwest','WSW','SouthWest','Southwest','SSW','From NNE']

        no_wind = ['clear','Calm']

        nan = ['1','8','13']



        df['WindDirection'] = df['WindDirection'].replace(north,'north')

        df['WindDirection'] = df['WindDirection'].replace(south,'south')

        df['WindDirection'] = df['WindDirection'].replace(west,'west')

        df['WindDirection'] = df['WindDirection'].replace(east,'east')

        df['WindDirection'] = df['WindDirection'].replace(north_east,'north_east')

        df['WindDirection'] = df['WindDirection'].replace(north_west,'north_west')

        df['WindDirection'] = df['WindDirection'].replace(south_east,'clear')

        df['WindDirection'] = df['WindDirection'].replace(south_west,'south_west')

        df['WindDirection'] = df['WindDirection'].replace(no_wind,'no_wind')

        df['WindDirection'] = df['WindDirection'].replace(nan,np.nan)       

        

        ## deal with PlayDirection

        df['PlayDirection'] = df['PlayDirection'].apply(lambda x: x.strip() == 'right')

        

        ## deal with Team

        df['Team'] = df['Team'].apply(lambda x: x.strip()=='home')

               

        ## deal with GameWeather

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

                

        df['GameWeather'] = df['GameWeather'].replace(rain,'rain')

        df['GameWeather'] = df['GameWeather'].replace(overcast,'overcast')

        df['GameWeather'] = df['GameWeather'].replace(clear,'clear')

        df['GameWeather'] = df['GameWeather'].replace(snow,'snow')

        df['GameWeather'] = df['GameWeather'].replace(none,'none')

        

        ## deal with NflId and NflIdRusher

        df['IsRusher'] = df['NflId'] == df['NflIdRusher']

        #df.drop(['NflId', 'NflIdRusher'], axis=1, inplace=True)

        



        ## deal with PlayDirection

        df.loc[df['PlayDirection'] == 'left', 'X'] = 120 - df.loc[df['PlayDirection'] == 'left', 'X']

        df.loc[df['PlayDirection'] == 'left', 'Y'] = (160 / 3) - df.loc[df['PlayDirection'] == 'left', 'Y']

        df.loc[df['PlayDirection'] == 'left', 'Orientation'] = np.mod(180 + df.loc[df['PlayDirection'] == 'left', 'Orientation'], 360)

        df.loc[df['PlayDirection'] == 'left', 'Dir'] = np.mod(180 + df.loc[df['PlayDirection'] == 'left', 'Dir'], 360)

        df['FieldPosition'].fillna('', inplace=True)

        df.loc[df['PossessionTeam'] != df['FieldPosition'], 'YardLine'] = 100 - df.loc[df['PossessionTeam'] != df['FieldPosition'], 'YardLine']



        # Add 90 to Orientation for 2017 season only

        df.loc[df['Season'] == 2017, 'Orientation'] = np.mod(90 + df.loc[df['Season'] == 2017, 'Orientation'], 360)

        

        ## deal with YardLine

        df['YardsLeft'] = df.apply(lambda row: 100-row['YardLine'] if row['HomeField'] else row['YardLine'], axis=1)

        df['YardsLeft'] = df.apply(lambda row: row['YardsLeft'] if row['PlayDirection'] else 100-row['YardsLeft'], axis=1)

        

        df['DiffScore'] = (df['HomeScoreBeforePlay'] - df['VisitorScoreBeforePlay']).abs()   

        

        df['MaxHomeScore'] = df.groupby('GameId')['HomeScoreBeforePlay'].transform('max')

        df['MaxVisitorScore'] = df.groupby('GameId')['VisitorScoreBeforePlay'].transform('max') 

        

        df.drop(self.drop_cols, axis=1, inplace=True)

        

        

        return df
df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)
# 'Season' is split feature

cat_features = ['PlayId', 'Team', 'NflId', 'Quarter', 'PossessionTeam', 'FieldPosition', 'NflIdRusher', 'OffenseFormation', 'OffensePersonnel', 'DefendersInTheBox', 'DefensePersonnel', 'PlayDirection', 'PlayerCollegeName', 'Position', 'HomeTeamAbbr', 'VisitorTeamAbbr', 'Week', 'Stadium', 'Location', 'StadiumType', 'Turf', 'GameWeather', 'WindDirection', 'HomePossesion', 'Field_eq_Possession', 'HomeField', 'IsRusher']

drop_cols = ['DisplayName', 'JerseyNumber', 'GameId', 'PlayId', 'NflId', 'NflIdRusher'] + ['HomePossesion', 'PlayerAge', 'Temperature', 'Turf']

drop_cols += ['YardLine', 'MaxVisitorScore', 'DiffScore', 'DefendersInTheBox', 'WindSpeed']
gd = GenerateData(cat_features, drop_cols)

df = gd.convert_train(df)
train = df[df['Season'] == 2017]

y_train = train['Yards']

train.drop(['Yards', 'Season'], axis=1, inplace=True)



valid = df[df['Season'] == 2018]

y_valid = valid['Yards']

valid.drop(['Yards', 'Season'], axis=1, inplace=True)



del df

gc.collect()
models = []
# Two Keras models

class KerasCustom:

    def __init__(self):

        self.model = None

        self.input_shape = None

              

    def create_model(self, input_shape):

        model = tf.keras.models.Sequential()

        model.add(tf.keras.layers.Conv1D(96, 2, activation='tanh', input_shape=(input_shape, 1), kernel_initializer=tf.keras.initializers.glorot_uniform(seed=42)))

        model.add(tf.keras.layers.BatchNormalization())       

        model.add(tf.keras.layers.Flatten())

        model.add(tf.keras.layers.Dropout(0.25))

        model.add(tf.keras.layers.Dense(96, activation='tanh', kernel_initializer=tf.keras.initializers.glorot_uniform(seed=42)))

        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Dropout(0.25))

        model.add(tf.keras.layers.Dense(1, activation='linear', kernel_initializer=tf.keras.initializers.glorot_uniform(seed=42)))

        model.compile(loss=['mae'], optimizer=tf.keras.optimizers.Adam(lr=0.001))



        return model 

    

    def fit(self, train, y_train):

        es = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0001, patience=5, verbose=1, mode='min', restore_best_weights=True)



        self.input_shape = train.shape[1]

        self.model = self.create_model(self.input_shape)

        self.model.fit(train.values.reshape(-1, self.input_shape, 1), y_train, 

                       #validation_data=(df.values.reshape(-1, input_shape, 1), y_test), 

                       epochs=100, #10 

                       batch_size=1024,

                       callbacks=[es],

                       verbose=1)



    def predict(self, df):

        return self.model.predict(df.values.reshape(-1, self.input_shape, 1))



model = KerasCustom()

model.fit(train, y_train)



models.append(model)



###



class KerasCustom:

    def __init__(self):

        self.model = None

        self.input_shape = None

              

    def create_model(self, input_shape):

        model = tf.keras.models.Sequential()

        model.add(tf.keras.layers.Conv1D(96, 2, activation='tanh', input_shape=(input_shape, 1), kernel_initializer=tf.keras.initializers.glorot_uniform(seed=42)))

        model.add(tf.keras.layers.BatchNormalization())       

        model.add(tf.keras.layers.Flatten())

        model.add(tf.keras.layers.Dropout(0.25))

#         model.add(tf.keras.layers.Dense(96, activation='tanh', kernel_initializer=tf.keras.initializers.glorot_uniform(seed=42)))

#         model.add(tf.keras.layers.BatchNormalization())

#         model.add(tf.keras.layers.Dropout(0.25))

        model.add(tf.keras.layers.Dense(1, activation='linear', kernel_initializer=tf.keras.initializers.glorot_uniform(seed=42)))

        model.compile(loss=['mae'], optimizer=tf.keras.optimizers.Adam(lr=0.001))



        return model 

    

    def fit(self, train, y_train):

        es = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0001, patience=3, verbose=1, mode='min', restore_best_weights=True)



        self.input_shape = train.shape[1]

        self.model = self.create_model(self.input_shape)

        self.model.fit(train.values.reshape(-1, self.input_shape, 1), y_train, 

                       #validation_data=(df.values.reshape(-1, input_shape, 1), y_test), 

                       epochs=100, #10 

                       batch_size=1024,

                       callbacks=[es],

                       verbose=1)



    def predict(self, df):

        return self.model.predict(df.values.reshape(-1, self.input_shape, 1))



model = KerasCustom()

model.fit(train, y_train)



models.append(model)
## Three catboost models

cat_features_ids = [c for c, col in enumerate(train.columns) if col in cat_features]



model = CatBoostRegressor(loss_function='MAE',

                          #eval_metric='MAE',

                          #early_stopping_rounds=200,

                          #learning_rate=0.01,

                          n_estimators=359,

                          depth=6,

                          one_hot_max_size=255,

                          random_seed=42)



model.fit(train, y_train,

          #eval_set=(df, y_test),

          cat_features=cat_features_ids, 

          #use_best_model=True, 

          verbose=200)



models.append(model)



###

#cat_features_ids = [c for c, col in enumerate(train.columns) if col in cat_features]



model = CatBoostRegressor(loss_function='MAE',

                          #eval_metric='MAE',

                          #early_stopping_rounds=200,

                          #learning_rate=0.01,

                          n_estimators=968,

                          depth=3,

                          one_hot_max_size=255,

                          random_seed=42)



model.fit(train, y_train,

          #eval_set=(df, y_test),

          cat_features=cat_features_ids, 

          #use_best_model=True, 

          verbose=200)



models.append(model)



###

#cat_features_ids = [c for c, col in enumerate(train.columns) if col in cat_features]



model = CatBoostRegressor(loss_function='MAE',

                          #eval_metric='MAE',

                          #early_stopping_rounds=200,

                          #learning_rate=0.01,

                          n_estimators=136,

                          depth=9,

                          one_hot_max_size=255,

                          random_seed=42)



model.fit(train, y_train,

          #eval_set=(df, y_test),

          cat_features=cat_features_ids, 

          #use_best_model=True, 

          verbose=200)



models.append(model)
#cat_features_ids = [c for c, col in enumerate(train.columns) if col in cat_features]



model = lgb.LGBMRegressor(n_estimators=194, 

                          random_state=42,

                          learning_rate=0.005,

                          importance_type = 'gain',

                          n_jobs = -1,

                          metric='mae')



model.fit(train, y_train,

          #eval_set=[(train, y), (test[train.columns], y_test)],

          categorical_feature = cat_features_ids,

          #early_stopping_rounds=200,

          verbose=10)



models.append(model)
model=ExtraTreesRegressor(n_estimators=400, n_jobs=-1, bootstrap=True)

model.fit(train, y_train)



models.append(model)
#make predicitons for validation

pred = predict_models(models, valid)
meta_model=LinearRegression(n_jobs=-1)



meta_model.fit(pred, y_valid)
from kaggle.competitions import nflrush

from tqdm import tqdm_notebook



env = nflrush.make_env()



# 3438

for (test, sub) in tqdm_notebook(env.iter_test()):

    df = gd.convert_test(test)

    df.drop('Season', axis=1, inplace=True)

    

    test_pred = predict_models(models, df)

    

    final_pred=meta_model.predict(test_pred)

    

    final_pred = np.mean(final_pred).astype('int')

    

    final_pred = convert_pred(final_pred)

    sub[sub.columns] = final_pred.values 

    

    env.predict(sub)



env.write_submission_file()