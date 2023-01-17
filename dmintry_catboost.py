import pandas as pd

import numpy as np

import warnings

from catboost import CatBoostRegressor

from tqdm import tqdm_notebook

from string import punctuation

import re

import datetime

from scipy.stats import norm



warnings.simplefilter('ignore')
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

        

        ## clean and transform Turf

        #from https://www.kaggle.com/c/nfl-big-data-bowl-2020/discussion/112681#latest-649087

        Turf = {'Field Turf':'Artificial', 'A-Turf Titan':'Artificial', 'Grass':'Natural', 'UBU Sports Speed S5-M':'Artificial', 

                'Artificial':'Artificial', 'DD GrassMaster':'Artificial', 'Natural Grass':'Natural', 

                'UBU Speed Series-S5-M':'Artificial', 'FieldTurf':'Artificial', 'FieldTurf 360':'Artificial', 'Natural grass':'Natural', 'grass':'Natural', 

                'Natural':'Natural', 'Artifical':'Artificial', 'FieldTurf360':'Artificial', 'Naturall Grass':'Natural', 'Field turf':'Artificial', 

                'SISGrass':'Artificial', 'Twenty-Four/Seven Turf':'Artificial', 'natural grass':'Natural'} 



        df['Turf'] = df['Turf'].map(Turf)

        df['Turf'] = df['Turf'] == 'Natural'

        

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

                return -1

            

        df['WindSpeed'] = df['WindSpeed'].apply(str_to_float)

        

        def clean_WindDirection(txt):

            if pd.isna(txt):

                return np.nan

            txt = txt.lower()

            txt = ''.join([c for c in txt if c not in punctuation])

            txt = txt.replace('from', '')

            txt = txt.replace(' ', '')

            txt = txt.replace('north', 'n')

            txt = txt.replace('south', 's')

            txt = txt.replace('west', 'w')

            txt = txt.replace('east', 'e')

            return txt

        

        df['WindDirection'] = df['WindDirection'].apply(clean_WindDirection)

        

        def transform_WindDirection(txt):

            if pd.isna(txt):

                return np.nan



            if txt=='n':

                return 0

            if txt=='nne' or txt=='nen':

                return 1/8

            if txt=='ne':

                return 2/8

            if txt=='ene' or txt=='nee':

                return 3/8

            if txt=='e':

                return 4/8

            if txt=='ese' or txt=='see':

                return 5/8

            if txt=='se':

                return 6/8

            if txt=='ses' or txt=='sse':

                return 7/8

            if txt=='s':

                return 8/8

            if txt=='ssw' or txt=='sws':

                return 9/8

            if txt=='sw':

                return 10/8

            if txt=='sww' or txt=='wsw':

                return 11/8

            if txt=='w':

                return 12/8

            if txt=='wnw' or txt=='nww':

                return 13/8

            if txt=='nw':

                return 14/8

            if txt=='nwn' or txt=='nnw':

                return 15/8

            return np.nan



        df['WindDirection'] = df['WindDirection'].apply(transform_WindDirection)

        

        ## deal with PlayDirection

        df['PlayDirection'] = df['PlayDirection'].apply(lambda x: x.strip() == 'right')

        

        ## deal with Team

        df['Team'] = df['Team'].apply(lambda x: x.strip()=='home')

               

        ## deal with GameWeather

        df['GameWeather'] = df['GameWeather'].str.lower()

        indoor = "indoor"

        df['GameWeather'] = df['GameWeather'].apply(lambda x: indoor if not pd.isna(x) and indoor in x else x)

        df['GameWeather'] = df['GameWeather'].apply(lambda x: x.replace('coudy', 'cloudy').replace('clouidy', 'cloudy').replace('party', 'partly') if not pd.isna(x) else x)

        df['GameWeather'] = df['GameWeather'].apply(lambda x: x.replace('clear and sunny', 'sunny and clear') if not pd.isna(x) else x)

        df['GameWeather'] = df['GameWeather'].apply(lambda x: x.replace('skies', '').replace("mostly", "").strip() if not pd.isna(x) else x)

        

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

        

        df['GameWeather'] = df['GameWeather'].apply(map_weather)

        

        ## deal with NflId and NflIdRusher

        df['IsRusher'] = df['NflId'] == df['NflIdRusher']

        #df.drop(['NflId', 'NflIdRusher'], axis=1, inplace=True)

        

        

        ## deal with PlayDirection and Orientation

        # inverse dirrection if it need

        df['X'] = df.apply(lambda row: row['X'] if row['PlayDirection'] else 120-row['X'], axis=1)

        

        #from https://www.kaggle.com/scirpus/hybrid-gp-and-nn

        def new_orientation(angle, play_direction):

            if play_direction == 0:

                new_angle = 360.0 - angle

                if new_angle == 360.0:

                    new_angle = 0.0

                return new_angle

            else:

                return angle



        df['Orientation'] = df.apply(lambda row: new_orientation(row['Orientation'], row['PlayDirection']), axis=1)

        df['Dir'] = df.apply(lambda row: new_orientation(row['Dir'], row['PlayDirection']), axis=1)

        

        ## deal with YardLine

        df['YardsLeft'] = df.apply(lambda row: 100-row['YardLine'] if row['HomeField'] else row['YardLine'], axis=1)

        df['YardsLeft'] = df.apply(lambda row: row['YardsLeft'] if row['PlayDirection'] else 100-row['YardsLeft'], axis=1)

               

        

        df.drop(self.drop_cols, axis=1, inplace=True)

        

        

        return df
train = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)

train = train[train['Season'] == 2018]
cat_features = ['PlayId', 'Team', 'NflId', 'Season', 'Quarter', 'PossessionTeam', 'FieldPosition', 'NflIdRusher', 'OffenseFormation', 'OffensePersonnel', 'DefendersInTheBox', 'DefensePersonnel', 'PlayDirection', 'PlayerCollegeName', 'Position', 'HomeTeamAbbr', 'VisitorTeamAbbr', 'Week', 'Stadium', 'Location', 'StadiumType', 'Turf', 'GameWeather', 'WindDirection', 'HomePossesion', 'Field_eq_Possession', 'HomeField', 'IsRusher']

drop_cols = ['DisplayName', 'JerseyNumber', 'GameId', 'PlayId', 'NflId', 'NflIdRusher']
gd = GenerateData(cat_features, drop_cols)

train = gd.convert_train(train)
y = train['Yards']

train.drop('Yards', axis=1, inplace=True)
cat_features_ids = [c for c, col in enumerate(train.columns) if col in cat_features]



model = CatBoostRegressor(loss_function='MAE',

                          eval_metric='MAE',

                          early_stopping_rounds=200,

                          #learning_rate=0.01,

                          #n_estimators=2000,

                          one_hot_max_size=255,

                          random_seed=42)



model.fit(train, y,

          #eval_set=(df, y_test),

          cat_features=cat_features_ids, 

          use_best_model=True, verbose=200)
from kaggle.competitions import nflrush

from tqdm import tqdm_notebook



env = nflrush.make_env()



# 3438

for (test, sub) in tqdm_notebook(env.iter_test()):

    

    df = gd.convert_test(test)

    pred = model.predict(df)

    

    pred = np.mean(pred).astype('int')

    

    pred = convert_pred(pred)

    sub[sub.columns] = pred.values 

    

    env.predict(sub)



env.write_submission_file()