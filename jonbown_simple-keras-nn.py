import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib as mpl

import seaborn as sns

import datetime

from kaggle.competitions import nflrush

import tqdm

import re

from string import punctuation

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import StandardScaler

import keras

from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

from keras.utils import plot_model

import keras.backend as K

import tensorflow as tf

from keras.models import Sequential

from keras.layers import Dense

from keras.utils import to_categorical



sns.set_style('darkgrid')

mpl.rcParams['figure.figsize'] = [15,10]
#Create competition environment

env = nflrush.make_env()
#read in training data

train = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv', dtype={'WindSpeed': 'object'})
#Examine training data

train.head()
#from https://www.kaggle.com/prashantkikani/nfl-starter-lgb-feature-engg

train['DefendersInTheBox_vs_Distance'] = train['DefendersInTheBox'] / train['Distance']
cat_features = []

for col in train.columns:

    if train[col].dtype =='object':

        cat_features.append((col, len(train[col].unique())))
cat_features
train['StadiumType'].value_counts()
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
train['StadiumType'] = train['StadiumType'].apply(clean_StadiumType)
def transform_StadiumType(txt):

    if pd.isna(txt):

        return np.nan

    if 'outdoor' in txt or 'open' in txt:

        return 1

    if 'indoor' in txt or 'closed' in txt:

        return 0

    

    return np.nan
train['StadiumType'] = train['StadiumType'].apply(transform_StadiumType)
#from https://www.kaggle.com/c/nfl-big-data-bowl-2020/discussion/112681#latest-649087

Turf = {'Field Turf':'Artificial', 'A-Turf Titan':'Artificial', 'Grass':'Natural', 'UBU Sports Speed S5-M':'Artificial', 

        'Artificial':'Artificial', 'DD GrassMaster':'Artificial', 'Natural Grass':'Natural', 

        'UBU Speed Series-S5-M':'Artificial', 'FieldTurf':'Artificial', 'FieldTurf 360':'Artificial', 'Natural grass':'Natural', 'grass':'Natural', 

        'Natural':'Natural', 'Artifical':'Artificial', 'FieldTurf360':'Artificial', 'Naturall Grass':'Natural', 'Field turf':'Artificial', 

        'SISGrass':'Artificial', 'Twenty-Four/Seven Turf':'Artificial', 'natural grass':'Natural'} 



train['Turf'] = train['Turf'].map(Turf)

train['Turf'] = train['Turf'] == 'Natural'
train[(train['PossessionTeam']!=train['HomeTeamAbbr']) & (train['PossessionTeam']!=train['VisitorTeamAbbr'])][['PossessionTeam', 'HomeTeamAbbr', 'VisitorTeamAbbr']]
sorted(train['HomeTeamAbbr'].unique()) == sorted(train['VisitorTeamAbbr'].unique())
diff_abbr = []

for x,y  in zip(sorted(train['HomeTeamAbbr'].unique()), sorted(train['PossessionTeam'].unique())):

    if x!=y:

        print(x + " " + y)
map_abbr = {'ARI': 'ARZ', 'BAL': 'BLT', 'CLE': 'CLV', 'HOU': 'HST'}

for abb in train['PossessionTeam'].unique():

    map_abbr[abb] = abb
train['PossessionTeam'] = train['PossessionTeam'].map(map_abbr)

train['HomeTeamAbbr'] = train['HomeTeamAbbr'].map(map_abbr)

train['VisitorTeamAbbr'] = train['VisitorTeamAbbr'].map(map_abbr)
train['HomePossesion'] = train['PossessionTeam'] == train['HomeTeamAbbr']
train['Field_eq_Possession'] = train['FieldPosition'] == train['PossessionTeam']

train['HomeField'] = train['FieldPosition'] == train['HomeTeamAbbr']
off_form = train['OffenseFormation'].unique()

train['OffenseFormation'].value_counts()
train = pd.concat([train.drop(['OffenseFormation'], axis=1), pd.get_dummies(train['OffenseFormation'], prefix='Formation')], axis=1)

dummy_col = train.columns
train['GameClock'].value_counts()
def strtoseconds(txt):

    txt = txt.split(':')

    ans = int(txt[0])*60 + int(txt[1]) + int(txt[2])/60

    return ans
train['GameClock'] = train['GameClock'].apply(strtoseconds)
sns.distplot(train['GameClock'])
train['PlayerHeight']
train['PlayerHeight'] = train['PlayerHeight'].apply(lambda x: 12*int(x.split('-')[0])+int(x.split('-')[1]))
train['PlayerBMI'] = 703*(train['PlayerWeight']/(train['PlayerHeight'])**2)
train['TimeHandoff']
train['TimeHandoff'] = train['TimeHandoff'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))

train['TimeSnap'] = train['TimeSnap'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))
train['TimeDelta'] = train.apply(lambda row: (row['TimeHandoff'] - row['TimeSnap']).total_seconds(), axis=1)
train['PlayerBirthDate'] = train['PlayerBirthDate'].apply(lambda x: datetime.datetime.strptime(x, "%m/%d/%Y"))
seconds_in_year = 60*60*24*365.25

train['PlayerAge'] = train.apply(lambda row: (row['TimeHandoff']-row['PlayerBirthDate']).total_seconds()/seconds_in_year, axis=1)
train = train.drop(['TimeHandoff', 'TimeSnap', 'PlayerBirthDate'], axis=1)
train['WindSpeed'].value_counts()
train['WindSpeed'] = train['WindSpeed'].apply(lambda x: x.lower().replace('mph', '').strip() if not pd.isna(x) else x)
train['WindSpeed'].value_counts()
#let's replace the ones that has x-y by (x+y)/2

# and also the ones with x gusts up to y

train['WindSpeed'] = train['WindSpeed'].apply(lambda x: (int(x.split('-')[0])+int(x.split('-')[1]))/2 if not pd.isna(x) and '-' in x else x)

train['WindSpeed'] = train['WindSpeed'].apply(lambda x: (int(x.split()[0])+int(x.split()[-1]))/2 if not pd.isna(x) and type(x)!=float and 'gusts up to' in x else x)
def str_to_float(txt):

    try:

        return float(txt)

    except:

        return -1
train['WindSpeed'] = train['WindSpeed'].apply(str_to_float)
train['WindDirection'].value_counts()
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
train['WindDirection'] = train['WindDirection'].apply(clean_WindDirection)
train['WindDirection'].value_counts()
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
train['WindDirection'] = train['WindDirection'].apply(transform_WindDirection)
train['PlayDirection'].value_counts()
train['PlayDirection'] = train['PlayDirection'].apply(lambda x: x.strip() == 'right')
train['Team'] = train['Team'].apply(lambda x: x.strip()=='home')
train['GameWeather'].unique()
train['GameWeather'] = train['GameWeather'].str.lower()

indoor = "indoor"

train['GameWeather'] = train['GameWeather'].apply(lambda x: indoor if not pd.isna(x) and indoor in x else x)

train['GameWeather'] = train['GameWeather'].apply(lambda x: x.replace('coudy', 'cloudy').replace('clouidy', 'cloudy').replace('party', 'partly') if not pd.isna(x) else x)

train['GameWeather'] = train['GameWeather'].apply(lambda x: x.replace('clear and sunny', 'sunny and clear') if not pd.isna(x) else x)

train['GameWeather'] = train['GameWeather'].apply(lambda x: x.replace('skies', '').replace("mostly", "").strip() if not pd.isna(x) else x)
train['GameWeather'].unique()
from collections import Counter

weather_count = Counter()

for weather in train['GameWeather']:

    if pd.isna(weather):

        continue

    for word in weather.split():

        weather_count[word]+=1

        

weather_count.most_common()[:15]
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
train['GameWeather'] = train['GameWeather'].apply(map_weather)
train['IsRusher'] = train['NflId'] == train['NflIdRusher']
train.drop(['NflId', 'NflIdRusher'], axis=1, inplace=True)
train['X'] = train.apply(lambda row: row['X'] if row['PlayDirection'] else 120-row['X'], axis=1)
#from https://www.kaggle.com/scirpus/hybrid-gp-and-nn

def new_orientation(angle, play_direction):

    if play_direction == 0:

        new_angle = 360.0 - angle

        if new_angle == 360.0:

            new_angle = 0.0

        return new_angle

    else:

        return angle

    

train['Orientation'] = train.apply(lambda row: new_orientation(row['Orientation'], row['PlayDirection']), axis=1)

train['Dir'] = train.apply(lambda row: new_orientation(row['Dir'], row['PlayDirection']), axis=1)
train['YardsLeft'] = train.apply(lambda row: 100-row['YardLine'] if row['HomeField'] else row['YardLine'], axis=1)

train['YardsLeft'] = train.apply(lambda row: row['YardsLeft'] if row['PlayDirection'] else 100-row['YardsLeft'], axis=1)
((train['YardsLeft']<train['Yards']) | (train['YardsLeft']-100>train['Yards'])).mean()
train.drop(train.index[(train['YardsLeft']<train['Yards']) | (train['YardsLeft']-100>train['Yards'])], inplace=True)
train = train.sort_values(by=['PlayId', 'Team', 'IsRusher', 'JerseyNumber']).reset_index()
train.drop(['GameId', 'PlayId', 'index', 'IsRusher', 'Team'], axis=1, inplace=True)
cat_features = []

for col in train.columns:

    if train[col].dtype =='object':

        cat_features.append(col)

        

train = train.drop(cat_features, axis=1)
train.fillna(-999, inplace=True)
players_col = []

for col in train.columns:

    if train[col][:22].std()!=0:

        players_col.append(col)
X_train = np.array(train[players_col]).reshape(-1, len(players_col)*22)
play_col = train.drop(players_col+['Yards'], axis=1).columns

X_play_col = np.zeros(shape=(X_train.shape[0], len(play_col)))

for i, col in enumerate(play_col):

    X_play_col[:, i] = train[col][::22]
X_train = np.concatenate([X_train, X_play_col], axis=1)

y_train = np.zeros(shape=(X_train.shape[0], 199))

for i,yard in enumerate(train['Yards'][::22]):

    y_train[i, yard+99:] = np.ones(shape=(1, 100-yard))
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
batch_size=64
#from https://www.kaggle.com/davidcairuz/nfl-neural-network-w-softmax

def crps(y_true, y_pred):

    return K.mean(K.square(y_true - K.cumsum(y_pred, axis=1)), axis=1)
#Create simple Keras model with one hidden layer with 10 nodes. 

def get_model():

    model = Sequential()

    model.add(Dense(10, activation='relu', input_shape=(X_train.shape[1],)))

    # compile model

    model.add(Dense(199, activation='softmax'))

    model.compile(optimizer='adam', loss=crps)

    return model





def train_model(X_train, y_train, X_val, y_val):

    model = get_model()

    er = EarlyStopping(patience=20, min_delta=1e-4, restore_best_weights=True, monitor='val_loss')

    model.fit(X_train, y_train, epochs=200, callbacks=[er], validation_data=[X_val, y_val])

    return model
from sklearn.model_selection import RepeatedKFold



rkf = RepeatedKFold(n_splits=5, n_repeats=5)



models = []



for tr_idx, vl_idx in rkf.split(X_train, y_train):

    

    x_tr, y_tr = X_train[tr_idx], y_train[tr_idx]

    x_vl, y_vl = X_train[vl_idx], y_train[vl_idx]

    

    model = train_model(x_tr, y_tr, x_vl, y_vl)

    models.append(model)
def make_pred(df, sample, env, models):

    df['StadiumType'] = df['StadiumType'].apply(clean_StadiumType)

    df['StadiumType'] = df['StadiumType'].apply(transform_StadiumType)

    df['DefendersInTheBox_vs_Distance'] = df['DefendersInTheBox'] / df['Distance']

    df['OffenseFormation'] = df['OffenseFormation'].apply(lambda x: x if x in off_form else np.nan)

    df = pd.concat([df.drop(['OffenseFormation'], axis=1), pd.get_dummies(df['OffenseFormation'], prefix='Formation')], axis=1)

    missing_cols = set( dummy_col ) - set( df.columns )-set('Yards')

    for c in missing_cols:

        df[c] = 0

    df = df[dummy_col]

    df.drop(['Yards'], axis=1, inplace=True)

    df['Turf'] = df['Turf'].map(Turf)

    df['Turf'] = df['Turf'] == 'Natural'

    df['PossessionTeam'] = df['PossessionTeam'].map(map_abbr)

    df['HomeTeamAbbr'] = df['HomeTeamAbbr'].map(map_abbr)

    df['VisitorTeamAbbr'] = df['VisitorTeamAbbr'].map(map_abbr)

    df['HomePossesion'] = df['PossessionTeam'] == df['HomeTeamAbbr']

    df['Field_eq_Possession'] = df['FieldPosition'] == df['PossessionTeam']

    df['HomeField'] = df['FieldPosition'] == df['HomeTeamAbbr']

    df['GameClock'] = df['GameClock'].apply(strtoseconds)

    df['PlayerHeight'] = df['PlayerHeight'].apply(lambda x: 12*int(x.split('-')[0])+int(x.split('-')[1]))

    df['PlayerBMI'] = 703*(df['PlayerWeight']/(df['PlayerHeight'])**2)

    df['TimeHandoff'] = df['TimeHandoff'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))

    df['TimeSnap'] = df['TimeSnap'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))

    df['TimeDelta'] = df.apply(lambda row: (row['TimeHandoff'] - row['TimeSnap']).total_seconds(), axis=1)

    df['PlayerBirthDate'] = df['PlayerBirthDate'].apply(lambda x: datetime.datetime.strptime(x, "%m/%d/%Y"))

    seconds_in_year = 60*60*24*365.25

    df['PlayerAge'] = df.apply(lambda row: (row['TimeHandoff']-row['PlayerBirthDate']).total_seconds()/seconds_in_year, axis=1)

    df['WindSpeed'] = df['WindSpeed'].apply(lambda x: x.lower().replace('mph', '').strip() if not pd.isna(x) else x)

    df['WindSpeed'] = df['WindSpeed'].apply(lambda x: (int(x.split('-')[0])+int(x.split('-')[1]))/2 if not pd.isna(x) and '-' in x else x)

    df['WindSpeed'] = df['WindSpeed'].apply(lambda x: (int(x.split()[0])+int(x.split()[-1]))/2 if not pd.isna(x) and type(x)!=float and 'gusts up to' in x else x)

    df['WindSpeed'] = df['WindSpeed'].apply(str_to_float)

    df['WindDirection'] = df['WindDirection'].apply(clean_WindDirection)

    df['WindDirection'] = df['WindDirection'].apply(transform_WindDirection)

    df['PlayDirection'] = df['PlayDirection'].apply(lambda x: x.strip() == 'right')

    df['Team'] = df['Team'].apply(lambda x: x.strip()=='home')

    indoor = "indoor"

    df['GameWeather'] = df['GameWeather'].apply(lambda x: indoor if not pd.isna(x) and indoor in x else x)

    df['GameWeather'] = df['GameWeather'].apply(lambda x: x.lower().replace('coudy', 'cloudy').replace('clouidy', 'cloudy').replace('party', 'partly').replace('clear and sunny', 'sunny and clear').replace('skies', '').replace("mostly", "").strip() if not pd.isna(x) else x)

    df['GameWeather'] = df['GameWeather'].apply(map_weather)

    df['IsRusher'] = df['NflId'] == df['NflIdRusher']

    df['X'] = df.apply(lambda row: row['X'] if row['PlayDirection'] else 120-row['X'], axis=1)

    df['Orientation'] = df.apply(lambda row: new_orientation(row['Orientation'], row['PlayDirection']), axis=1)

    df['Dir'] = df.apply(lambda row: new_orientation(row['Dir'], row['PlayDirection']), axis=1)

    df['YardsLeft'] = df.apply(lambda row: 100-row['YardLine'] if row['HomeField'] else row['YardLine'], axis=1)

    df['YardsLeft'] = df.apply(lambda row: row['YardsLeft'] if row['PlayDirection'] else 100-row['YardsLeft'], axis=1)

    df = df.sort_values(by=['PlayId', 'Team', 'IsRusher', 'JerseyNumber']).reset_index()

    df = df.drop(['TimeHandoff', 'TimeSnap', 'PlayerBirthDate', 'NflId', 'NflIdRusher', 'GameId', 'PlayId', 'index', 'IsRusher', 'Team'], axis=1)

    cat_features = []

    for col in df.columns:

        if df[col].dtype =='object':

            cat_features.append(col)



    df = df.drop(cat_features, axis=1)

    df.fillna(-999, inplace=True)

    X = np.array(df[players_col]).reshape(-1, len(players_col)*22)

    play_col = df.drop(players_col, axis=1).columns

    X_play_col = np.zeros(shape=(X.shape[0], len(play_col)))

    for i, col in enumerate(play_col):

        X_play_col[:, i] = df[col][::22]

    X = np.concatenate([X, X_play_col], axis=1)

    X = scaler.transform(X)

    y_pred = np.mean([np.cumsum(model.predict(X), axis=1) for model in models], axis=0)

    yardsleft = np.array(df['YardsLeft'][::22])

    

    for i in range(len(yardsleft)):

        y_pred[i, :yardsleft[i]-1] = 0

        y_pred[i, yardsleft[i]+100:] = 1

    env.predict(pd.DataFrame(data=y_pred.clip(0,1),columns=sample.columns))

    return y_pred
for test, sample in tqdm.tqdm(env.iter_test()):

     make_pred(test, sample, env, models)
env.write_submission_file()