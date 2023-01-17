import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib as mpl

import seaborn as sns

import datetime

from kaggle.competitions import nflrush

import tqdm

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import StandardScaler

import keras

import tensorflow as tf



sns.set_style('darkgrid')

mpl.rcParams['figure.figsize'] = [15,10]
env = nflrush.make_env()
train = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv', dtype={'WindSpeed': 'object'})
train.head()
train['PlayId'].value_counts()
train['Yards'].describe()
ax = sns.distplot(train['Yards'])

plt.vlines(train['Yards'].mean(), plt.ylim()[0], plt.ylim()[1], color='r', linestyles='--');

plt.text(train['Yards'].mean()-8, plt.ylim()[1]-0.005, "Mean yards travaled", size=15, color='r')

plt.xlabel("")

plt.title("Yards travaled distribution", size=20);
#from https://www.kaggle.com/prashantkikani/nfl-starter-lgb-feature-engg

train['DefendersInTheBox_vs_Distance'] = train['DefendersInTheBox'] / train['Distance']
cat_features = []

for col in train.columns:

    if train[col].dtype =='object':

        cat_features.append((col, len(train[col].unique())))
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
#let's replace the ones that has x-y by (x+y)/2

# and also the ones with x gusts up to y

#train['WindSpeed'] = train['WindSpeed'].apply(lambda x: (int(x.split('-')[0])+int(x.split('-')[1]))/2 if not pd.isna(x) and '-' in x else x)

#train['WindSpeed'] = train['WindSpeed'].apply(lambda x: (int(x.split()[0])+int(x.split()[-1]))/2 if not pd.isna(x) and type(x)!=float and 'gusts up to' in x else x)

def str_to_float(txt):

    try:

        return float(txt)

    except:

        return -1
train['WindSpeed'] = train['WindSpeed'].apply(str_to_float)
train['WindDirection'].value_counts()
train.drop('WindDirection', axis=1, inplace=True)
train['PlayDirection'].value_counts()
train['PlayDirection'] = train['PlayDirection'].apply(lambda x: x is 'right')
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
train['Y'].describe()
#train['TimeDelta'] = train.apply(lambda row: (row['TimeHandoff'] - row['TimeSnap']).total_seconds(), axis=1)

#train['GameWeather'] = train['GameWeather'].apply(lambda x: indoor if not pd.isna(x) and indoor in x else x)

#train['Delta_X_Y'] = train.apply(lambda row: (row['X'] + row['Y']), axis=1)

#from math import sqrt

#train['Delta_X_Y'] = train['Delta_X_Y'].apply(lambda x: sqrt(x) if x!=0 else 0)
train = train.sort_values(by=['PlayId', 'Team', 'IsRusher']).reset_index()
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
X_train = np.array(train[players_col]).reshape(-1, 11*22)
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
model = keras.models.Sequential([

    keras.layers.Dense(units=300, activation='relu', input_shape=[X_train.shape[1]]),

    keras.layers.BatchNormalization(),

    keras.layers.Dropout(0.15),

    keras.layers.Dense(units=260, activation='relu'),

    keras.layers.BatchNormalization(),

    keras.layers.Dropout(0.15),

    keras.layers.Dense(units=199, activation='sigmoid')

])



model.compile(optimizer='adam', loss='mse')

model.fit(X_train, y_train, epochs=200, batch_size=32)
def make_pred(df, sample, env, model):

    df['DefendersInTheBox_vs_Distance'] = df['DefendersInTheBox'] / df['Distance']

    df['OffenseFormation'] = df['OffenseFormation'].apply(lambda x: x if x in off_form else np.nan)

    df = pd.concat([df.drop(['OffenseFormation'], axis=1), pd.get_dummies(df['OffenseFormation'], prefix='Formation')], axis=1)

    missing_cols = set( dummy_col ) - set( test.columns )-set('Yards')

    for c in missing_cols:

        df[c] = 0

    df = df[dummy_col]

    df.drop(['Yards'], axis=1, inplace=True)

    df['PossessionTeam'] = df['PossessionTeam'].map(map_abbr)

    df['HomeTeamAbbr'] = df['HomeTeamAbbr'].map(map_abbr)

    df['VisitorTeamAbbr'] = df['VisitorTeamAbbr'].map(map_abbr)

    df['HomePossesion'] = df['PossessionTeam'] == df['HomeTeamAbbr']

    df['Field_eq_Possession'] = df['FieldPosition'] == df['PossessionTeam']

    df['GameClock'] = df['GameClock'].apply(strtoseconds)

    df['PlayerHeight'] = df['PlayerHeight'].apply(lambda x: 12*int(x.split('-')[0])+int(x.split('-')[1]))

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

    df['PlayDirection'] = train['PlayDirection'].apply(lambda x: x is 'right')

    df['Team'] = df['Team'].apply(lambda x: x.strip()=='home')

    indoor = "indoor"

    df['GameWeather'] = df['GameWeather'].apply(lambda x: indoor if not pd.isna(x) and indoor in x else x)

    df['GameWeather'] = df['GameWeather'].apply(lambda x: x.lower().replace('coudy', 'cloudy').replace('clouidy', 'cloudy').replace('party', 'partly').replace('clear and sunny', 'sunny and clear').replace('skies', '').replace("mostly", "").strip() if not pd.isna(x) else x)

    df['GameWeather'] = df['GameWeather'].apply(map_weather)

    df['IsRusher'] = df['NflId'] == df['NflIdRusher']

    

    df = df.sort_values(by=['PlayId', 'Team', 'IsRusher']).reset_index()

    df = df.drop(['TimeHandoff', 'TimeSnap', 'PlayerBirthDate', 'WindDirection', 'NflId', 'NflIdRusher', 'GameId', 'PlayId', 'index', 'IsRusher', 'Team'], axis=1)

    cat_features = []

    for col in df.columns:

        if df[col].dtype =='object':

            cat_features.append(col)



    df = df.drop(cat_features, axis=1)

    df.fillna(-999, inplace=True)

    X = np.array(df[players_col]).reshape(-1, 11*22)

    play_col = df.drop(players_col, axis=1).columns

    X_play_col = np.zeros(shape=(X.shape[0], len(play_col)))

    for i, col in enumerate(play_col):

        X_play_col[:, i] = df[col][::22]

    X = np.concatenate([X, X_play_col], axis=1)

    X = scaler.transform(X)

    y_pred = model.predict(X)

    for pred in y_pred:

        prev = 0

        for i in range(len(pred)):

            if pred[i]<prev:

                pred[i]=prev

            prev=pred[i]

    y_pred[:, -1] = np.ones(shape=(y_pred.shape[0], 1))

    y_pred[:, 0] = np.zeros(shape=(y_pred.shape[0], 1))

    env.predict(pd.DataFrame(data=y_pred,columns=sample.columns))

    return y_pred
for test, sample in tqdm.tqdm(env.iter_test()):

    make_pred(test, sample, env, model)
env.write_submission_file()