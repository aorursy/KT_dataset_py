# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datetime

from tqdm.auto import tqdm



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#NN layer node counts

LAYER1_SIZE = 50

LAYER2_SIZE = 512

TEAM_LAYER = True

EPOCHS=30

REPEATS=3

df_train = pd.read_csv("/kaggle/input/nfl-big-data-bowl-2020/train.csv")

yards = df_train["Yards"]
play_groups = df_train.groupby("PlayId")

sizes = play_groups.size()
sizes.describe()
#this simply assumes the home team abbreviation and possion team abbreviation are matching in number and in alphabetical order

#then compares where the lines differ to find mismatching use of abbreviations

for x,y  in zip(sorted(df_train['HomeTeamAbbr'].unique()), sorted(df_train['PossessionTeam'].unique())):

    if x!=y:

        print(x + " " + y)
#this creates a mapping of the mismatchin abbreviations above

map_abbr = {'ARI': 'ARZ', 'BAL': 'BLT', 'CLE': 'CLV', 'HOU': 'HST'}

#and this adds all the rest of abbreviations from PossessionTeam, mapping their abbreviations to themselves

#this dos not override above map initialization, since it only has the ones missing from PossessionTeam (as keys)

for abb in df_train['PossessionTeam'].unique():

    map_abbr[abb] = abb



for abb in sorted(map_abbr.keys()):

    print(f"{abb}={map_abbr[abb]}, ", end="")
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
off_form = df_train['OffenseFormation'].unique()

off_form
def str_to_float(txt):

    try:

        return float(txt)

    except:

        return -1
def map_turf(txt):

    txt = txt.lower()

    words = txt.split(" ")

    if "grass" in words or "natural" in words:

        return "natural"

    return "artificial"


def create_features(df, dummy_cols=None, show_tqdm=True):

    if show_tqdm:

        pbar = tqdm(total=21)

    #defenders in the box = defenders near the line of "scrimmage" (scrimmage seems to be line parallel to goal line, at the ball)

    #distance = yards needed for first down

    #so what is the point of this feature? the closer you are to getting the yards, the less you want defenders at ball?

    df['DefendersInTheBox_vs_Distance'] = df['DefendersInTheBox'] / df['Distance']

    if show_tqdm:

        pbar.update()

    #this seems to be just making sure the OffenseFormation only has values that were already present in the training set

    df['OffenseFormation'] = df['OffenseFormation'].apply(lambda x: x if x in off_form else np.nan)

    if show_tqdm:

        pbar.update()

    #OffenseFormation: one-hot encode and drop the original. needed for most classifiers, including keras used later here

    df = pd.concat([df.drop(['OffenseFormation'], axis=1), pd.get_dummies(df['OffenseFormation'], prefix='Formation')], axis=1)

    if show_tqdm:

        pbar.update()

    #Position: one-hot encode and drop the original. needed for most classifiers, including keras used later here

    #https://en.wikipedia.org/wiki/American_football_positions



    df = pd.concat([df.drop(['Position'], axis=1), pd.get_dummies(df['Position'], prefix='Position')], axis=1)

    if show_tqdm:

        pbar.update()

    if dummy_cols is not None:

        missing_cols = set( dummy_cols ) - set( df.columns ) - set('Yards')

        for c in missing_cols:

            df[c] = 0

        df = df[dummy_cols]

    

    #unify all team abbreviations used in different columns

    df['PossessionTeam'] = df['PossessionTeam'].map(map_abbr)

    if show_tqdm:

        pbar.update()

    df['HomeTeamAbbr'] = df['HomeTeamAbbr'].map(map_abbr)

    if show_tqdm:

        pbar.update()

    df['VisitorTeamAbbr'] = df['VisitorTeamAbbr'].map(map_abbr)

    if show_tqdm:

        pbar.update()

    #is the team in possession of the ball in this play the home team?

    df['HomePossesion'] = df['PossessionTeam'] == df['HomeTeamAbbr']

    if show_tqdm:

        pbar.update()

    #Field_eq_Possession = is the ball on the field side of the possessing team

    df['Field_eq_Possession'] = df['FieldPosition'] == df['PossessionTeam']

    if show_tqdm:

        pbar.update()

    #convert the game clock string with hours, minutes, seconds into a single number of seconds

    df['GameClock'] = df['GameClock'].apply(strtoseconds)

    if show_tqdm:

        pbar.update()

    #convert height from foots-inches notation to inches only (or a single number anyway)

    df['PlayerHeight'] = df['PlayerHeight'].apply(lambda x: 12*int(x.split('-')[0])+int(x.split('-')[1]))

    if show_tqdm:

        pbar.update()

    #convert handoff and snap times to datetime format so one can calculate their diff

    df['TimeHandoff'] = df['TimeHandoff'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))

    if show_tqdm:

        pbar.update()

    df['TimeSnap'] = df['TimeSnap'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))

    if show_tqdm:

        pbar.update()

    #calculate the delta (diff) from snap (picking up the ball) to handoff (giving it to next snapper)

    df['TimeDelta'] = df.apply(lambda row: (row['TimeHandoff'] - row['TimeSnap']).total_seconds(), axis=1)

    if show_tqdm:

        pbar.update()

    #convert player birthdate into datetime format to calculate age

    df['PlayerBirthDate'] = df['PlayerBirthDate'].apply(lambda x: datetime.datetime.strptime(x, "%m/%d/%Y"))

    if show_tqdm:

        pbar.update()

    seconds_in_year = 60*60*24*365.25

    #calculate player age at the time of handoff, float in years

    df['PlayerAge'] = df.apply(lambda row: (row['TimeHandoff']-row['PlayerBirthDate']).total_seconds()/seconds_in_year, axis=1)

    if show_tqdm:

        pbar.update()

    #clean up windspeed from all the garbage strings, convert to float

    df['WindSpeed'] = df['WindSpeed'].astype("str").apply(lambda x: x.lower().replace('mph', '').strip() if not pd.isna(x) else x)

    df['WindSpeed'] = df['WindSpeed'].apply(lambda x: (int(x.split('-')[0])+int(x.split('-')[1]))/2 if not pd.isna(x) and '-' in x else x)

    df['WindSpeed'] = df['WindSpeed'].apply(lambda x: (int(x.split()[0])+int(x.split()[-1]))/2 if not pd.isna(x) and type(x)!=float and 'gusts up to' in x else x)

    df['WindSpeed'] = df['WindSpeed'].apply(str_to_float)

    if show_tqdm:

        pbar.update()

    #is play going left or right on the field? convert to boolean

    df['PlayDirection'] = df['PlayDirection'].apply(lambda x: x is 'right')

    if show_tqdm:

        pbar.update()

    #perhaps "team" defines if the team whose turn is to "play" is the home team or not?

    df['Team'] = df['Team'].apply(lambda x: x.strip()=='home')

    if show_tqdm:

        pbar.update()

    df["Turf"] = df["Turf"].map(map_turf)

    if show_tqdm:

        pbar.update()

    indoor = "indoor"

    #replace variants of indoor text in GameWeather with just "indoor"

    df['GameWeather'] = df['GameWeather'].apply(lambda x: indoor if not pd.isna(x) and indoor in x else x)

    #seems to be fixing typos

    df['GameWeather'] = df['GameWeather'].apply(lambda x: x.lower().replace('coudy', 'cloudy').replace('clouidy', 'cloudy').replace('party', 'partly').replace('clear and sunny', 'sunny and clear').replace('skies', '').replace("mostly", "").strip() if not pd.isna(x) else x)

    #convert weather into a number between -3 to 3, where -3 is snow, 2 is sunny and 3 is indoor climate control

    df['GameWeather'] = df['GameWeather'].apply(map_weather)

    if show_tqdm:

        pbar.update()

    #mark the rushing player

    df['IsRusher'] = df['NflId'] == df['NflIdRusher']

    if show_tqdm:

        pbar.update()

    return df



df = df_train

df.shape
df = df_train

#drop the target variable

df.drop(['Yards'], axis=1, inplace=True)

df = create_features(df)

#order by play id, team type, and rusher last

df = df.sort_values(by=['PlayId', 'Team', 'IsRusher']).reset_index()

df.shape
all_cols = df.columns
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()


df = df.drop(['TimeHandoff', 'TimeSnap', 'PlayerBirthDate', 'WindDirection', 'NflId', 'NflIdRusher', 'GameId', 'PlayId', 'index', 'Team'], axis=1)

cat_features = []

for col in df.columns:

    if df[col].dtype =='object':

        cat_features.append(col)

df = df.drop(cat_features, axis=1)



df.fillna(-999, inplace=True)

X = df.values

X = scaler.fit_transform(X)

X.shape
X_train = X

X_slices = []

for x in range(0,22):

    player_slice = X_train[x::22]

    X_slices.append(player_slice)

X_train = X_slices

X_train[0].shape
y_train = np.zeros(shape=(X_train[0].shape[0], 199))

for i,yard in enumerate(yards[::22]):

    y_train[i, yard+99:] = np.ones(shape=(1, 100-yard))
len(X_train)
import keras

from keras.callbacks import EarlyStopping

from keras.optimizers import Adam

from keras.models import Model

import tensorflow as tf

from keras.layers import Input, Dense, BatchNormalization, Dropout, concatenate



def create_model(team_layer, player_feature_count):

    player_layers = []

    player_inputs = []

    for x in range(1,23):

        player_input = Input(shape=(player_feature_count,), name=f"in{x}")

        if not team_layer or True: #the OR TRUE part avoids the other branch as I just wanted to give it a go this way

            player_layer = Dense(units=LAYER1_SIZE, activation='relu', name=f"d{x}")(player_input)

            player_layer = BatchNormalization(name=f"bn{x}")(player_layer)

            player_layers.append(player_layer)

        else:

            player_layers.append(player_input)

        player_inputs.append(player_input)



    if team_layer:

        layer_size = int(LAYER2_SIZE/2)

        

        team1_layer = concatenate(player_layers[:11], name="team1")

        team1_layer = Dense(units=layer_size, activation='relu', name="t1_dense")(team1_layer)

        team1_layer = BatchNormalization(name="t1_bn")(team1_layer)

        

        team2_layer = concatenate(player_layers[11:], name="team2")

        team2_layer = Dense(units=layer_size, activation='relu', name="t2_dense")(team2_layer)

        team2_layer = BatchNormalization(name="t2_bn")(team2_layer)

        

        combined = concatenate([team1_layer, team2_layer], name="all_merge")

    else:

        combined = concatenate(player_layers, name="all_merge")



    mid = Dropout(0.3, name="mid_drop")(combined)

    mid = Dense(units=LAYER2_SIZE, activation='relu', name="mid_dense")(mid)

    mid = BatchNormalization(name="final_bn")(mid)

    mid = Dropout(0.2, name="final_drop")(mid)

    output=keras.layers.Dense(units=199, activation='sigmoid', name="output")(mid)



    model = Model(inputs=player_inputs, outputs=output)

    return model

#https://keras.io/visualization/

from IPython.display import SVG

from keras.utils import model_to_dot



model = create_model(True, df_train.shape[1])

#print(model.summary())

SVG(model_to_dot(model, dpi=48, rankdir="LR").create(prog='dot', format='svg'))
model = create_model(False, df_train.shape[1])

#print(model.summary())

SVG(model_to_dot(model, dpi=48, rankdir="LR").create(prog='dot', format='svg'))
def train_model(x_tr, y_tr, x_vl, y_vl):

    player_feature_count = x_tr[0].shape[1]



    model = create_model(TEAM_LAYER, player_feature_count)



    er = EarlyStopping(patience=5, min_delta=1e-4, restore_best_weights=True, monitor='val_loss')

    metric = None



    loss = "mse"

    model.compile(optimizer=Adam(lr=0.0005), loss=loss)



    model.fit(x_tr, y_tr, epochs=EPOCHS, batch_size=32, callbacks=[er], validation_data=[x_vl, y_vl])

    return model
from sklearn.model_selection import RepeatedKFold



rkf = RepeatedKFold(n_splits=5, n_repeats=REPEATS)
len(y_train)
models = []



for tr_idx, vl_idx in rkf.split(X_train[0], y_train):

    

    x_tr = []

    for player in X_train:

        x_tr.append(player[tr_idx])

    y_tr = y_train[tr_idx]



    x_vl = []

    for player in X_train:

        x_vl.append(player[vl_idx])

    y_vl = y_train[vl_idx]

    

    model = train_model(x_tr, y_tr, x_vl, y_vl)

    models.append(model)
def make_pred(X):

    player_feature_count = X.shape[1]

    inputs = np.hsplit(X.reshape(-1, player_feature_count*22), 22)

    y_pred = np.mean([model.predict(inputs) for model in models], axis=0)

    for pred in y_pred:

        prev = 0

        for i in range(len(pred)):

            if pred[i]<prev:

                pred[i]=prev

            prev=pred[i]

    y_pred[:, -1] = np.ones(shape=(y_pred.shape[0]))

    y_pred[:, 0] = np.zeros(shape=(y_pred.shape[0]))

    return y_pred

y_pred = make_pred(X)

y_pred
def preprocess_test(df):

    #order by play id, team, and rusher last

    df = df.sort_values(by=['PlayId', 'Team', 'IsRusher']).reset_index()

    df = df.drop(['TimeHandoff', 'TimeSnap', 'PlayerBirthDate', 'WindDirection', 'NflId', 'NflIdRusher', 'GameId', 'PlayId', 'index', 'Team'], axis=1)

    df = df.drop(['level_0'], axis=1)

    df = df.drop(cat_features, axis=1)



    df.fillna(-999, inplace=True)

    X = df.values

    X = scaler.transform(X)

    return X
from kaggle.competitions import nflrush



env = nflrush.make_env()
for test, sample in tqdm(env.iter_test()):

    #add new features, no scaling or anything yet

    df_test = create_features(test, all_cols, False)

    X = preprocess_test(df_test)

    y_pred = make_pred(X)

    env.predict(pd.DataFrame(data=y_pred,columns=sample.columns))

env.write_submission_file()
 