import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import datetime, tqdm

from kaggle.competitions import nflrush

from sklearn.model_selection import KFold, RepeatedKFold
env = nflrush.make_env()
train = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv', dtype={'WindSpeed': 'object'})

train.head()
# from https://www.kaggle.com/tunguz/adversarial-nfl

train.drop(['Season', 'DefensePersonnel', 'Temperature', 'Humidity'], axis = 1, inplace = True)
sns.distplot(train['Yards'])

plt.show()
cat_features = []

for col in train.columns:

    if train[col].dtype =='object':

        cat_features.append((col, len(train[col].unique())))
off_form = train['OffenseFormation'].unique()

train = pd.concat([train.drop(['OffenseFormation'], axis=1), pd.get_dummies(train['OffenseFormation'], prefix='Formation')], axis=1)

dummy_col = train.columns
# Some fetures borrowed from https://www.kaggle.com/bgmello/neural-networks-feature-engineering-for-the-win and https://www.kaggle.com/prashantkikani/nfl-starter-lgb-feature-engg

# Refactored the code for readability



def fe(train):

#     train['X1'] = 120 - train['X']

#     train['Y1'] = 53.3 - train['Y']

    train['DefendersInTheBox_vs_Distance'] = train['DefendersInTheBox'] / train['Distance']

    

    def strtoseconds(txt):

        txt = txt.split(':')

        return int(txt[0])*60 + int(txt[1]) + int(txt[2])/60

    

    # from https://www.kaggle.com/c/nfl-big-data-bowl-2020/discussion/112173#latest-647309

    train['JerseyNumber_grouped'] = train['JerseyNumber'] // 10

    

    train['GameClock'] = train['GameClock'].apply(strtoseconds)

    train['Height_feet'] = train['PlayerHeight'].apply(lambda x: int(x.split('-')[0]))

#     train['Height_cm'] = train['PlayerHeight'].apply(lambda x: int(x.split('-')[1]))

    train['PlayerHeight'] = train['PlayerHeight'].apply(lambda x: 12*int(x.split('-')[0])+int(x.split('-')[1]))

    train['BMI'] = (train['PlayerWeight'] * 703) / (train['PlayerHeight'] ** 2)

    

#     arr = [[int(s[0]) for s in t.split(", ")] for t in train["DefensePersonnel"]]

#     train['DL'] = [a[0] for a in arr]

#     train['LB'] = [a[1] for a in arr]

#     train['DB'] = [a[2] for a in arr]

    

#     arr = [[int(s[0]) for s in t.split(", ")] for t in train["OffensePersonnel"]]

#     train['RB'] = [a[0] for a in arr]

#     train['TE'] = [a[1] for a in arr]

#     train['WR'] = [a[2] for a in arr]

    

    train['TimeHandoff'] = train['TimeHandoff'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))

    train['TimeSnap'] = train['TimeSnap'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))

    train['TimeDelta'] = train.apply(lambda row: (row['TimeHandoff'] - row['TimeSnap']).total_seconds(), axis=1)

#     train['Birth_year'] = train['PlayerBirthDate'].apply(lambda x: int(x.split('/')[2]))

    train['PlayerBirthDate'] = train['PlayerBirthDate'].apply(lambda x: datetime.datetime.strptime(x, "%m/%d/%Y"))

    

    seconds_in_year = 60*60*24*365

#     train['PlayerAge'] = train.apply(lambda row: (row['TimeHandoff']-row['PlayerBirthDate']).total_seconds()/seconds_in_year, axis=1)

    train = train.drop(['TimeHandoff', 'TimeSnap', 'PlayerBirthDate'], axis=1)

    

    def give_me_WindSpeed(txt):

        txt = str(txt).lower().replace('mph', '').strip()

        if '-' in txt:

            txt = (int(txt.split('-')[0]) + int(txt.split('-')[1])) / 2

        try:

            return float(txt)

        except:

            return -1

        

    train['WindSpeed'] = train['WindSpeed'].apply(give_me_WindSpeed)

    

    train['PlayDirection'] = train['PlayDirection'].apply(lambda x: x is 'right')

    train['Team'] = train['Team'].apply(lambda x: x.strip()=='home')

    

    # from https://www.kaggle.com/ryches/model-free-benchmark

    train['Field_eq_Possession'] = train['FieldPosition'] == train['PossessionTeam']

    

    def give_me_GameWeather(txt):

        txt = str(txt).lower().replace('coudy', 'cloudy').replace('clouidy', 'cloudy').replace('party', 'partly').replace('clear and sunny', 'sunny and clear')

        txt = txt.replace('skies', '').replace("mostly", "").strip()

        if "indoor" in txt:

            txt = "indoor"

        ans = 1

        if pd.isna(txt):

            return 0

        if 'partly' in txt:

            ans*=0.5

        if 'climate controlled' in txt or 'indoor' in txt:

            return ans*5

        if 'sunny' in txt or 'sun' in txt:

            return ans*3

        if 'clear' in txt:

            return ans

        if 'cloudy' in txt:

            return -ans

        if 'rain' in txt or 'rainy' in txt:

            return -3*ans

        if 'snow' in txt:

            return -5*ans

        return 0



    train['GameWeather'] = train['GameWeather'].apply(give_me_GameWeather)

    train['IsRusher'] = train['NflId'] == train['NflIdRusher']

    train = train.sort_values(by=['PlayId', 'Team', 'IsRusher']).reset_index()

    train.drop(['GameId', 'PlayId', 'index', 'IsRusher', 'Team', 'NflId', 'NflIdRusher'], axis=1, inplace=True)

    

    cat_features = []

    for col in train.columns:

        if train[col].dtype =='object':

            cat_features.append(col)



    train = train.drop(cat_features, axis=1)

    train.fillna(-999, inplace=True)

    return train
train = fe(train)
players_col = []

for col in train.columns:

    if train[col][:22].std()!=0:

        players_col.append(col)
X_train = np.array(train[players_col]).reshape(-1, (len(players_col))*22)

play_col = train.drop(players_col+['Yards'], axis=1).columns

X_play_col = np.zeros(shape=(X_train.shape[0], len(play_col)))

for i, col in enumerate(play_col):

    X_play_col[:, i] = train[col][::22]
X_train = np.concatenate([X_train, X_play_col], axis=1)

y_train = np.zeros(shape=(X_train.shape[0], 199))

for i,yard in enumerate(train['Yards'][::22]):

    y_train[i, yard+99:] = np.ones(shape=(1, 100-yard))
from keras.models import Sequential, Model

from keras.layers import Input, Dense, Embedding, Concatenate, Flatten, BatchNormalization, Dropout

from keras.callbacks import ModelCheckpoint

from keras.utils.vis_utils import plot_model
folds = 10

seed = 1997

kf = KFold(n_splits = folds, shuffle = True, random_state=seed)

# y_valid_pred = np.zeros((X_train.shape[0], 199))

models_1, models_2 = [], []



def give_me_model_1():

    numerical_inputs = Input(shape=(X_train.shape[1],)) 

    x = Dense(X_train.shape[1], activation='relu')(numerical_inputs)

    x = BatchNormalization()(x)

    

    logits1 = Dense(512,activation='relu')(x)

    logits = Concatenate()([x, logits1])

    logits = Dropout(0.3)(logits)

    

    logits1 = Dense(512,activation='relu')(logits)

    logits = Concatenate()([logits, logits1])

    logits = Dropout(0.2)(logits)

    

    logits1 = Dense(256,activation='relu')(logits)

    logits1 = Concatenate()([logits, logits1])

    logits1 = Dropout(0.1)(logits1)

    

#     logits1 = Dense(256,activation='relu')(logits1)

    logits1 = Dense(256,activation='relu')(logits1)

    out = Dense(199, activation='sigmoid')(logits1)

    

    model = Model(inputs = numerical_inputs, outputs=out)

    model.compile(optimizer='adam',loss='mse')

    return model



def give_me_model_2():

    numerical_inputs = Input(shape=(X_train.shape[1],)) 

    x = Dense(X_train.shape[1], activation='relu')(numerical_inputs)

    x = BatchNormalization()(x)

    

    logits = Dense(512,activation='relu')(x)

    logits = Dropout(0.3)(logits)

    

    logits = Dense(256,activation='relu')(logits)

    logits = Dropout(0.2)(logits)

    

    logits = Dense(256,activation='relu')(logits)

    out = Dense(199, activation='sigmoid')(logits)

    

    model = Model(inputs = numerical_inputs, outputs=out)

    model.compile(optimizer='adam',loss='mse')

    return model



for tr_idx, val_idx in kf.split(X_train, y_train):

    tr_x, tr_y = X_train[tr_idx,:], y_train[tr_idx]

    vl_x, vl_y = X_train[val_idx,:], y_train[val_idx]



    checkpoint_1 = ModelCheckpoint('keras_model_1.hdf5', monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')

    callbacks_list_1 = [checkpoint_1]

    

    checkpoint_2 = ModelCheckpoint('keras_model_2.hdf5', monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')

    callbacks_list_2 = [checkpoint_2]

    

    model_1 = give_me_model_1()

    model_2 = give_me_model_2()

    

    model_1.fit(tr_x, tr_y, 

              epochs=10,

              verbose=0,

              callbacks=callbacks_list_1,

              validation_data=(vl_x, vl_y))

    

    model_2.fit(tr_x, tr_y, 

              epochs=10,

              verbose=0,

              callbacks=callbacks_list_2,

              validation_data=(vl_x, vl_y))

    

    model_1.load_weights('keras_model_1.hdf5')

    model_2.load_weights('keras_model_2.hdf5')

    

    models_1.append(model_1)

    models_2.append(model_2)
# y_ans = np.zeros((509762//22,199))



# for i,p in enumerate([yard for i,yard in enumerate(train['Yards'][::22])]):

#     for j in range(199):

#         if j>=p:

#             y_ans[i][j]=1.0



# print("validation score:",np.sum(np.power(y_valid_pred-y_ans,2))/(199*(509762//22)))
plot_model(model_1, to_file='model_plot_1.png', show_shapes=True)
plot_model(model_2, to_file='model_plot_2.png', show_shapes=True)
for df, sample in tqdm.tqdm(env.iter_test()):

    df.drop(['Season', 'DefensePersonnel', 'Temperature', 'Humidity'], axis = 1, inplace = True)

    df['OffenseFormation'] = df['OffenseFormation'].apply(lambda x: x if x in off_form else np.nan)

    df = pd.concat([df.drop(['OffenseFormation'], axis=1), pd.get_dummies(df['OffenseFormation'], prefix='Formation')], axis=1)

    missing_cols = set( dummy_col ) - set( df.columns )-set('Yards')

    for c in missing_cols:

        df[c] = 0

    df = df[dummy_col]

    df.drop(['Yards'], axis=1, inplace=True)

    df = fe(df)

    X = np.array(df[players_col]).reshape(-1, (len(players_col))*22)

    play_col = df.drop(players_col, axis=1).columns

    X_play_col = np.zeros(shape=(X.shape[0], len(play_col)))

    for i, col in enumerate(play_col):

        X_play_col[:, i] = df[col][::22]

    X = np.concatenate([X, X_play_col], axis=1)

    

    y_pred_1 = np.sum([model.predict(X) for model in models_1], axis=0) / kf.n_splits

    y_pred_2 = np.sum([model.predict(X) for model in models_2], axis=0) / kf.n_splits

    y_pred = y_pred_1 * 0.5 + y_pred_2 * 0.5

    

    for pred in y_pred:

        prev = 0

        for i in range(len(pred)):

            if pred[i]<prev:

                pred[i]=prev

            prev=pred[i]

    env.predict(pd.DataFrame(data=y_pred,columns=sample.columns))

    

env.write_submission_file()