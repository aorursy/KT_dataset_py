import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
from kaggle.competitions import nflrush

import pandas as pd

import numpy as np

import xgboost as xgb

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.model_selection import TimeSeriesSplit

from time import time

import pickle

import datetime as dt

from imblearn.over_sampling import RandomOverSampler

from scipy.sparse import csc_matrix

import matplotlib.pyplot as plt



pd.set_option('display.max_rows', 200)

pd.set_option('display.max_columns', 200)



train = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)

train_df = train.copy()
#train_df[:100]

#train_df.dtypes
drop_columns_train = [

                'GameId',

                'PlayId',

                'NflId',

                'DisplayName',

                'JerseyNumber',

                'GameClock',

                'PossessionTeam',

                'FieldPosition',

                'NflIdRusher',

                'OffenseFormation',

                'OffensePersonnel',

                'DefensePersonnel',

                'TimeHandoff',

                'TimeSnap',

                'PlayerHeight',

                'PlayerBirthDate',

                'PlayerCollegeName',

                'HomeTeamAbbr',

                'VisitorTeamAbbr',

                'Stadium',

                'Location',

                'Turf',

                'GameWeather',

                'StadiumType',

                'WindDirection',

                'WindSpeed',

    

                'Yards'   # TARGET

          

                ]



drop_columns_test = [

                'GameId',

                'PlayId',

                'NflId',

                'DisplayName',

                'JerseyNumber',

                'GameClock',

                'PossessionTeam',

                'FieldPosition',

                'NflIdRusher',

                'OffenseFormation',

                'OffensePersonnel',

                'DefensePersonnel',

                'TimeHandoff',

                'TimeSnap',

                'PlayerHeight',

                'PlayerBirthDate',

                'PlayerCollegeName',

                'HomeTeamAbbr',

                'VisitorTeamAbbr',

                'Stadium',

                'Location',

                'Turf',

                'GameWeather',

                'StadiumType',

                'WindDirection',

                'WindSpeed',

    

          

                ]
from sklearn import preprocessing

lbl = preprocessing.LabelEncoder()



def preprocessing(data, drop_columns):

    

    data = data.fillna(-999)

    

    month_list = [int(str(A)[4:6]) for A in data['GameId']]

    day_list = [int(str(A)[6:8]) for A in data['GameId']]

    game_list = [int(str(A)[-2:]) for A in data['GameId']]

    GameClock_list = [int(A[:2])*60 + int(A[3:5]) for A in data['GameClock']]

    age_list = [2020-int(str(A)[-4:]) for A in data['PlayerBirthDate']]

    

    data['month'] = month_list

    data['day'] = day_list

    data['game'] = game_list

    data['GameTimeRemain'] = GameClock_list

    data['PlayerAge'] = age_list

    

    height_list = []

    for A in data['PlayerHeight']:

        if len(A) == 3:

            height = round((int(A[0])*30.48+int(A[2])*2.54),3)

        else:

            height = round((int(A[0])*30.48+int(A[-2:])*2.54),3)

        height_list.append(height)

    

    data['PlayerHeight_cm'] = height_list

  

    data = data.drop(columns= drop_columns)

    

    for f in data.columns:

        if data[f].dtype=='object':

            lbl.fit(data[f].values)

            data[f] = lbl.transform(data[f].values)

    

    data['DefendersInTheBox'] =  data['DefendersInTheBox'].astype('int64')

    data['Temperature'] =  data['Temperature'].astype('int64')

    data['Humidity'] =  data['Humidity'].astype('int64')



    return data

    



y = train_df.Yards.values

target = y[np.arange(0, len(train_df), 22)]

train_df_processed = preprocessing(train_df,drop_columns_train)

train_df_processed = train_df_processed.iloc[np.arange(0, len(train_df_processed), 22)]
train_df_processed 
X_train, X_valid, y_train, y_valid = train_test_split(train_df_processed,target, test_size = 0.1,random_state = 666)
clf = xgb.XGBRegressor(

                            

                        n_estimators=200,

                        min_child_weight = 1.5,

                        max_depth=3,

                        verbosity = 1,

                        n_jobs=8,                                              

                        scale_pos_weight=1.025,

                        tree_method='exact',

                        objective = 'reg:squarederror',

                        predictor='cpu_predictor',

                        colsample_bytree = 0.66,

                        subsample = 1,

                        gamma = 0,

                        learning_rate=0.15,

                        num_parallel_tree = 1 

                       )





clf.fit(X_train, y_train, eval_metric="rmse", early_stopping_rounds=50,

                    eval_set=[(X_train, y_train), (X_valid, y_valid)],verbose=True)
env = nflrush.make_env()

iter_test = env.iter_test()
batch_no = 0

for (test_df, sample_prediction_df) in iter_test:

    

    print(f'Predicting Batch Number {batch_no}')

    test_df_processed = preprocessing(test_df, drop_columns_test)

    y_pred = clf.predict(test_df_processed)

    y_pred_int = np.array([int(round(A,0)) for A in y_pred])

    pred_df = np.zeros((22, 199))

    

    for A in range(len(y_pred_int)):

        pred_df[A][y_pred_int[A]+99:] = 1

        

    pred_df_mean = pred_df.mean(axis = 0)

    pred_df_mean = pred_df_mean.reshape(1, -1)

    

    final_pred_df = pd.DataFrame(data=pred_df_mean, columns=sample_prediction_df.columns)



    env.predict(final_pred_df)



    batch_no += 1



env.write_submission_file()

'''

yards = train_df.Yards

y = np.zeros((yards.shape[0], 199))

for idx, target in enumerate(list(yards)):

    y[idx][(99 + target):] = 1

    

train_df_processed = preprocessing(train_df,drop_columns_train)

train_df_processed = train_df_processed.groupby(['PlayId']).agg('mean').reset_index()

target = train_df_processed.Yards.values

train_df_processed = train_df_processed.drop(columns= ['Yards','PlayId'])

'''