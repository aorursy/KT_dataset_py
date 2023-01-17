import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import itertools

import gc

import os

import sys



from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error,mean_absolute_error

from sklearn.ensemble import RandomForestRegressor

from timeit import default_timer as timer



import lightgbm as lgb



from keras.models import Sequential

from keras.layers import Dense

from keras.callbacks import ModelCheckpoint
# Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage

def reduce_mem_usage(df):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.

    """

    start_mem = df.memory_usage().sum() / 1024**2

    

    for col in df.columns:

        col_type = df[col].dtype

        

        if col_type != object:

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

                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB --> {:.2f} MB (Decreased by {:.1f}%)'.format(

        start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
def state(message,start = True, time = 0):

    if(start):

        print(f'Working on {message} ... ')

    else :

        print(f'Working on {message} took ({round(time , 3)}) Sec \n')
# Import dataset

df_train = pd.read_csv('../input/pubg-finish-placement-prediction/train_V2.csv')

df_test = pd.read_csv('../input/pubg-finish-placement-prediction/test_V2.csv')



# Reduce memory use

df_train=reduce_mem_usage(df_train)

df_test=reduce_mem_usage(df_test)



# Show some data

df_train.head()

df_train.describe()
# # Reference link: https://www.kaggle.com/melonaded/a-beginner-guide-to-top-35-lasso-rf-lgbm



# # Drop features

# df_train = df_train.drop(['longestKill', 'rankPoints', 'numGroups'], axis=1)

# df_test = df_test.drop(['longestKill', 'rankPoints', 'numGroups'], axis=1)



# # Check row with NaN value

# df_train[df_train['winPlacePerc'].isnull()]

# # Drop row with NaN 'winPlacePerc' value

# df_train.drop(2744604, inplace=True)



# df_train['kills'].value_counts()

# df_train['DBNOs'].value_counts()

# df_train['weaponsAcquired'].value_counts()
# # Reference link: https://www.kaggle.com/deffro/eda-is-fun

# # Reference link: https://www.kaggle.com/carlolepelaars/pubg-data-exploration-rf-funny-gifs



# # Get alldata for feature engineering

# all_data = df_train.append(df_test, sort=False).reset_index(drop=True)



# # Map the matchType

# all_data['matchType'] = all_data['matchType'].map({

#     'crashfpp':1,

#     'crashtpp':2,

#     'duo':3,

#     'duo-fpp':4,

#     'flarefpp':5,

#     'flaretpp':6,

#     'normal-duo':7,

#     'normal-duo-fpp':8,

#     'normal-solo':9,

#     'normal-solo-fpp':10,

#     'normal-squad':11,

#     'normal-squad-fpp':12,

#     'solo':13,

#     'solo-fpp':14,

#     'squad':15,

#     'squad-fpp':16

#     })



# # Normalize features

# all_data['playersJoined'] = all_data.groupby('matchId')['matchId'].transform('count')

# all_data['killsNorm'] = all_data['kills']*((100-all_data['playersJoined'])/100 + 1)

# all_data['damageDealtNorm'] = all_data['damageDealt']*((100-all_data['playersJoined'])/100 + 1)

# all_data['maxPlaceNorm'] = all_data['maxPlace']*((100-all_data['playersJoined'])/100 + 1)

# all_data['matchDurationNorm'] = all_data['matchDuration']*((100-all_data['playersJoined'])/100 + 1)



# all_data['healsandboosts'] = all_data['heals'] + all_data['boosts']

# all_data['totalDistance'] = all_data['rideDistance'] + all_data['walkDistance'] + all_data['swimDistance']

# all_data['killsWithoutMoving'] = ((all_data['kills'] > 0) & (all_data['totalDistance'] == 0))



# all_data=reduce_mem_usage(all_data)



# # Split the train and the test

# df_train = all_data[all_data['winPlacePerc'].notnull()].reset_index(drop=True)

# df_test = all_data[all_data['winPlacePerc'].isnull()].drop(['winPlacePerc'], axis=1).reset_index(drop=True)



# target = 'winPlacePerc'

# features = list(df_train.columns)

# features.remove("Id")

# features.remove("matchId")

# features.remove("groupId")

# features.remove("matchType")



# y_train = np.array(df_train[target])

# features.remove(target)

# x_train = df_train[features]



# x_test = df_test[features]
def feature_engineering(df,is_train=True):

    if is_train: 

        df = df[df['maxPlace'] > 1]



    state('totalDistance')

    s = timer()

    df['totalDistance'] = df['rideDistance'] + df["walkDistance"] + df["swimDistance"]

    e = timer()

    state('totalDistance', False, e - s)

          



    state('rankPoints')

    s = timer()

    df['rankPoints'] = np.where(df['rankPoints'] <= 0 ,0 , df['rankPoints'])

    e = timer()                                  

    state('rankPoints', False, e-s)

    



    target = 'winPlacePerc'

    features = list(df.columns)

    

    # Remove some features from the features list :

    features.remove("Id")

    features.remove("matchId")

    features.remove("groupId")

    features.remove("matchDuration")

    features.remove("matchType")

    

    y = None

    if is_train: 

        y = np.array(df.groupby(['matchId','groupId'])[target].agg('mean'), dtype=np.float64)

        # Remove the target from the features list :

        features.remove(target)

    

    # Make new features indicating the mean of the features ( grouped by match and group ) :

    print("get group mean feature")

    agg = df.groupby(['matchId','groupId'])[features].agg('mean')

    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()

    

    

    # If we are processing the training data let df_out = the grouped  'matchId' and 'groupId'

    if is_train: 

        df_out = agg.reset_index()[['matchId','groupId']]

    else: 

        df_out = df[['matchId','groupId']]

    

    # Merge agg and agg_rank (that we got before) with df_out :

    df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])

    df_out = df_out.merge(agg_rank, suffixes=["_mean", "_mean_rank"], how='left', on=['matchId', 'groupId'])

    

    # Make new features indicating the max value of the features for each group ( grouped by match )

    print("get group max feature")

    agg = df.groupby(['matchId','groupId'])[features].agg('max')

    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()

    

    # Merge the new (agg and agg_rank) with df_out :

    df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])

    df_out = df_out.merge(agg_rank, suffixes=["_max", "_max_rank"], how='left', on=['matchId', 'groupId'])

    

    # Make new features indicating the minimum value of the features for each group ( grouped by match )

    print("get group min feature")

    agg = df.groupby(['matchId','groupId'])[features].agg('min')

    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()

    

    # Merge the new (agg and agg_rank) with df_out :

    df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])

    df_out = df_out.merge(agg_rank, suffixes=["_min", "_min_rank"], how='left', on=['matchId', 'groupId'])

    

    # Make new features indicating the number of players in each group ( grouped by match )

    print("get group size feature")

    agg = df.groupby(['matchId','groupId']).size().reset_index(name='group_size')

     

    # Merge the group_size feature with df_out :

    df_out = df_out.merge(agg, how='left', on=['matchId', 'groupId'])

    

    # Make new features indicating the mean value of each features for each match :

    print("get match mean feature")

    agg = df.groupby(['matchId'])[features].agg('mean').reset_index()

    

    # Merge the new agg with df_out :

    df_out = df_out.merge(agg, suffixes=["", "_match_mean"], how='left', on=['matchId'])

    

    # Make new features indicating the number of groups in each match :

    print("get match size feature")

    agg = df.groupby(['matchId']).size().reset_index(name='match_size')

    

    # Merge the match_size feature with df_out :

    df_out = df_out.merge(agg, how='left', on=['matchId'])

    

    # Drop matchId and groupId

    df_out.drop(["matchId", "groupId"], axis=1, inplace=True)

    df_out = reduce_mem_usage(df_out)

    

    # X is the output dataset (without the target) and y is the target :

    X = np.array(df_out, dtype=np.float64)

    

    

    del df, df_out, agg, agg_rank

    gc.collect()



    return X, y

x_train, y_train = feature_engineering(df_train,True)

x_test,_ = feature_engineering(df_test,False)
# Split the train and the validation set for the fitting

random_seed=1

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.05, random_state=random_seed)
# Random Forest

RF = RandomForestRegressor(n_estimators=10, min_samples_leaf=3, max_features=0.5, n_jobs=-1)
%%time

RF.fit(x_train, y_train)
mae_train_RF = mean_absolute_error(RF.predict(x_train), y_train)

mae_val_RF = mean_absolute_error(RF.predict(x_val), y_val)

print('mae train RF: ', mae_train_RF)

print('mae val RF: ', mae_val_RF)
# Reference link: https://www.kaggle.com/chocozzz/lightgbm-baseline

def run_lgb(train_X, train_y, val_X, val_y, x_test):

    params = {"objective" : "regression", 

              "metric" : "mae", 

              'n_estimators':20000, 

              'early_stopping_rounds':200,

              "num_leaves" : 31, 

              "learning_rate" : 0.05, 

              "bagging_fraction" : 0.7,

              "bagging_seed" : 0, 

              "num_threads" : 4,

              "colsample_bytree" : 0.7

             }

    

    lgtrain = lgb.Dataset(train_X, label=train_y)

    lgval = lgb.Dataset(val_X, label=val_y)

    model = lgb.train(params, lgtrain, valid_sets=[lgtrain, lgval], early_stopping_rounds=200, verbose_eval=1000)

    

    pred_test_y = model.predict(x_test, num_iteration=model.best_iteration)

    return pred_test_y, model
%%time

# Training the model #

pred_test_lgb, model = run_lgb(x_train, y_train, x_val, y_val, x_test)
mae_train_lgb = mean_absolute_error(model.predict(x_train, num_iteration=model.best_iteration), y_train)

mae_val_lgb = mean_absolute_error(model.predict(x_val, num_iteration=model.best_iteration), y_val)



print('mae train lgb: ', mae_train_lgb)

print('mae val lgb: ', mae_val_lgb)
# Reference link: https://www.kaggle.com/qingyuanwu/deep-neural-network

def run_DNN(x_train, y_train, x_val, y_val, x_test):

    NN_model = Sequential()

    NN_model.add(Dense(x_train.shape[1],  input_dim = x_train.shape[1], activation='relu'))

    NN_model.add(Dense(136, activation='relu'))

    NN_model.add(Dense(136, activation='relu'))

    NN_model.add(Dense(136, activation='relu'))

    NN_model.add(Dense(136, activation='relu'))



    # output Layer

    NN_model.add(Dense(1, activation='linear'))



    # Compile the network :

    NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])

    NN_model.summary()

    

    checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 

    checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')

    callbacks_list = [checkpoint]

    

    NN_model.fit(x=x_train, 

                 y=y_train, 

                 batch_size=1000,

                 epochs=30, 

                 verbose=1, 

                 callbacks=callbacks_list,

                 validation_split=0.15, 

                 validation_data=None, 

                 shuffle=True,

                 class_weight=None, 

                 sample_weight=None, 

                 initial_epoch=0,

                 steps_per_epoch=None, 

                 validation_steps=None)



    pred_test_y = NN_model.predict(x_test)

    pred_test_y = pred_test_y.reshape(-1)

    return pred_test_y, NN_model
%%time

# Training the model #

pred_test_DNN, model = run_DNN(x_train, y_train, x_val, y_val, x_test)
mae_train_DNN = mean_absolute_error(model.predict(x_train), y_train)

mae_val_DNN = mean_absolute_error(model.predict(x_val), y_val)

print('mae train dnn: ', mae_train_DNN)

print('mae val dnn: ', mae_val_DNN)
pred_test_RF = RF.predict(x_test)

df_test['winPlacePerc_RF'] = pred_test_RF

submission = df_test[['Id', 'winPlacePerc_RF']]

submission.to_csv('submission_RF.csv', index=False)
df_test['winPlacePerc_lgb'] = pred_test_lgb

submission = df_test[['Id', 'winPlacePerc_lgb']]

submission.to_csv('submission_lgb.csv', index=False)
df_test['winPlacePerc_DNN'] = pred_test_DNN

submission = df_test[['Id', 'winPlacePerc_DNN']]

submission.to_csv('submission_DNN.csv', index=False)
weight_DNN = (1 - mae_val_DNN) / (3 - mae_val_DNN - mae_val_RF - mae_val_lgb)

weight_RF = (1 - mae_val_RF) / (3 - mae_val_DNN - mae_val_RF - mae_val_lgb)

weight_lgb = (1 - mae_val_lgb) / (3 - mae_val_DNN - mae_val_RF - mae_val_lgb)



df_test['winPlacePerc'] = df_test.apply(lambda x: x['winPlacePerc_RF'] * weight_RF + x['winPlacePerc_DNN'] * weight_DNN + x['winPlacePerc_lgb'] * weight_lgb, axis=1)

submission = df_test[['Id', 'winPlacePerc']]

submission.to_csv('submission.csv', index=False)