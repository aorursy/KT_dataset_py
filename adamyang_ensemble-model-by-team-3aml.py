import numpy as np

import pandas as pd

from sklearn import preprocessing



import os

import time

import warnings

warnings.filterwarnings("ignore")



import matplotlib.pyplot as plt

import seaborn as sns

# model

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error

import lightgbm as lgb



import gc, sys

gc.enable()



# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



from keras import regularizers

from keras.models import Sequential

from keras.layers import Dense, LeakyReLU

from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV



from sklearn.pipeline import Pipeline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
def feature_engineering(is_train=True,debug=True):

    if is_train: 

        print("processing train.csv")

        if debug == True:

            df = pd.read_csv('../input/train_V2.csv', nrows=10000)

        else:

            df = pd.read_csv('../input/train_V2.csv')           



        df = df[df['maxPlace'] > 1]

    else:

        print("processing test.csv")

        df = pd.read_csv('../input/test_V2.csv')

    

    print("remove some columns")

    target = 'winPlacePerc'



    print("Adding Features")

 

    df['headshotrate'] = df['kills']/df['headshotKills']

    df['killStreakrate'] = df['killStreaks']/df['kills']

    df['healthitems'] = df['heals'] + df['boosts']

    df['totalDistance'] = df['rideDistance'] + df["walkDistance"] + df["swimDistance"]

    df['killPlace_over_maxPlace'] = df['killPlace'] / df['maxPlace']

    df['headshotKills_over_kills'] = df['headshotKills'] / df['kills']

    df['distance_over_weapons'] = df['totalDistance'] / df['weaponsAcquired']

    df['walkDistance_over_heals'] = df['walkDistance'] / df['heals']

    df['walkDistance_over_kills'] = df['walkDistance'] / df['kills']

    df['killsPerWalkDistance'] = df['kills'] / df['walkDistance']

    df["skill"] = df["headshotKills"] + df["roadKills"]



    df[df == np.Inf] = np.NaN

    df[df == np.NINF] = np.NaN

    

    print("Removing Na's From DF")

    df.fillna(0, inplace=True)



    

    features = list(df.columns)

    features.remove("Id")

    features.remove("matchId")

    features.remove("groupId")

    features.remove("matchType")

    

    y = None

    

    

    if is_train: 

        print("get target")

        y = df.groupby(['matchId','groupId'])[target].agg('mean')

        features.remove(target)



    print("get group mean feature")

    agg = df.groupby(['matchId','groupId'])[features].agg('mean')

    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()

    

    if is_train: df_out = agg.reset_index()[['matchId','groupId']]

    else: df_out = df[['matchId','groupId']]



    df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])

    df_out = df_out.merge(agg_rank, suffixes=["_mean", "_mean_rank"], how='left', on=['matchId', 'groupId'])

    

    # print("get group sum feature")

    # agg = df.groupby(['matchId','groupId'])[features].agg('sum')

    # agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()

    # df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])

    # df_out = df_out.merge(agg_rank, suffixes=["_sum", "_sum_rank"], how='left', on=['matchId', 'groupId'])

    

    print("get group max feature")

    agg = df.groupby(['matchId','groupId'])[features].agg('max')

    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()

    df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])

    df_out = df_out.merge(agg_rank, suffixes=["_max", "_max_rank"], how='left', on=['matchId', 'groupId'])

    

    print("get group min feature")

    agg = df.groupby(['matchId','groupId'])[features].agg('min')

    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()

    df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])

    df_out = df_out.merge(agg_rank, suffixes=["_min", "_min_rank"], how='left', on=['matchId', 'groupId'])

    

    print("get group size feature")

    agg = df.groupby(['matchId','groupId']).size().reset_index(name='group_size')

    df_out = df_out.merge(agg, how='left', on=['matchId', 'groupId'])

    

    print("get match mean feature")

    agg = df.groupby(['matchId'])[features].agg('mean').reset_index()

    df_out = df_out.merge(agg, suffixes=["", "_match_mean"], how='left', on=['matchId'])



    print("get match size feature")

    agg = df.groupby(['matchId']).size().reset_index(name='match_size')

    df_out = df_out.merge(agg, how='left', on=['matchId'])

    

    df_out.drop(["matchId", "groupId"], axis=1, inplace=True)



    X = df_out

    

    del df, df_out, agg, agg_rank

    gc.collect()



    return X, y
def reduce_mem_usage(df):

    # iterate through all the columns of a dataframe and modify the data type

    #   to reduce memory usage.



    start_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))



    for col in df.columns:

        col_type = df[col].dtype



        if col_type != object:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(

                        np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)

            else:

                if c_min > np.finfo(

                        np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(

        100 * (start_mem - end_mem) / start_mem))

    return df
x_train, y_train = feature_engineering(True,False)

x_test, _ = feature_engineering(False,True)
x_train = reduce_mem_usage(x_train)

x_test = reduce_mem_usage(x_test)
def post_rst(pred_test):

    df_sub = pd.read_csv("../input/sample_submission_V2.csv")

    df_test = pd.read_csv("../input/test_V2.csv")

    df_sub['winPlacePerc'] = pred_test

    # Restore some columns

    df_sub = df_sub.merge(df_test[["Id", "matchId", "groupId", "maxPlace", "numGroups"]], on="Id", how="left")



    # Sort, rank, and assign adjusted ratio

    df_sub_group = df_sub.groupby(["matchId", "groupId"]).first().reset_index()

    df_sub_group["rank"] = df_sub_group.groupby(["matchId"])["winPlacePerc"].rank()

    df_sub_group = df_sub_group.merge(

        df_sub_group.groupby("matchId")["rank"].max().to_frame("max_rank").reset_index(), 

        on="matchId", how="left")

    df_sub_group["adjusted_perc"] = (df_sub_group["rank"] - 1) / (df_sub_group["numGroups"] - 1)



    df_sub = df_sub.merge(df_sub_group[["adjusted_perc", "matchId", "groupId"]], on=["matchId", "groupId"], how="left")

    df_sub["winPlacePerc"] = df_sub["adjusted_perc"]



    # Deal with edge cases

    df_sub.loc[df_sub.maxPlace == 0, "winPlacePerc"] = 0

    df_sub.loc[df_sub.maxPlace == 1, "winPlacePerc"] = 1



    # Align with maxPlace

    # Credit: https://www.kaggle.com/anycode/simple-nn-baseline-4

    subset = df_sub.loc[df_sub.maxPlace > 1]

    gap = 1.0 / (subset.maxPlace.values - 1)

    new_perc = np.around(subset.winPlacePerc.values / gap) * gap

    df_sub.loc[df_sub.maxPlace > 1, "winPlacePerc"] = new_perc



    # Edge case

    df_sub.loc[(df_sub.maxPlace > 1) & (df_sub.numGroups == 1), "winPlacePerc"] = 0

    assert df_sub["winPlacePerc"].isnull().sum() == 0



    df_sub[["Id", "winPlacePerc"]].to_csv("submission.csv", index=False)
def nn_model(input_shape=x_train.shape[1], hidden_size=64):

    model = Sequential()

    model.add(Dense(hidden_size, input_dim=input_shape, kernel_initializer='normal'))

    model.add(LeakyReLU(0.1))

    model.add(Dense(hidden_size, kernel_initializer='normal'))

    model.add(LeakyReLU(0.1))

    model.add(Dense(hidden_size, kernel_initializer='normal'))

    model.add(LeakyReLU(0.1))

    model.add(Dense(hidden_size, kernel_initializer='normal'))

    model.add(LeakyReLU(0.1))



    model.add(Dense(1, kernel_initializer='normal'))

    

    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mae'])

    return model
def train_val_split(x_train, y_train):

    train_index = round(int(x_train.shape[0]*0.8))

    dev_X = x_train[:train_index] 

    val_X = x_train[train_index:]

    dev_y = y_train[:train_index] 

    val_y = y_train[train_index:]

    del x_train, y_train

    gc.collect();

    

    return dev_X, val_X, dev_y, val_y
dev_X, val_X, dev_y, val_y = train_val_split(x_train, y_train)



# custom function to run light gbm model

def run_lgb(train_X, train_y, val_X, val_y, x_test):

    params = {"objective" : "regression", "metric" : "mae", 'n_estimators':5000, 'early_stopping_rounds':200,

              "num_leaves" : 150, "learning_rate" : 0.05, "bagging_fraction" : 0.5,

               "bagging_seed" : 0, "num_threads" : 4,"colsample_bytree" : 0.5

             }

    

    lgtrain = lgb.Dataset(train_X, label=train_y)

    lgval = lgb.Dataset(val_X, label=val_y)

    model = lgb.train(params, lgtrain, valid_sets=[lgtrain, lgval], early_stopping_rounds=200, verbose_eval=1000)

    

    pred_test_y = model.predict(x_test, num_iteration=model.best_iteration)

    return pred_test_y



# Training the model #

pred_test_lgb = run_lgb(dev_X, dev_y, val_X, val_y, x_test)

pred_test_lgb = pred_test_lgb.reshape(-1,1)



del dev_X, val_X, dev_y, val_y

gc.collect()
# dev_X, val_X, dev_y, val_y = train_val_split(x_train, y_train)



np.random.seed(42)



std_scaler = StandardScaler().fit(x_train)

x_train = std_scaler.transform(x_train)

x_test = std_scaler.transform(x_test)



mlp_model = nn_model(x_train.shape[1], hidden_size=64)

mlp_model.fit(x_train, y_train, batch_size=128, epochs=8, verbose=1, validation_split=0.1, shuffle=True)

pred_test_mlp = mlp_model.predict(x_test)





del mlp_model

gc.collect()
del x_train, y_train, x_test

gc.collect()
pred_test = (pred_test_mlp + pred_test_lgb) / 2.0
post_rst(pred_test)