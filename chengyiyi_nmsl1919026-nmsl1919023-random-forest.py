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

                #if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                #    df[col] = df[col].astype(np.float16)

                #el

                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

        #else:

            #df[col] = df[col].astype('category')



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB --> {:.2f} MB (Decreased by {:.1f}%)'.format(

        start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df



# Import dataset

df_train = pd.read_csv('../input/pubg-finish-placement-prediction/train_V2.csv')

df_test = pd.read_csv('../input/pubg-finish-placement-prediction/test_V2.csv')



# Reduce memory use

df_train=reduce_mem_usage(df_train)

df_test=reduce_mem_usage(df_test)



# Show some data

df_train.head()

df_train.describe()
# First five rows (From Head)

print('First 5 rows: ')

display(df_train.head())



# Last five rows (To Tail)

print('Last 5 rows: ')

display(df_train.tail())
# Types, Data points, memory usage, etc.

df_train.info()
df_train[df_train['winPlacePerc'].isnull()]
df_train.drop(2744604, inplace=True)
df_train[df_train['winPlacePerc'].isnull()]
def feature_barplot(feature, df_train = df_train, figsize=(15,6), rot = 90, saveimg = False): 

    feat_train = df_train[feature].value_counts()

    fig_feature, axis1, = plt.subplots(1,1,sharex=True, sharey = True, figsize = figsize)

    sns.barplot(feat_train.index.values, feat_train.values, ax = axis1)

    axis1.set_xticklabels(axis1.xaxis.get_majorticklabels(), rotation = rot)

    axis1.set_title(feature + ' of training dataset')

    axis1.set_ylabel('Counts')

    plt.tight_layout()

    if saveimg == True:

        figname = feature + ".png"

        fig_feature.savefig(figname, dpi = 75)



feature_barplot('DBNOs') #DBNO



data = df_train.copy()

data.loc[data['kills'] > data['kills'].quantile(0.99)] = '8+'

plt.figure(figsize=(15,10))

sns.countplot(data['kills'].astype('str').sort_values())

plt.title("Kill Count",fontsize=15)

plt.show()
sns.jointplot(x="winPlacePerc", y="kills", data=df_train, height=10, ratio=3, color="r")

plt.show()
kills = df_train.copy()



kills['killsCategories'] = pd.cut(kills['kills'], [-1, 0, 2, 5, 10, 60], labels=['0_kills','1-2_kills', '3-5_kills', '6-10_kills', '10+_kills'])



plt.figure(figsize=(15,8))

sns.boxplot(x="killsCategories", y="winPlacePerc", data=kills)

plt.show()



# Get alldata for feature engineering

all_data = df_train.append(df_test, sort=False).reset_index(drop=True)



# Map the matchType

all_data['matchType'] = all_data['matchType'].map({

    'crashfpp':1,

    'crashtpp':2,

    'duo':3,

    'duo-fpp':4,

    'flarefpp':5,

    'flaretpp':6,

    'normal-duo':7,

    'normal-duo-fpp':8,

    'normal-solo':9,

    'normal-solo-fpp':10,

    'normal-squad':11,

    'normal-squad-fpp':12,

    'solo':13,

    'solo-fpp':14,

    'squad':15,

    'squad-fpp':16

    })



# Normalize features

all_data['playersJoined'] = all_data.groupby('matchId')['matchId'].transform('count')

all_data['killsNorm'] = all_data['kills']*((100-all_data['playersJoined'])/100 + 1)

all_data['damageDealtNorm'] = all_data['damageDealt']*((100-all_data['playersJoined'])/100 + 1)

all_data['maxPlaceNorm'] = all_data['maxPlace']*((100-all_data['playersJoined'])/100 + 1)

all_data['matchDurationNorm'] = all_data['matchDuration']*((100-all_data['playersJoined'])/100 + 1)



all_data['healsandboosts'] = all_data['heals'] + all_data['boosts']

all_data['totalDistance'] = all_data['rideDistance'] + all_data['walkDistance'] + all_data['swimDistance']

all_data['killsWithoutMoving'] = ((all_data['kills'] > 0) & (all_data['totalDistance'] == 0))



all_data=reduce_mem_usage(all_data)



# Split the train and the test

df_train = all_data[all_data['winPlacePerc'].notnull()].reset_index(drop=True)

df_test = all_data[all_data['winPlacePerc'].isnull()].drop(['winPlacePerc'], axis=1).reset_index(drop=True)



target = 'winPlacePerc'

features = list(df_train.columns)

features.remove("Id")

features.remove("matchId")

features.remove("groupId")

features.remove("matchType")



y_train = np.array(df_train[target])

features.remove(target)

x_train = df_train[features]



x_test = df_test[features]
# Split the train and the validation set for the fitting

random_seed=1

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.05, random_state=random_seed)
RF = RandomForestRegressor(n_estimators=10, min_samples_leaf=3, max_features=0.5, n_jobs=-1)
RF.fit(x_train, y_train)
mae_train_RF = mean_absolute_error(RF.predict(x_train), y_train)

mae_val_RF = mean_absolute_error(RF.predict(x_val), y_val)

print('mae train RF: ', mae_train_RF)

print('mae val RF: ', mae_val_RF)
%%time

pred_test_RF = RF.predict(x_test)

df_test['winPlacePerc_RF'] = pred_test_RF

submission = df_test[['Id', 'winPlacePerc_RF']]

submission.to_csv('submission_RF.csv', index=False)