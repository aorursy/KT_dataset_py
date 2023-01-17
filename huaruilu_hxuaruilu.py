# if need please install them



#pip install pandas

#pip install numpy

#pip install seaborn

#pip install lightgbm

#pip install matplotlib
import pandas as pd

import numpy as np

import seaborn as sns

import lightgbm as lgb

import matplotlib 

from matplotlib import pyplot as plt
df_train = pd.read_csv('../input/pubg-finish-placement-prediction/train_V2.csv')

df_test = pd.read_csv('../input/pubg-finish-placement-prediction/test_V2.csv')

df_train
df_test
df_train.info()
df_train.describe()
train_columns = ['assists', 'boosts', 'damageDealt', 'DBNOs', 'headshotKills', 

                  'heals', 'killPlace', 'killPoints', 'kills', 'killStreaks', 

                  'longestKill', 'maxPlace', 'numGroups', 'revives','rideDistance', 

                  'roadKills', 'swimDistance', 'teamKills', 'vehicleDestroys', 'walkDistance', 

                  'weaponsAcquired', 'winPoints']
def reduce_mem_usage(df):

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

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

    return df
df_train = reduce_mem_usage(df_train)

df_test = reduce_mem_usage(df_test)
correlation = df_train.corr()

plt.figure(figsize=(25,25))

sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='viridis')

plt.title('Correlation between different fearures')
def show_countplot(column):

    plt.figure(figsize=(20,4))

    sns.countplot(data=df_train, x=column).set_title(column)

    plt.show()
show_countplot('matchType')
train_solo = df_train.loc[df_train["matchType"] == "solo" ]

train_solo = train_solo.append( df_train.loc[df_train["matchType"] == "solo-fpp" ])

train_solo = train_solo.append( df_train.loc[df_train["matchType"] == "normal-solo-fpp" ])

train_solo = train_solo.append( df_train.loc[df_train["matchType"] == "normal-solo" ])

train_solo
def split_train_val(data, fraction):

    matchIds = data['matchId'].unique().reshape([-1])

    train_size = int(len(matchIds)*fraction)

    

    random_idx = np.random.RandomState(seed=2).permutation(len(matchIds))

    train_matchIds = matchIds[random_idx[:train_size]]

    val_matchIds = matchIds[random_idx[train_size:]]

    

    data_train = data.loc[data['matchId'].isin(train_matchIds)]

    data_val = data.loc[data['matchId'].isin(val_matchIds)]

    return data_train, data_val
x_train, x_train_test = split_train_val(train_solo, 0.91)
params = {

        "objective" : "regression", 

        "metric" : "mae", 

        "num_leaves" : 149, 

        "learning_rate" : 0.03, 

        "bagging_fraction" : 0.9,

        "bagging_seed" : 0, 

        "num_threads" : 4,

        "colsample_bytree" : 0.5,

        'min_data_in_leaf':1900, 

        'min_split_gain':0.00011,

        'lambda_l2':9

}

train_set = lgb.Dataset(x_train[train_columns], label=x_train['winPlacePerc'])

valid_set = lgb.Dataset(x_train_test[train_columns], label=x_train_test['winPlacePerc'])
model_solo = lgb.train(  params, 

                    train_set = train_set,

                    num_boost_round=9400,

                    early_stopping_rounds=200,

                    verbose_eval=100, 

                    valid_sets=[train_set,valid_set]

                  )
test_solo = df_test.loc[df_test["matchType"] == "solo" ]

test_solo = test_solo.append( df_test.loc[df_test["matchType"] == "solo-fpp" ])

test_solo = test_solo.append( df_test.loc[df_test["matchType"] == "normal-solo-fpp" ])

test_solo = test_solo.append( df_test.loc[df_test["matchType"] == "normal-solo" ])
test_solo
pre_solo = model_solo.predict(test_solo[train_columns])

test_solo['winPlacePerc'] = pre_solo

test_solo
plt.figure(figsize=(20,8))

lgb.plot_importance(model_solo, max_num_features=22)

plt.title("Featurertances with Solo pattern")

plt.show()
train_duo = df_train.loc[df_train["matchType"] == "duo" ]

train_duo = train_duo.append( df_train.loc[df_train["matchType"] == "duo-fpp" ])

train_duo = train_duo.append( df_train.loc[df_train["matchType"] == "normal-duo-fpp" ])

train_duo = train_duo.append( df_train.loc[df_train["matchType"] == "normal-duo" ])

train_duo
x_train, x_train_test = split_train_val(train_duo, 0.91)

train_set = lgb.Dataset(x_train[train_columns], label=x_train['winPlacePerc'])

valid_set  = lgb.Dataset(x_train_test[train_columns], label=x_train_test['winPlacePerc'])
model_duo = lgb.train( params, 

                    train_set = train_set,

                    num_boost_round=9400,

                    early_stopping_rounds=200,

                    verbose_eval=100, 

                    valid_sets=[train_set,valid_set]

                  )
test_duo = df_test.loc[df_test["matchType"] == "duo" ]

test_duo = test_duo.append( df_test.loc[df_test["matchType"] == "duo-fpp" ])

test_duo = test_duo.append( df_test.loc[df_test["matchType"] == "normal-duo-fpp" ])

test_duo = test_duo.append( df_test.loc[df_test["matchType"] == "normal-duo" ])
pre_duo = model_duo.predict(test_duo[train_columns])

test_duo['winPlacePerc'] = pre_duo

test_duo
plt.figure(figsize=(20,8))

lgb.plot_importance(model_duo, max_num_features=30)

plt.title("Featurertances with Duo pattern")

plt.show()
train_squad = df_train.loc[df_train["matchType"] == "squad" ]

train_squad = train_squad.append(df_train.loc[df_train["matchType"] == "squad-fpp" ])

train_squad = train_squad.append( df_train.loc[df_train["matchType"] == "normal-squad-fpp" ])

train_squad = train_squad.append( df_train.loc[df_train["matchType"] == "normal-squad" ])

train_squad
x_train, x_train_test = split_train_val(train_squad, 0.91)

train_set = lgb.Dataset(x_train[train_columns], label=x_train['winPlacePerc'])

valid_set  = lgb.Dataset(x_train_test[train_columns], label=x_train_test['winPlacePerc'])
model_squad = lgb.train( params, 

                    train_set = train_set,

                    num_boost_round=9400,

                    early_stopping_rounds=200,

                    verbose_eval=100, 

                    valid_sets=[train_set,valid_set]

                  )
test_squad = df_test.loc[df_test["matchType"] == "squad" ]

test_squad = test_squad.append( df_test.loc[df_test["matchType"] == "squad-fpp" ])

test_squad = test_squad.append( df_test.loc[df_test["matchType"] == "normal-squad-fpp" ])

test_squad = test_squad.append( df_test.loc[df_test["matchType"] == "normal-squad" ])
pre_squad = model_squad.predict(test_squad[train_columns])

test_squad['winPlacePerc'] = pre_squad

test_squad
plt.figure(figsize=(20,8))

lgb.plot_importance(model_squad, max_num_features=22)

plt.title("Featurertances with Squad pattern")

plt.show()
train_others = df_train.loc[df_train["matchType"] == "flaretpp" ]

train_others = train_others.append(df_train.loc[df_train["matchType"] == "crashfpp" ])

train_others = train_others.append( df_train.loc[df_train["matchType"] == "flarefpp" ])

train_others = train_others.append( df_train.loc[df_train["matchType"] == "crashtpp" ])

train_others
x_train, x_train_test = split_train_val(train_others, 0.91)

train_set = lgb.Dataset(x_train[train_columns], label=x_train['winPlacePerc'])

valid_set  = lgb.Dataset(x_train_test[train_columns], label=x_train_test['winPlacePerc'])
model_others = lgb.train( params, 

                    train_set = train_set,

                    num_boost_round=9400,

                    early_stopping_rounds=200,

                    verbose_eval=100, 

                    valid_sets=[train_set,valid_set]

                  )
test_others = df_test.loc[df_test["matchType"] == "flaretpp" ]

test_others = test_others.append(df_test.loc[df_test["matchType"] == "crashfpp" ])

test_others = test_others.append( df_test.loc[df_test["matchType"] == "flarefpp" ])

test_others = test_others.append( df_test.loc[df_test["matchType"] == "crashtpp" ])
pre_others = model_others.predict(test_others[train_columns])

test_others['winPlacePerc'] = pre_others

test_others
plt.figure(figsize=(20,8))

lgb.plot_importance(model_others, max_num_features=22)

plt.title("Featurertances with Others pattern")

plt.show()
test_result = test_solo.append(test_duo).append(test_squad).append(test_others)
test_result = test_result.sort_index(ascending=True)
result = test_result[['Id','winPlacePerc']]

result
import os

os.getcwd()
result.to_csv('submission.csv',index=False)