import pandas as pd
import numpy as np
import gc
gc.enable()
# Thanks and credited to https://www.kaggle.com/gemartin who created this wonderful mem reducer
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
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
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


def import_data(file):
    """create a dataframe and optimize its memory usage"""
    df = pd.read_csv(file, parse_dates=True, keep_date_col=True)
    df = reduce_mem_usage(df)
    return df
print('-' * 80)
print('train')
train = import_data('../input/train_V2.csv')
# train = train[train['maxPlace'] > 1]

train['totalDistance'] = train['rideDistance'] + train["walkDistance"] + train["swimDistance"]
train.head()
train.info()
train.isnull().sum().sum()
# y = np.array(train.groupby(['matchId','groupId'])['winPlacePerc'].agg('mean'), dtype=np.float64)
y = train['winPlacePerc']
train.drop('winPlacePerc', axis=1, inplace=True)
"""
it is a team game, scores within the same group is same, so let's get the feature of each group
"""
train_size = train.groupby(['matchId','groupId']).size().reset_index(name='group_size')
train_mean = train.groupby(['matchId','groupId']).mean().reset_index()
train_max = train.groupby(['matchId','groupId']).max().reset_index()
train_min = train.groupby(['matchId','groupId']).min().reset_index()
"""
although you are a good game player, 
but if other players of other groups in the same match is better than you, you will still get little score
so let's add the feature of each match
"""
train_match_mean = train.groupby(['matchId']).mean().reset_index()

train = pd.merge(train, train_mean, suffixes=["", "_mean"], how='left', on=['matchId', 'groupId'])
del train_mean

train = pd.merge(train, train_max, suffixes=["", "_max"], how='left', on=['matchId', 'groupId'])
del train_max

train = pd.merge(train, train_min, suffixes=["", "_min"], how='left', on=['matchId', 'groupId'])
del train_min

train = pd.merge(train, train_match_mean, suffixes=["", "_match_mean"], how='left', on=['matchId'])
del train_match_mean

train = pd.merge(train, train_size, how='left', on=['matchId', 'groupId'])
del train_size

train_columns = list(train.columns)
""" remove some columns """
train_columns.remove("Id")
train_columns.remove("matchId")
train_columns.remove("groupId")
train_columns.remove("Id_mean")
train_columns.remove("Id_max")
train_columns.remove("Id_min")
train_columns.remove("Id_match_mean")
train_columns.remove("matchType")
"""
in this game, team skill level is more important than personal skill level 
maybe you are a good player, but if your teammates is bad, you will still lose
so let's remove the features of each player, just select the features of group and match
"""
train_columns_new = []
for name in train_columns:
    if '_' in name:
        train_columns_new.append(name)
train_columns = train_columns_new    
print(train_columns)
# train_columns = ['assists', 'boosts', 'damageDealt', 'DBNOs', 'headshotKills', 
#                 'heals', 'killPlace', 'killPoints', 'kills', 'killStreaks', 
#                 'longestKill', 'maxPlace', 'numGroups', 'revives','rideDistance', 
#                 'roadKills', 'swimDistance', 'teamKills', 'vehicleDestroys', 'walkDistance', 
#                 'weaponsAcquired', 'winPoints']

X_train = train[train_columns]
X_train['winPlacePerc'] = y
x_train = X_train.sample(frac=0.8)
X_val = X_train.loc[~X_train.index.isin(x_train.index)]
y_train = x_train['winPlacePerc']
x_train.drop('winPlacePerc', axis=1, inplace=True)
y_val = X_val['winPlacePerc']
X_val.drop('winPlacePerc', axis=1, inplace=True)

del X_train, train
gc.collect()
print(x_train.info(memory_usage='deep', verbose=False))
print(X_val.info(memory_usage='deep', verbose=False))
# from sklearn import preprocessing
# scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1), copy=False).fit(x_train)
# # scaler = preprocessing.QuantileTransformer().fit(x_train)

# x_train = scaler.transform(x_train)
# X_val = scaler.transform(X_val)
# X_test = scaler.transform(X_test)
# y_train = y_train*2 - 1

# print("x_train", x_train.shape, x_train.min(), x_train.max())
# print("X_val", X_val.shape, X_val.min(), X_val.max())
# print("X_test", X_test.shape, X_test.min(), X_test.max())
# print("y_train", y_train.shape, y_train.max(), y_train.min())

# X_val = np.clip(X_val, a_min=-1, a_max=1)
# print("X_val", X_val.shape, X_val.min(), X_val.max())
# X_test = np.clip(X_test, a_min=-1, a_max=1)
# print("x_test", X_test.shape, X_test.min(), X_test.max())
# gc.collect()
import catboost
import time
from catboost import CatBoostRegressor

start_time = time.time()
model = CatBoostRegressor(iterations=500, learning_rate=0.05, loss_function='MAE',eval_metric='MAE', depth = 15,
#                           task_type = "GPU", 
#                           one_hot_max_size = 64,
                          use_best_model=True, od_type="Iter", od_wait=20, thread_count=128, random_seed = 123)
model.fit(x_train, y_train, eval_set=(X_val, y_val))
end_time = time.time()
print('The training time = {}'.format(end_time - start_time))
print('-' * 80)
print('test')
test = import_data('../input/test_V2.csv')
test_new = test[['Id', 'matchId', 'groupId']]

test['totalDistance'] = test['rideDistance'] + test["walkDistance"] + test["swimDistance"]

test.info(memory_usage='deep', verbose=False)

test_size = test.groupby(['matchId','groupId']).size().reset_index(name='group_size')
test_mean = test.groupby(['matchId','groupId']).mean().reset_index()
test_max = test.groupby(['matchId','groupId']).max().reset_index()
test_min = test.groupby(['matchId','groupId']).min().reset_index()

test_match_mean = test.groupby(['matchId']).mean().reset_index()
test = pd.merge(test, test_mean, suffixes=["", "_mean"], how='left', on=['matchId', 'groupId'])
del test_mean
test = pd.merge(test, test_max, suffixes=["", "_max"], how='left', on=['matchId', 'groupId'])
del test_max
test = pd.merge(test, test_min, suffixes=["", "_min"], how='left', on=['matchId', 'groupId'])
del test_min
test = pd.merge(test, test_match_mean, suffixes=["", "_match_mean"], how='left', on=['matchId'])
del test_match_mean
test = pd.merge(test, test_size, how='left', on=['matchId', 'groupId'])
del test_size
X_test = test[train_columns]
del test
gc.collect()
pred = model.predict(X_test)
test_new['winPlacePercPred'] = pred

print('Correcting predictions')
        
test_new['prediction_mod'] = -1.0
matchId = test_new['matchId'].unique()

for match in matchId:
    df_match = test_new[test_new['matchId']==match]

    df_max = df_match.groupby(['groupId']).max()
    pred_sort = sorted(df_max['winPlacePercPred'])

    for i in df_max.index:
        groupPlace = pred_sort.index(df_max.loc[i]['winPlacePercPred'])
        if len(pred_sort) > 1:
            df_max.at[i,'prediction_mod'] = groupPlace/(len(pred_sort)-1)
        else:
            df_max.at[i,'prediction_mod'] = 1.0

    for i in df_match.index:
        test_new.at[i, 'prediction_mod'] = df_max['prediction_mod'].loc[df_match['groupId'].loc[i]]

y_submit_cor = test_new['prediction_mod']
print('Submission scores corrected')

test_new.head()
df_test = import_data('../input/sample_submission_V2.csv')
df_test['winPlacePerc'] = y_submit_cor
df_test['Id'] = df_test['Id'].astype(str)
submission = df_test[['Id', 'winPlacePerc']]

submission.to_csv('submission.csv', index=False) 
print('Submission file made\n')
