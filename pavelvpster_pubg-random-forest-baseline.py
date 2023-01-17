import numpy as np
import pandas as pd

import warnings
warnings.simplefilter('ignore')

from sklearn.ensemble import RandomForestRegressor

import os
print(os.listdir("../input"))
# Thanks to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage

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
#        else:
#            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df
df_train = pd.read_csv('../input/train_V2.csv', index_col='Id')
df_train.shape
df_train = reduce_mem_usage(df_train)
df_train.head().T
df_test = pd.read_csv('../input/test_V2.csv', index_col = 'Id')
df_test.shape
df_test = reduce_mem_usage(df_test)
df_test_id = pd.DataFrame(index=df_test.index)
part_train = 0.05
part_valid = 0.05
match_ids = df_train['matchId'].unique()
match_ids_train = np.random.choice(match_ids, int(part_train * len(match_ids)))
len(match_ids_train)
df_train_train = df_train[df_train['matchId'].isin(match_ids_train)]
df_train_train.shape[0]
match_ids_valid = np.random.choice(np.setdiff1d(match_ids, match_ids_train), int(part_valid * len(match_ids)))
len(match_ids_valid)
del df_train
del match_ids
del match_ids_train
# Thanks to many kernels in the competition

def feature_engineering(df, is_train=True):
    
    # fix rank points
    df['rankPoints'] = np.where(df['rankPoints'] <= 0, 0, df['rankPoints'])
    
    features = list(df.columns)
    features.remove("matchId")
    features.remove("groupId")
    features.remove("matchDuration")
    features.remove("matchType")
    if 'winPlacePerc' in features:
        features.remove('winPlacePerc')
    
    y = None
    
    # average y for training dataset
    if is_train:
        y = df.groupby(['matchId','groupId'])['winPlacePerc'].agg('mean')
    elif 'winPlacePerc' in df.columns:
        y = df['winPlacePerc']
    
    # mean by match and group
    agg = df.groupby(['matchId','groupId'])[features].agg('mean')
    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()
    
    if is_train:
        df_out = agg.reset_index()[['matchId','groupId']]
    else:
        df_out = df[['matchId','groupId']]
    
    df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    df_out = df_out.merge(agg_rank, suffixes=["_mean", "_mean_rank"], how='left', on=['matchId', 'groupId'])
    
    # max by match and group
    agg = df.groupby(['matchId','groupId'])[features].agg('max')
    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()
    
    df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    df_out = df_out.merge(agg_rank, suffixes=["_max", "_max_rank"], how='left', on=['matchId', 'groupId'])
    
    # max by match and group
    agg = df.groupby(['matchId','groupId'])[features].agg('min')
    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()
    
    df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    df_out = df_out.merge(agg_rank, suffixes=["_min", "_min_rank"], how='left', on=['matchId', 'groupId'])
    
    # number of players in group
    agg = df.groupby(['matchId','groupId']).size().reset_index(name='group_size')
    
    df_out = df_out.merge(agg, how='left', on=['matchId', 'groupId'])
    
    # mean by match
    agg = df.groupby(['matchId'])[features].agg('mean').reset_index()
    
    df_out = df_out.merge(agg, suffixes=["", "_match_mean"], how='left', on=['matchId'])
    
    # number of groups in match
    agg = df.groupby(['matchId']).size().reset_index(name='match_size')
    
    df_out = df_out.merge(agg, how='left', on=['matchId'])
    
    # drop match id and group id
    df_out.drop(["matchId", "groupId"], axis=1, inplace=True)
    
    del agg, agg_rank
    
    return df_out, y
%%time
fe_x_train, fe_y_train = feature_engineering(df_train_train, is_train=True)
fe_x_train.shape
del df_train_train
%%time
fe_x_test, t = feature_engineering(df_test, is_train=False)
fe_x_test.shape
del df_test
%%time
rf = RandomForestRegressor(n_estimators=30, criterion='mae', n_jobs=-1)
rf.fit(fe_x_train, fe_y_train)
rf_y_pred = rf.predict(fe_x_test)
# Thanks to https://www.kaggle.com/anycode/simple-nn-baseline-4

def fix_pred(x, pred):
    
    updated_pred = []
    for i in range(len(x)):
        winPlacePerc = pred[i]
        
        maxPlace = int(x.iloc[i]['maxPlace'])
        if maxPlace == 0:
            winPlacePerc = 0.0
        elif maxPlace == 1:
            winPlacePerc = 1.0
        else:
            gap = 1.0 / (maxPlace - 1)
            winPlacePerc = round(winPlacePerc / gap) * gap
        
        if winPlacePerc < 0: winPlacePerc = 0.0
        if winPlacePerc > 1: winPlacePerc = 1.0
        
        updated_pred.append(winPlacePerc)
    
    return updated_pred
%%time
fixed_rf_y_pred = fix_pred(fe_x_test, rf_y_pred)
def make_submission(x, y, filename):
    submission = pd.DataFrame(index=x.index)
    submission['winPlacePerc'] = np.clip(y, a_min=0, a_max=1)
    submission.to_csv(filename, index_label='Id')
make_submission(df_test_id, fixed_rf_y_pred, 'random_forest_baseline.csv')