import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.metrics import mean_absolute_error

import warnings
warnings.simplefilter(action='ignore', category=Warning)

from sklearn.metrics import mean_squared_error, mean_absolute_error

import os
print(os.listdir("../input"))

def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object and col_type.name != 'category' and 'datetime' not in col_type.name:
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
        elif 'datetime' not in col_type.name:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df
df_trn = pd.read_csv('../input/pubg-finish-placement-prediction/train.csv',  nrows=None)
df_trn = reduce_mem_usage(df_trn)

df_tst = pd.read_csv('../input/pubg-finish-placement-prediction/test.csv',  nrows=None)
df_tst = reduce_mem_usage(df_tst)
lgbm_trn = pd.read_csv('../input/pubg-survivor-kit/oof_lgbm1_reg.csv')
lgbm_trn.columns = [c if 'winPlacePerc' not in c else c+'Pred' for c in lgbm_trn.columns]

lgbm_tst = pd.read_csv('../input/pubg-survivor-kit/sub_lgbm1_reg.csv')
lgbm_tst.columns = [c if 'winPlacePerc' not in c else c+'Pred' for c in lgbm_tst.columns]
df_trn2 = pd.concat([df_trn, lgbm_trn['winPlacePercPred']], axis=1)
df_tst2 = pd.concat([df_tst, lgbm_tst['winPlacePercPred']], axis=1)
df_trn3 = df_trn2.groupby(['matchId','groupId'])['winPlacePercPred'].agg('mean').groupby('matchId').rank(pct=True).reset_index()
df_trn3.columns = [c if 'winPlacePerc' not in c else c+'_Rank' for c in df_trn3.columns]
df_trn3 = df_trn3.merge(df_trn2, how='left', on=['matchId','groupId'])
print('MAE by default: {:.4f}'.format(
    mean_absolute_error(df_trn3['winPlacePerc'], df_trn3['winPlacePercPred'])
                                 )
     )
print('MAE after group ranking: {:.4f}'.format(
    mean_absolute_error(df_trn3['winPlacePerc'], df_trn3['winPlacePercPred_Rank'])
                                 )
     )
df_tst3 = df_tst2.groupby(['matchId','groupId'])['winPlacePercPred'].agg('mean').groupby('matchId').rank(pct=True).reset_index()
df_tst3.columns = [c if 'winPlacePerc' not in c else c+'_Rank' for c in df_tst3.columns]
df_tst3 = df_tst2.merge(df_tst3, how='left', on=['matchId','groupId'])
del lgbm_tst['winPlacePercPred']
lgbm_tst['winPlacePerc'] = df_tst3['winPlacePercPred_Rank']
lgbm_tst.to_csv('sub_lgbm_group_ranked_within_game.csv', index=False)
!head sub_lgbm_group_ranked_within_game.csv
