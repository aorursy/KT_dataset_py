import numpy as np 
import pandas as pd 

from sklearn import model_selection
from sklearn.metrics import mean_squared_error, make_scorer
from scipy.stats import mode, skew, kurtosis, entropy
from sklearn.ensemble import ExtraTreesRegressor

import matplotlib.pyplot as plt
import seaborn as sns

import dask.dataframe as dd
from dask.multiprocessing import get

from tqdm import tqdm, tqdm_notebook
tqdm.pandas(tqdm_notebook)


import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

import warnings
warnings.filterwarnings("ignore")

# Print all rows and columns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
IS_LOCAL = False

import os

if(IS_LOCAL):
    PATH="../input/santander-value-prediction-challenge/"
else:
    PATH="../input/"
print(os.listdir(PATH))
train = pd.read_csv(PATH+"train.csv")
test = pd.read_csv(PATH+"test.csv")
NLAGS = 29 #number of lags for leak calculation
all_cols = [f for f in train.columns if f not in ["ID", "target"]]
y = np.log1p(train["target"]).values
cols = ['f190486d6', '58e2e02e6', 'eeb9cd3aa', '9fd594eec', '6eef030c1',
        '15ace8c9f', 'fb0f5dbfe', '58e056e12', '20aa07010', '024c577b9',
        'd6bb78916', 'b43a7cfd5', '58232a6fb', '1702b5bf0', '324921c7b',
        '62e59a501', '2ec5b290f', '241f0f867', 'fb49e4212', '66ace2992',
        'f74e8f13d', '5c6487af1', '963a49cdc', '26fc93eb7', '1931ccfdd',
        '703885424', '70feb1494', '491b9ee45', '23310aa6f', 'e176a204a',
        '6619d81fc', '1db387535', 'fc99f9426', '91f701ba2', '0572565c2',
        '190db8488', 'adb64ff71', 'c47340d97', 'c5a231d81', '0ff32eb98']
def _get_leak(df, cols, lag=0):
    """ To get leak value, we do following:
       1. Get string of all values after removing first two time steps
       2. For all rows we shift the row by two steps and again make a string
       3. Just find rows where string from 2 matches string from 1
       4. Get 1st time step of row in 3 (Currently, there is additional condition to only fetch value if we got exactly one match in step 3)"""
    series_str = df[cols[lag+2:]].apply(lambda x: "_".join(x.round(2).astype(str)), axis=1)
    series_shifted_str = df[cols].shift(lag+2, axis=1)[cols[lag+2:]].apply(lambda x: "_".join(x.round(2).astype(str)), axis=1)
    target_rows = series_shifted_str.progress_apply(lambda x: np.where(x == series_str)[0])
    target_vals = target_rows.apply(lambda x: df.loc[x[0], cols[lag]] if len(x)==1 else 0)
    return target_vals

def get_all_leak(df, cols=None, nlags=15):
    """
    We just recursively fetch target value for different lags
    """
    df =  df.copy()
    for i in range(nlags):
        print("Processing lag {}".format(i))
        df["leaked_target_"+str(i)] = _get_leak(df, cols, i)
    return df
test["target"] = train["target"].mean()
all_df = pd.concat([train[["ID", "target"] + cols], test[["ID", "target"]+ cols]]).reset_index(drop=True)
all_df.head()
all_df = get_all_leak(all_df, cols=cols, nlags=NLAGS)
leaky_cols = ["leaked_target_"+str(i) for i in range(NLAGS)]
train = train.join(all_df.set_index("ID")[leaky_cols], on="ID", how="left")
test = test.join(all_df.set_index("ID")[leaky_cols], on="ID", how="left")
train[["target"]+leaky_cols].head(10)
train["nz_mean"] = train[all_cols].apply(lambda x: np.expm1(np.log1p(x[x!=0]).mean()), axis=1)
test["nz_mean"] = test[all_cols].apply(lambda x: np.expm1(np.log1p(x[x!=0]).mean()), axis=1)
train["compiled_leak"] = 0
test["compiled_leak"] = 0
for i in range(NLAGS):
    train.loc[train["compiled_leak"] == 0, "compiled_leak"] = train.loc[train["compiled_leak"] == 0, "leaked_target_"+str(i)]
    test.loc[test["compiled_leak"] == 0, "compiled_leak"] = test.loc[test["compiled_leak"] == 0, "leaked_target_"+str(i)]
    
print("Leak values found in train and test ", sum(train["compiled_leak"] > 0), sum(test["compiled_leak"] > 0))
print("% of correct leaks values in train ", sum(train["compiled_leak"] == train["target"])/sum(train["compiled_leak"] > 0))
train.loc[train["compiled_leak"] == 0, "compiled_leak"] = train.loc[train["compiled_leak"] == 0, "nz_mean"]
test.loc[test["compiled_leak"] == 0, "compiled_leak"] = test.loc[test["compiled_leak"] == 0, "nz_mean"]
np.sqrt(mean_squared_error(y, np.log1p(train["compiled_leak"]).fillna(14.49)))

sub1 = test[["ID"]]
sub1["target"] = test["compiled_leak"]
NUMBER_KFOLDS  = 5
NFOLDS = 5 #folds number for CV
MAX_ROUNDS = 3000 #lgb iterations
EARLY_STOP = 100 #lgb early stop 
VERBOSE_EVAL = 200 #Print out metric result
train = pd.read_csv(PATH+"train.csv")
test = pd.read_csv(PATH+"test.csv")
all_cols = [c for c in train.columns if c not in ['ID', 'target']]
leak_col = []
for c in all_cols:
    leak1 = np.sum((train[c]==train['target']).astype(int))
    leak2 = np.sum((((train[c] - train['target']) / train['target']) < 0.05).astype(int))
    if leak1 > 30 and leak2 > 3500:
        leak_col.append(c)
print('Leak columns: ',len(leak_col))
print('Leak columns: ',leak_col)
col = list(leak_col)
train_lk = train[col +  ['ID', 'target']]
test_lk = test[col +  ['ID']]
for df in [train_lk, test_lk]:
    df["nz_mean"] = df[col].apply(lambda x: x[x!=0].mean(), axis=1)
    df["nz_max"] = df[col].apply(lambda x: x[x!=0].max(), axis=1)
    df["nz_min"] = df[col].apply(lambda x: x[x!=0].min(), axis=1)
    df["ez"] = df[col].apply(lambda x: len(x[x==0]), axis=1)
    df["mean"] = df[col].apply(lambda x: x.mean(), axis=1)
    df["max"] = df[col].apply(lambda x: x.max(), axis=1)
    df["min"] = df[col].apply(lambda x: x.min(), axis=1)
    df["kurtosis"] = df[col].apply(lambda x: x.kurtosis(), axis=1)
col += ['nz_mean', 'nz_max', 'nz_min', 'ez', 'mean', 'max', 'min', 'kurtosis']
for i in range(2, 100):
    train_lk['index'+str(i)] = ((train_lk.index + 2) % i == 0).astype(int)
    test_lk['index'+str(i)] = ((test_lk.index + 2) % i == 0).astype(int)
    col.append('index'+str(i))
test_lk = pd.merge(test_lk, sub1, how='left', on='ID',)
from scipy.sparse import csr_matrix, vstack
train_lk = train_lk.replace(0, np.nan)
test_lk = test_lk.replace(0, np.nan)
train_lk = pd.concat((train_lk, test_lk), axis=0, ignore_index=True)
test_lk['target'] = 0.0
folds = NFOLDS
for fold in range(folds):
    x1, x2, y1, y2 = model_selection.train_test_split(train_lk[col], 
                                                      np.log1p(train_lk.target.values), 
                                                      test_size=0.20, 
                                                      random_state=fold)
    params = {'learning_rate': 0.02,
              'max_depth': 7, 
              'boosting': 'gbdt', 
              'objective': 'regression', 
              'metric': 'rmse', 
              'is_training_metric': True, 
              'feature_fraction': 0.9, 
              'bagging_fraction': 0.8, 
              'bagging_freq': 5, 
              'seed':fold}
    model = lgb.train(params, 
                      lgb.Dataset(x1, label=y1), 
                      MAX_ROUNDS, 
                      lgb.Dataset(x2, label=y2), 
                      verbose_eval=VERBOSE_EVAL, 
                      early_stopping_rounds=EARLY_STOP)
    test_lk['target'] += np.expm1(model.predict(test_lk[col], 
                                num_iteration=model.best_iteration))
test_lk['target'] /= folds
sub1 = test_lk[['ID', 'target']]
#submission
test_lk[['ID', 'target']].to_csv('submission.csv', index=False)