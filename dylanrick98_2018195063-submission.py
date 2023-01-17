import gc
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import random
random.seed(42)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def reload_train():
    gc.collect()
    df = pd.read_csv('../input/train_v2.csv')
    invalid_match_ids = df[df['winPlacePErc'].isna()]['matchId'].values
    df = df[-df['matchId'].isin(invalid_match_ids)]
    return df

def reload_test():
    gc.collect()
    df = pd.read_csv('../input/test_v2.csv')
    return df
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

def run_train(preprocess):
    df = reload_train()
    df.drop(columns=['matchType'], inplace=True)
    
    df = preprocess(df)
    
    target = 'winPlacePerc'
    cols_to_drop = ['Id', 'groupId', 'matchId', target]
    cols_to_fit = [col for col in df.columns if col not in cols_to_drop]
    
    print(df.columns)
    print(cols_to_fit)
    model = XGBRegressor()
    model.fit(df[cols_to_fit], df[target],verbose=False)
    
def run_test(preprocess):
    df = reload_test()
    df.drop(columns=['matchType'], inplace=True)
    
    df = preprocess(df)
    print(df.columns)
    return df
def rank_by_team(df):
    cols_to_drop = ['Id', 'groupId', 'matchId', 'winPlacePerc']
    features = [col for col in df.columns if col not in cols_to_drop]
    agg = df.groupby(['matchId', 'gropuId'])[features].mean()
    agg = agg.groupby('matchId')[features].reank(pct=True)
    return df.merge(agg, suffixes=['', '_mean_rank'], how='left', on=['matchId', 'groupId'])
model = run_train(rank_by_team)
test = run_test(rank_by_team)
test_id = test.Id
cols_to_drop = ['Id', 'gropuId', 'matchId']
features = [col for col in test.columns if col not in cols_to_drop]
test = test[features]
test.columns
pred = model.predict(test)
pred.shape
pred_df = pd.DataFrame({'Id' : test_id, 'winPlacePerc' : pred})

pred_df.to_csv("submission.csv", index=False)
