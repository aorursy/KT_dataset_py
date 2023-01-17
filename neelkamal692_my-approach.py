import gc
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#### hi this is neel
import random
random.seed(42)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def reload_train():
    gc.collect()
    df = pd.read_csv('../input/train_V2.csv')
    invalid_match_ids = df[df['winPlacePerc'].isna()]['matchId'].values
    df = df[-df['matchId'].isin(invalid_match_ids)]
    return df

def reload_test():
    gc.collect()
    df = pd.read_csv('../input/test_V2.csv')
    return df
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import mean_absolute_error

def run_train(preprocess):
    df = reload_train()
    df.drop(columns=['matchType'], inplace=True)
    
    df = preprocess(df)

    target = 'winPlacePerc'
    cols_to_drop = ['Id', 'groupId', 'matchId', target]
    cols_to_fit = [col for col in df.columns if col not in cols_to_drop]
    
    
    
    df = SelectKBest(chi2, k=30).fit_transform(df[cols_to_fit], df[target])
    col = df.columns
    print(df.columns)                                             
    model = XGBRegressor()
    model.fit(df[cols_to_fit], df[target],verbose=False)
    return model,col

def run_test(preprocess):
    df = reload_test()
    df.drop(columns=['matchType'], inplace=True)
    
    df = preprocess(df)
    print(df.columns)
    return df
def rank_by_team(df):
    cols_to_drop = ['Id', 'groupId', 'matchId', 'winPlacePerc']
    features = [col for col in df.columns if col not in cols_to_drop]
    agg = df.groupby(['matchId', 'groupId'])[features].mean()
    agg = agg.groupby('matchId')[features].rank(pct=True)
    return df.merge(agg, suffixes=['', '_mean_rank'], how='left', on=['matchId', 'groupId'])
model,col = run_train(rank_by_team)
test = run_test(rank_by_team)
test_id = test.Id
cols_to_drop = ['Id', 'groupId', 'matchId']
features = [col for col in test.columns if col not in cols_to_drop]
test = test[features]
test.columns
pred = model.predict(test)
pred.shape
pred_df = pd.DataFrame({'Id' : test_id, 'winPlacePerc' : pred})
# Create submission file

pred_df.to_csv("submission.csv", index=False)
