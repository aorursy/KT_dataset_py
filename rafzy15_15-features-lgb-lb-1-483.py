import os

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold



import pandas as pd

import numpy as np 

import matplotlib.pyplot as plt

import seaborn as sns

from tqdm import tqdm_notebook



import lightgbm as lgb



import scipy as sp

from scipy.fftpack import fft

from tsfresh.feature_extraction import feature_calculators



import gc

%matplotlib inline

print(os.listdir("../input"))
%time

train_df = pd.read_csv(os.path.join("../input",'train.csv'), dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})
train_df.shape
rows = 150000

segments = int(np.floor(train_df.shape[0] / rows))

print("Number of segments: ", segments)
#Prepare empty frame

train_X = pd.DataFrame(index=range(segments), dtype=np.float64)

train_y = pd.DataFrame(index=range(segments), dtype=np.float64, columns=['time_to_failure'])
def create_features(seg_id,seg, X):

    xc = seg["acoustic_data"]

    

    X.loc[seg_id,"num_peaks_1"] = feature_calculators.number_peaks(xc,1)

    X.loc[seg_id,"num_peaks_5"] = feature_calculators.number_peaks(xc,5)

    X.loc[seg_id,"num_peaks_10"] = feature_calculators.number_peaks(xc,10)

    

    X.loc[seg_id,"cid_ce_1"] = feature_calculators.cid_ce(xc, 1)

    

    X.loc[seg_id,'moment_4'] = sp.stats.moment(xc, 4)

    X.loc[seg_id,'moment_2'] = sp.stats.moment(xc, 2)

    

    X.loc[seg_id,"range_m1000_0"] = feature_calculators.range_count(xc, -1000, 0)

    X.loc[seg_id,"c_5"] = feature_calculators.c3(xc, 5)

    X.loc[seg_id,"mean"] = xc.mean()

    X.loc[seg_id,"binned_entropy_5"] = feature_calculators.binned_entropy(xc, 5)

    X.loc[seg_id,"autocorrelation_10"] = feature_calculators.autocorrelation(xc, 10)

    

    

    window_size = 10

    xc_rolled = xc.rolling(window_size)

    xc_rolled_var = xc_rolled.var().dropna()

    xc_rolled_mean = xc_rolled.mean().dropna()

        

    window_str = str(window_size)



    X.loc[seg_id,"rollingMean"+window_str+"_quantile_4"] = xc_rolled_mean.quantile(0.04)

    rolled_var_quantiles = xc_rolled_var.quantile([0.01,0.04])

    X.loc[seg_id,"rollingVar"+window_str+"_quantile_4"] = rolled_var_quantiles[0.04]

    X.loc[seg_id,"rollingVar"+window_str+"_quantile_1"] = rolled_var_quantiles[0.01]

    

    window_size = 10

    window_str = str(window_size)

    xc_rolled = xc.rolling(300)

    xc_rolled_var = xc_rolled.var().dropna()

    X.loc[seg_id,"rollingVar"+window_str+"_quantile_2"] = xc_rolled_var.quantile(0.02)
#create features from train

for seg_id in tqdm_notebook(range(segments)):

    seg = train_df.iloc[seg_id*rows:seg_id*rows+rows]

    create_features(seg_id, seg,train_X)

    train_y.loc[seg_id, 'time_to_failure'] = seg['time_to_failure'].values[-1]    
#create features from test

segment_names = [file for file in os.listdir("../input/test") if file.startswith("seg")]

test_df = pd.DataFrame(index=segment_names, dtype=np.float64)

test_df.index = test_df.index.str[:-4]

for file in tqdm_notebook(segment_names):

    seg_id = file[:-4]

    segment = pd.read_csv(os.path.join("../input/test",file),dtype={'acoustic_data': np.int16})

    create_features(seg_id,segment,test_df)

train_X.head()
train_X.shape
n_fold = 5

folds = KFold(n_splits=n_fold, shuffle=True, random_state=42)

params = {

    'lambda_l1': 0.012465994599126015, 

    'bagging_freq': 15, 

    'verbose': -1, 

    'min_data_in_leaf': 5, 

    'feature_fraction': 0.7143153769050614, 

    'objective': 'MAE',

    'lambda_l2': 0.055052283158846985, 

    'metric': 'MAE', 

    'bagging_fraction': 0.4871803105884792,

    'max_depth': -1, 

    'learning_rate': 0.007017896834582354, 

    'boosting_type': 'gbdt', 

    'num_leaves': 9

}
def train_lgb(train_X,train_y,test_df,params,folds):

    features_importance = pd.DataFrame({"features":train_X.columns,

                                        "importance":np.zeros(train_X.columns.shape[0])})

    predictions = pd.DataFrame({"seg_id":test_df.index,"time_to_failure":np.zeros(test_df.shape[0])})

    oof = np.zeros(train_X.shape[0])



    for train_idx,val_idx in folds.split(train_X,train_y):

        X_train,y_train = train_X.iloc[train_idx],train_y.iloc[train_idx]

        X_val,y_val = train_X.iloc[val_idx],train_y.iloc[val_idx]



        model = lgb.LGBMRegressor(**params, n_estimators = 20000,n_jobs=-1)

        model.fit(X_train,y_train,

                  eval_set=[(X_train,y_train),(X_val,y_val)], 

                  verbose=1000,

                  early_stopping_rounds=1000)



        oof[val_idx] = model.predict(X_val, num_iteration=model.best_iteration_)



        features_importance["importance"] += model.feature_importances_

        predictions["time_to_failure"] += model.predict(test_df, num_iteration=model.best_iteration_)

    return oof,predictions,features_importance



oof,predictions,features_importance = train_lgb(train_X,train_y,test_df,params,folds)
mean_absolute_error(train_y,oof)
features_importance["importance"] = features_importance["importance"]/5

predictions["time_to_failure"] = predictions["time_to_failure"]/5



plt.figure(figsize=(10,10))

ax = sns.barplot(x="importance", y="features", data=features_importance.sort_values(by="importance",ascending=False))
predictions.head()
predictions.to_csv("submission_lgb_15_col.csv",index=False)