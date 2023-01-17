import gc

import os

import time

import logging

import datetime

import warnings

import numpy as np

import pandas as pd

import scipy as sp

import seaborn as sns

import xgboost as xgb

import lightgbm as lgb

from scipy import stats

from scipy.signal import hann

from tqdm import tqdm_notebook

import matplotlib.pyplot as plt

from scipy.signal import hilbert

from scipy.signal import convolve

from sklearn.svm import NuSVR, SVR

from catboost import CatBoostRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import mean_absolute_error

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import KFold,StratifiedKFold, RepeatedKFold

warnings.filterwarnings("ignore")
!pip install tsfresh


from tsfresh.feature_extraction import feature_calculators
# Install tpot on the server

!pip install tpot

from tpot.builtins import StackingEstimator, ZeroCount

# We load the train.csv

PATH="../input/"

train_df = pd.read_csv(PATH + 'train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})

train_df.head()
# We define the number of rows in each segment as the same number of rows in the real test segments (150000 rows)

rows = 150000

segments = int(np.floor(train_df.shape[0] / rows))

print("Number of segments: ", segments)
train_X = pd.DataFrame(index=range(segments), dtype=np.float64)

train_y = pd.DataFrame(index=range(segments), dtype=np.float64, columns=['time_to_failure'])
def add_trend_feature(arr, abs_values=False):

    idx = np.array(range(len(arr)))

    if abs_values:

        arr = np.abs(arr)

    lr = LinearRegression()

    lr.fit(idx.reshape(-1, 1), arr)

    return lr.coef_[0]



def classic_sta_lta(x, length_sta, length_lta):

    sta = np.cumsum(x ** 2)

    # Convert to float

    sta = np.require(sta, dtype=np.float)

    # Copy for LTA

    lta = sta.copy()

    # Compute the STA and the LTA

    sta[length_sta:] = sta[length_sta:] - sta[:-length_sta]

    sta /= length_sta

    lta[length_lta:] = lta[length_lta:] - lta[:-length_lta]

    lta /= length_lta

    # Pad zeros

    sta[:length_lta - 1] = 0

    # Avoid division by zero by setting zero values to tiny float

    dtiny = np.finfo(0.0).tiny

    idx = lta < dtiny

    lta[idx] = dtiny

    return sta / lta
def create_features(seg_id, xc, X):

    zc = np.fft.fft(xc)

    X.loc[seg_id,'cid_ce1']=feature_calculators.cid_ce(xc, 1) #Great

    #FFT transform values



    

    realFFT = np.real(zc)

    imagFFT = np.imag(zc)

    X.loc[seg_id, 'Rmin'] = realFFT.min()

    X.loc[seg_id, 'Imin'] = imagFFT.min()

    X.loc[seg_id, 'Rmax_last_15000'] = realFFT[-15000:].max()

    X.loc[seg_id, 'Rmin_last_15000'] = realFFT[-15000:].min()

    

    X.loc[seg_id, 'autocorrelation_10'] = feature_calculators.autocorrelation(xc, 10)

    

    X.loc[seg_id, 'Hilbert_mean'] = np.abs(hilbert(xc)).mean()

    X.loc[seg_id, 'classic_sta_lta1_mean'] = classic_sta_lta(xc, 500, 10000).mean()

    no_of_std = 2

    X.loc[seg_id, 'Moving_average_700_mean'] = xc.rolling(window=700).mean().mean(skipna=True)

    X.loc[seg_id, 'Moving_average_400_mean'] = xc.rolling(window=400).mean().mean(skipna=True)

    X.loc[seg_id, 'MA_700MA_std_mean'] = xc.rolling(window=700).std().mean()

    X.loc[seg_id,'MA_700MA_BB_high_mean'] = (X.loc[seg_id, 'Moving_average_700_mean'] + no_of_std * X.loc[seg_id, 'MA_700MA_std_mean']).mean()

    X.loc[seg_id,'MA_700MA_BB_low_mean'] = (X.loc[seg_id, 'Moving_average_700_mean'] - no_of_std * X.loc[seg_id, 'MA_700MA_std_mean']).mean()

    X.loc[seg_id, 'MA_400MA_std_mean'] = xc.rolling(window=400).std().mean()

    X.loc[seg_id,'MA_400MA_BB_high_mean'] = (X.loc[seg_id, 'Moving_average_400_mean'] + no_of_std * X.loc[seg_id, 'MA_400MA_std_mean']).mean()

    X.loc[seg_id,'MA_400MA_BB_low_mean'] = (X.loc[seg_id, 'Moving_average_400_mean'] - no_of_std * X.loc[seg_id, 'MA_400MA_std_mean']).mean()

    X.loc[seg_id, 'MA_1000MA_std_mean'] = xc.rolling(window=1000).std().mean()

    #added scipy features next to numpy

    X.loc[seg_id,'kstat2']=sp.stats.kstat(xc, 2)

    X.loc[seg_id,'kstat3']=sp.stats.kstat(xc, 3)

    

    

    #using feature_calculators

    X.loc[seg_id,'abs_sum_changes']=feature_calculators.absolute_sum_of_changes(xc)



    X.loc[seg_id,'mean_abs_change']=feature_calculators.mean_abs_change(xc)

 

    X.loc[seg_id,'ratio_value_number_to_timeseries']=feature_calculators.ratio_value_number_to_time_series_length(xc)



    X.loc[seg_id,'ac10']=feature_calculators.autocorrelation(xc, 10)

    X.loc[seg_id,'ac50']=feature_calculators.autocorrelation(xc, 50)



    

    X.loc[seg_id,'npeaks_0']=feature_calculators.number_crossing_m(xc, 0)

    X.loc[seg_id,'npeaks_10']=feature_calculators.number_peaks(xc, 10)

    X.loc[seg_id,'npeaks_50']=feature_calculators.number_peaks(xc, 50)

    X.loc[seg_id,'npeaks_100']=feature_calculators.number_peaks(xc, 100)

    

    X.loc[seg_id,'lsbm']=feature_calculators.longest_strike_below_mean(xc)

    X.loc[seg_id,'lsam']=feature_calculators.longest_strike_above_mean(xc)

    

    

    X.loc[seg_id, 'MA_15000MA_std_mean'] = xc.rolling(window=15000).std().mean() #low improvement

    for windows in [10]:

        x_roll_std = xc.rolling(windows).std().dropna().values

        x_roll_mean = xc.rolling(windows).mean().dropna().values

        

        X.loc[seg_id, 'ave_roll_std_' + str(windows)] = x_roll_std.mean()

        X.loc[seg_id, 'q01_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.01)

        X.loc[seg_id, 'q05_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.05)

        X.loc[seg_id, 'q95_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.95)

        X.loc[seg_id, 'q01_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.01)

        X.loc[seg_id, 'q05_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.05)

        X.loc[seg_id, 'q95_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.95)

        X.loc[seg_id, 'q99_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.99)

    for windows in [100]:

        X.loc[seg_id, 'ave_roll_std_' + str(windows)] = x_roll_std.mean()

        X.loc[seg_id, 'min_roll_std_' + str(windows)] = x_roll_std.min()

        X.loc[seg_id, 'q01_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.01)

        X.loc[seg_id, 'q05_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.05)

        X.loc[seg_id, 'q95_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.95)



    for windows in [1000]:

        X.loc[seg_id, 'ave_roll_std_' + str(windows)] = x_roll_std.mean()

        X.loc[seg_id, 'min_roll_std_' + str(windows)] = x_roll_std.min()

        X.loc[seg_id, 'q01_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.01)

        X.loc[seg_id, 'q05_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.05)

        X.loc[seg_id, 'q95_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.95)



#def create_features(seg_id, xc, X):

#    zc = np.fft.fft(xc)

#    X.loc[seg_id,'cid_ce1']=feature_calculators.cid_ce(xc, 1) #Great

        #FFT transform values

#    X.loc[seg_id, 'max'] = xc.max() #GREAT

#    X.loc[seg_id, 'min'] = xc.min()

    

#    realFFT = np.real(zc)

#    imagFFT = np.imag(zc)

#    X.loc[seg_id, 'Rmax'] = realFFT.max() # Great

#    X.loc[seg_id, 'Rmin'] = realFFT.min()

#    X.loc[seg_id, 'Imin'] = imagFFT.min()

#    X.loc[seg_id, 'Rmin_last_5000'] = realFFT[-5000:].min()

#    X.loc[seg_id, 'Rmin_last_15000'] = realFFT[-15000:].min()

   

#    X.loc[seg_id, 'min_first_50000'] = xc[:50000].min()

#    X.loc[seg_id, 'min_last_50000'] = xc[-50000:].min()

#    X.loc[seg_id, 'min_first_10000'] = xc[:10000].min()

#    X.loc[seg_id, 'min_last_10000'] = xc[-10000:].min()



    #X.loc[seg_id, 'autocorrelation_10'] = feature_calculators.autocorrelation(xc, 10)

    

#    X.loc[seg_id, 'classic_sta_lta3_mean'] = classic_sta_lta(xc, 3333, 6666).mean()

   # no_of_std = 2 

  #  X.loc[seg_id, 'Moving_average_700_mean'] = xc.rolling(window=700).mean().mean(skipna=True)

  #  X.loc[seg_id, 'MA_700MA_std_mean'] = xc.rolling(window=700).std().mean()

  #  X.loc[seg_id, 'Moving_average_400_mean'] = xc.rolling(window=400).mean().mean(skipna=True)

  #  X.loc[seg_id, 'MA_400MA_std_mean'] = xc.rolling(window=400).std().mean()



  #  X.loc[seg_id,'MA_700MA_BB_low_mean'] = (X.loc[seg_id, 'Moving_average_700_mean'] - no_of_std * X.loc[seg_id, 'MA_700MA_std_mean']).mean()



  #  X.loc[seg_id,'MA_400MA_BB_low_mean'] = (X.loc[seg_id, 'Moving_average_400_mean'] - no_of_std * X.loc[seg_id, 'MA_400MA_std_mean']).mean()



    #added scipy features next to numpy

  #  X.loc[seg_id,'kstat3']=sp.stats.kstat(xc, 3)

  #  X.loc[seg_id,'kstat4']=sp.stats.kstat(xc, 4)

 



 #    X.loc[seg_id,'ac10']=feature_calculators.autocorrelation(xc, 10)

    

  #  X.loc[seg_id,'be_20']=feature_calculators.binned_entropy(xc, 20)

  #  X.loc[seg_id,'be_50']=feature_calculators.binned_entropy(xc, 50)

  #  X.loc[seg_id,'be_80']=feature_calculators.binned_entropy(xc, 80)

  #  X.loc[seg_id,'be_100']=feature_calculators.binned_entropy(xc, 100)

    



#    for windows in [10,100]:

 #       x_roll_std = xc.rolling(windows).std().dropna().values

  #      x_roll_mean = xc.rolling(windows).mean().dropna().values

   #     X.loc[seg_id, 'min_roll_mean_' + str(windows)] = x_roll_mean.min()

    #    X.loc[seg_id, 'q01_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.01)

     #   X.loc[seg_id, 'q05_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.05)





    #for windows in [1000]:

     #   x_roll_std = xc.rolling(windows).std().dropna().values

      #  x_roll_mean = xc.rolling(windows).mean().dropna().values   

       # X.loc[seg_id, 'min_roll_mean_' + str(windows)] = x_roll_mean.min()







# iterate over all segments

for seg_id in tqdm_notebook(range(segments)):

    seg = train_df.iloc[seg_id*rows:seg_id*rows + rows]

    create_features(seg_id, seg['acoustic_data'], train_X)

    train_y.loc[seg_id, 'time_to_failure'] = seg['time_to_failure'].values[-1]
scaler = StandardScaler()

scaler.fit(train_X)

scaled_train_X = pd.DataFrame(scaler.transform(train_X), columns=train_X.columns)
import statsmodels.api as sm

from scipy.stats.mstats import zscore


#betas = []

#names = []

#for column in kk:

#    print(column)

    #print(kk[column].isnull().values.any())

#    result = sm.OLS(zscore(train_y.values), zscore(kk[column])).fit()

#    print(result.params)

#    betas.append(result.params)

    #if result.params > 0.25 or result.params < -0.25:

#    if result.params < 0.25 and result.params > -0.25:

#        names.append(column)

    
scaled_train_X = scaled_train_X.drop(['Moving_average_700_mean'],axis=1)

##scaled_train_X = scaled_train_X.drop(['MA_700MA_std_mean'],axis=1)

scaled_train_X = scaled_train_X.drop(['Moving_average_400_mean'],axis=1)

#scaled_train_X = scaled_train_X.drop(['MA_400MA_std_mean'],axis=1)


#kkk = kk

#for n in names:

#    kkk = kkk.drop([n], axis =1)

#for column in kkk:

#    print(column)
#scaled_train_X= scaled_train_X.drop(['max'],axis=1)
#scaled_train_X= scaled_train_X.drop(['Rmax'],axis=1)
submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')

test_X = pd.DataFrame(columns=train_X.columns, dtype=np.float32, index=submission.index)

for seg_id in tqdm_notebook(test_X.index):

    seg = pd.read_csv('../input/test/' + seg_id + '.csv')

    create_features(seg_id, seg['acoustic_data'], test_X)

scaled_test_X = pd.DataFrame(scaler.transform(test_X), columns=test_X.columns)
scaled_test_X = scaled_test_X.drop(['Moving_average_700_mean'],axis=1)

scaled_test_X = scaled_test_X.drop(['Moving_average_400_mean'],axis=1)

n_fold = 5

folds = KFold(n_splits=n_fold, shuffle=True, random_state=42)

train_columns = scaled_train_X.columns.values

#train_columns = kkk.columns.values
#best was

#'num_leaves': 41, 

#         'min_data_in_leaf': 20,



params = {'num_leaves': 240, 

         'min_data_in_leaf': 180,

         'objective':'regression',

         'max_depth': -1,

         'learning_rate': 0.001,

         "boosting": "gbdt",

         "feature_fraction": 0.91,

         "bagging_freq": 2,

         "bagging_fraction": 0.91,

         "bagging_seed": 42,

         "metric": 'mae',

         "lambda_l1": 0.1,

         "verbosity": -1,

         "random_state": 42}

oof = np.zeros(len(scaled_train_X))

#oof = np.zeros(len(kkk))





predictions = np.zeros(len(scaled_test_X))

feature_importance_df = pd.DataFrame()

#run model

#for fold_, (trn_idx, val_idx) in enumerate(folds.split(scaled_train_X,train_y.values)):

for fold_, (trn_idx, val_idx) in enumerate(folds.split(scaled_train_X,train_y.values)):

    strLog = "fold {}".format(fold_)

    print(strLog)

    

    X_tr, X_val = scaled_train_X.iloc[trn_idx], scaled_train_X.iloc[val_idx]

    y_tr, y_val = train_y.iloc[trn_idx], train_y.iloc[val_idx]



    model = lgb.LGBMRegressor(**params, n_estimators = 20000, n_jobs = -1)

    model.fit(X_tr, 

              y_tr, 

              eval_set=[(X_tr, y_tr), (X_val, y_val)], 

              eval_metric='mae',

              verbose=1000, 

              early_stopping_rounds=900)

    oof[val_idx] = model.predict(X_val, num_iteration=model.best_iteration_)

    #feature importance

    fold_importance_df = pd.DataFrame()

    fold_importance_df["Feature"] = train_columns

    fold_importance_df["importance"] = model.feature_importances_[:len(train_columns)]

    fold_importance_df["fold"] = fold_ + 1

    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    #predictions

    predictions += model.predict(scaled_test_X, num_iteration=model.best_iteration_) / folds.n_splits
cols = (feature_importance_df[["Feature", "importance"]]

        .groupby("Feature")

        .mean()

        .sort_values(by="importance", ascending=False)[:200].index)

best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]



plt.figure(figsize=(14,26))

sns.barplot(x="importance", y="Feature", data=best_features.sort_values(by="importance",ascending=False))

plt.title('LightGBM Features (averaged over folds)')

plt.tight_layout()

plt.savefig('lgbm_importances.png')
plt.show()
submission.time_to_failure = predictions

submission.to_csv('submission.csv',index=True)