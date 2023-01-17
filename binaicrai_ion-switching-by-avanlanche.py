# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



import warnings

warnings.filterwarnings('ignore')



import time

import datetime

import seaborn as sns

import math

import matplotlib.pyplot as plt



import gc

import scipy.fftpack

from scipy.signal import butter,filtfilt,freqz



import scipy as sp

import scipy.fftpack



from sklearn import *

from sklearn.metrics import f1_score

import lightgbm as lgb

import xgboost as xgb

from catboost import Pool,CatBoostRegressor



from sklearn.model_selection import KFold
def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:

        if col != 'time':

            col_type = df[col].dtypes

            if col_type in numerics:

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

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df



def import_data(file):

    """create a dataframe and optimize its memory usage"""

    df = pd.read_csv(file, parse_dates=True, keep_date_col=True)

    df = reduce_mem_usage(df)

    return df
train = import_data('../input/liverpool-ion-switching/train.csv')

test = import_data('../input/liverpool-ion-switching/test.csv')

ss = pd.read_csv('../input/liverpool-ion-switching/sample_submission.csv')
display(train.head())

display(test.head())
def butter_lowpass_filter(data, cutoff, fs, order):

    normal_cutoff = cutoff / nyq

    # Get the filter coefficients 

    b, a = butter(order, normal_cutoff, btype='low', analog=False)

    y = filtfilt(b, a, data)

    return y



def signaltonoise(a, axis=0, ddof=0):

    a = np.asanyarray(a)

    m = a.mean(axis)

    sd = a.std(axis=axis, ddof=ddof)

    return np.where(sd == 0, 0, ((m*m)/(sd*sd))) ## Slight change over here
batch_size = 500000

num_batches = 10

res = 1000 # Resolution of signal plots



fs = 10000       # sample rate, 10kHz

nyq = 0.5 * fs  # Nyquist Frequency

cutoff_freq_sweep = range(250,5000,10) # Sweeping from 250 to 4750 Hz for SNR measurement
train_copy = train.copy()
plt.figure(figsize=(20,10));



# Filter requirements.

order = 20  

SNR = np.zeros(len(cutoff_freq_sweep))



for batch in range(num_batches):

    for index,cut in enumerate(cutoff_freq_sweep): 

        signal_lpf = butter_lowpass_filter(train.signal[batch_size*(batch):batch_size*(batch+1)], cut, fs, order)

        SNR[index] = signaltonoise(signal_lpf)

    

    plt.plot(cutoff_freq_sweep,SNR)



plt.title('Signal-to-Noise Ratio Per Batch with Low Pass Filter')    

plt.xlabel('Frequency')

plt.ylabel('SNR')

plt.legend(['Batch 1','Batch 2','Batch 3','Batch 4','Batch 5','Batch 6','Batch 7','Batch 8','Batch 9','Batch 10',])
cutoff_freq_sweep = range(250,700,10) # Expanding a bit



plt.figure(figsize=(25,10));



# Filter requirements.

order = 20  

SNR = np.zeros(len(cutoff_freq_sweep))



for batch in range(num_batches):

    for index,cut in enumerate(cutoff_freq_sweep): 

        signal_lpf = butter_lowpass_filter(train.signal[batch_size*(batch):batch_size*(batch+1)], cut, fs, order)

        SNR[index] = signaltonoise(signal_lpf)

    

    plt.plot(cutoff_freq_sweep,SNR)



plt.title('Expansion of Signal-to-Noise Ratio Per Batch with Low Pass Filter')    

plt.xlabel('Frequency')

plt.ylabel('SNR')

plt.legend(['Batch 1','Batch 2','Batch 3','Batch 4','Batch 5','Batch 6','Batch 7','Batch 8','Batch 9','Batch 10',])
lpf_cutoff = 600 # Sticking with 600

batch = 1



signal_lpf_batch_1 = butter_lowpass_filter(train.signal[batch_size*(batch-1):batch_size*batch], lpf_cutoff, fs, order)



fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(25, 10))

ax[0].plot(range(0,batch_size,res),train.open_channels[batch_size*(batch-1):batch_size*batch:res],color='g')

ax[1].plot(range(0,batch_size,res),train.signal[batch_size*(batch-1):batch_size*batch:res])

ax[1].plot(range(0,batch_size,res),signal_lpf_batch_1[::res])



ax[0].legend(['Batch-1: open_channels'])

ax[1].legend(['signal', 'filtered signal'])
batch = 2



signal_lpf_batch_2 = butter_lowpass_filter(train.signal[batch_size*(batch-1):batch_size*batch], lpf_cutoff, fs, order)



fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(25, 10))

ax[0].plot(range(0,batch_size,res),train.open_channels[batch_size*(batch-1):batch_size*batch:res],color='g')

ax[1].plot(range(0,batch_size,res),train.signal[batch_size*(batch-1):batch_size*batch:res])

ax[1].plot(range(0,batch_size,res),signal_lpf_batch_2[::res])



ax[0].legend(['Batch-2: open_channels'])

ax[1].legend(['signal', 'filtered signal'])
batch = 3



signal_lpf_batch_3 = butter_lowpass_filter(train.signal[batch_size*(batch-1):batch_size*batch], lpf_cutoff, fs, order)



fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(25, 10))

ax[0].plot(range(0,batch_size,res),train.open_channels[batch_size*(batch-1):batch_size*batch:res],color='g')

ax[1].plot(range(0,batch_size,res),train.signal[batch_size*(batch-1):batch_size*batch:res])

ax[1].plot(range(0,batch_size,res),signal_lpf_batch_3[::res])



ax[0].legend(['Batch-3: open_channels'])

ax[1].legend(['signal', 'filtered signal'])
batch = 4



signal_lpf_batch_4 = butter_lowpass_filter(train.signal[batch_size*(batch-1):batch_size*batch], lpf_cutoff, fs, order)



fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(25, 10))

ax[0].plot(range(0,batch_size,res),train.open_channels[batch_size*(batch-1):batch_size*batch:res],color='g')

ax[1].plot(range(0,batch_size,res),train.signal[batch_size*(batch-1):batch_size*batch:res])

ax[1].plot(range(0,batch_size,res),signal_lpf_batch_4[::res])



ax[0].legend(['Batch-4: open_channels'])

ax[1].legend(['signal', 'filtered signal'])
batch = 5



signal_lpf_batch_5 = butter_lowpass_filter(train.signal[batch_size*(batch-1):batch_size*batch], lpf_cutoff, fs, order)



fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(25, 10))

ax[0].plot(range(0,batch_size,res),train.open_channels[batch_size*(batch-1):batch_size*batch:res],color='g')

ax[1].plot(range(0,batch_size,res),train.signal[batch_size*(batch-1):batch_size*batch:res])

ax[1].plot(range(0,batch_size,res),signal_lpf_batch_5[::res])



ax[0].legend(['Batch-5: open_channels'])

ax[1].legend(['signal', 'filtered signal'])
batch = 6



signal_lpf_batch_6 = butter_lowpass_filter(train.signal[batch_size*(batch-1):batch_size*batch], lpf_cutoff, fs, order)



fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(25, 10))

ax[0].plot(range(0,batch_size,res),train.open_channels[batch_size*(batch-1):batch_size*batch:res],color='g')

ax[1].plot(range(0,batch_size,res),train.signal[batch_size*(batch-1):batch_size*batch:res])

ax[1].plot(range(0,batch_size,res),signal_lpf_batch_6[::res])



ax[0].legend(['Batch-6: open_channels'])

ax[1].legend(['signal', 'filtered signal'])
batch = 7



signal_lpf_batch_7 = butter_lowpass_filter(train.signal[batch_size*(batch-1):batch_size*batch], lpf_cutoff, fs, order)



fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(25, 10))

ax[0].plot(range(0,batch_size,res),train.open_channels[batch_size*(batch-1):batch_size*batch:res],color='g')

ax[1].plot(range(0,batch_size,res),train.signal[batch_size*(batch-1):batch_size*batch:res])

ax[1].plot(range(0,batch_size,res),signal_lpf_batch_7[::res])



ax[0].legend(['Batch-7: open_channels'])

ax[1].legend(['signal', 'filtered signal'])
batch = 8



signal_lpf_batch_8 = butter_lowpass_filter(train.signal[batch_size*(batch-1):batch_size*batch], lpf_cutoff, fs, order)



fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(25, 10))

ax[0].plot(range(0,batch_size,res),train.open_channels[batch_size*(batch-1):batch_size*batch:res],color='g')

ax[1].plot(range(0,batch_size,res),train.signal[batch_size*(batch-1):batch_size*batch:res])

ax[1].plot(range(0,batch_size,res),signal_lpf_batch_8[::res])



ax[0].legend(['Batch-8: open_channels'])

ax[1].legend(['signal', 'filtered signal'])
batch = 9



signal_lpf_batch_9 = butter_lowpass_filter(train.signal[batch_size*(batch-1):batch_size*batch], lpf_cutoff, fs, order)



fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(25, 10))

ax[0].plot(range(0,batch_size,res),train.open_channels[batch_size*(batch-1):batch_size*batch:res],color='g')

ax[1].plot(range(0,batch_size,res),train.signal[batch_size*(batch-1):batch_size*batch:res])

ax[1].plot(range(0,batch_size,res),signal_lpf_batch_9[::res])



ax[0].legend(['Batch-9: open_channels'])

ax[1].legend(['signal', 'filtered signal'])
batch = 10



signal_lpf_batch_10 = butter_lowpass_filter(train.signal[batch_size*(batch-1):batch_size*batch], lpf_cutoff, fs, order)



fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(25, 10))

ax[0].plot(range(0,batch_size,res),train.open_channels[batch_size*(batch-1):batch_size*batch:res],color='g')

ax[1].plot(range(0,batch_size,res),train.signal[batch_size*(batch-1):batch_size*batch:res])

ax[1].plot(range(0,batch_size,res),signal_lpf_batch_10[::res])



ax[0].legend(['Batch-10: open_channels'])

ax[1].legend(['signal', 'filtered signal'])
batch = 1

train['signal'][batch_size*(batch-1):batch_size*batch] = signal_lpf_batch_1

train['signal_undrifted'] = train['signal']
batch = 2

train['signal'][batch_size*(batch-1):batch_size*batch] = signal_lpf_batch_2

train['signal_undrifted'] = train['signal']
batch = 3

train['signal'][batch_size*(batch-1):batch_size*batch] = signal_lpf_batch_3

train['signal_undrifted'] = train['signal']
batch = 7

train['signal'][batch_size*(batch-1):batch_size*batch] = signal_lpf_batch_7

train['signal_undrifted'] = train['signal']
# Test Data

test['signal_undrifted'] = test['signal']
%%time

def features(df):

    df = df.sort_values(by=['time']).reset_index(drop=True)

    df.index = ((df.time * 10_000) - 1).values

    df['batch'] = df.index // 25_000

    df['batch_index'] = df.index  - (df.batch * 25_000)

    df['batch_slices'] = df['batch_index']  // 2500

    df['batch_slices2'] = df.apply(lambda r: '_'.join([str(r['batch']).zfill(3), str(r['batch_slices']).zfill(3)]), axis=1)

    

    for c in ['batch','batch_slices2']:

        d = {}

        d['mean'+c] = df.groupby([c])['signal'].mean()

        d['median'+c] = df.groupby([c])['signal'].median()

        d['max'+c] = df.groupby([c])['signal'].max()

        d['min'+c] = df.groupby([c])['signal'].min()

        d['std'+c] = df.groupby([c])['signal'].std()

        d['mean_abs_chg'+c] = df.groupby([c])['signal'].apply(lambda x: np.mean(np.abs(np.diff(x))))

        d['abs_max'+c] = df.groupby([c])['signal'].apply(lambda x: np.max(np.abs(x)))

        d['abs_min'+c] = df.groupby([c])['signal'].apply(lambda x: np.min(np.abs(x)))

        d['range'+c] = d['max'+c] - d['min'+c]

        d['maxtomin'+c] = d['max'+c] / d['min'+c]

        d['abs_avg'+c] = (d['abs_min'+c] + d['abs_max'+c]) / 2

        for v in d:

            df[v] = df[c].map(d[v].to_dict())



    

    # add shifts_1

    df['signal_shift_+1'] = [0,] + list(df['signal'].values[:-1])

    df['signal_shift_-1'] = list(df['signal'].values[1:]) + [0]

    for i in df[df['batch_index']==0].index:

        df['signal_shift_+1'][i] = np.nan

    for i in df[df['batch_index']==49999].index:

        df['signal_shift_-1'][i] = np.nan

    

    # add shifts_2

    df['signal_shift_+2'] = [0,] + [1,] + list(df['signal'].values[:-2])

    df['signal_shift_-2'] = list(df['signal'].values[2:]) + [0] + [1]

    for i in df[df['batch_index']==0].index:

        df['signal_shift_+2'][i] = np.nan

    for i in df[df['batch_index']==1].index:

        df['signal_shift_+2'][i] = np.nan

    for i in df[df['batch_index']==49999].index:

        df['signal_shift_-2'][i] = np.nan

    for i in df[df['batch_index']==49998].index:

        df['signal_shift_-2'][i] = np.nan

        

      

    df = df.replace([np.inf, -np.inf], np.nan)    

    df.fillna(0, inplace=True)

    gc.collect()

    return df



train = features(train)

test = features(test)
train = reduce_mem_usage(train)

test = reduce_mem_usage(test)
def f1_score_calc(y_true, y_pred): 

    return f1_score(y_true, y_pred, average="macro")



def lgb_Metric(preds, dtrain):

    labels = dtrain.get_label()

    preds = np.round(np.clip(preds, 0, 10)).astype(int)

    score = f1_score(labels, preds, average="macro")

    return ('KaggleMetric', score, True)





def train_model_classification(X, X_test, y, params, model_type='lgb', eval_metric='f1score',

                               columns=None, plot_feature_importance=False, model=None,

                               verbose=50, early_stopping_rounds=200, n_estimators=2000):



    columns = X.columns if columns == None else columns

    X_test = X_test[columns]

    

    # to set up scoring parameters

    metrics_dict = {

                    'f1score': {'lgb_metric_name': lgb_Metric,}

                   }

    

    result_dict = {}

    

    # out-of-fold predictions on train data

    oof = np.zeros(len(X) )

    

    # averaged predictions on train data

    prediction = np.zeros((len(X_test)))

    

    # list of scores on folds

    scores = []

    feature_importance = pd.DataFrame()

    

    # split and train on folds

    '''for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):

        print(f'Fold {fold_n + 1} started at {time.ctime()}')

        if type(X) == np.ndarray:

            X_train, X_valid = X[columns][train_index], X[columns][valid_index]

            y_train, y_valid = y[train_index], y[valid_index]

        else:

            X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]

            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]'''

            

    if True:        

        X_train, X_valid, y_train, y_valid = model_selection.train_test_split(X, y, test_size=0.3, random_state=7)    

            

        if model_type == 'lgb':

            #model = lgb.LGBMClassifier(**params, n_estimators=n_estimators)

            #model.fit(X_train, y_train, 

            #        eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric=metrics_dict[eval_metric]['lgb_metric_name'],

            #       verbose=verbose, early_stopping_rounds=early_stopping_rounds)

            

            model = lgb.train(params, lgb.Dataset(X_train, y_train),

                              n_estimators,  lgb.Dataset(X_valid, y_valid),

                              verbose_eval=verbose, early_stopping_rounds=early_stopping_rounds, feval=lgb_Metric)

            

            

            preds = model.predict(X, num_iteration=model.best_iteration) #model.predict(X_valid) 



            y_pred = model.predict(X_test, num_iteration=model.best_iteration)

            

        if model_type == 'xgb':

            train_set = xgb.DMatrix(X_train, y_train)

            val_set = xgb.DMatrix(X_valid, y_valid)

            model = xgb.train(params, train_set, num_boost_round=2222, evals=[(train_set, 'train'), (val_set, 'val')], 

                                     verbose_eval=verbose, early_stopping_rounds=early_stopping_rounds)

            

            preds = model.predict(xgb.DMatrix(X)) 



            y_pred = model.predict(xgb.DMatrix(X_test))

            



        if model_type == 'cat':

            # Initialize CatBoostRegressor

            model = CatBoostRegressor(params)

            # Fit model

            model.fit(X_train, y_train)

            # Get predictions

            y_pred_valid = np.round(np.clip(preds, 0, 10)).astype(int)



            y_pred = model.predict(X_test, num_iteration=model.best_iteration)

            y_pred = np.round(np.clip(y_pred, 0, 10)).astype(int)



 

        oof = preds

        

        scores.append(f1_score_calc(y, np.round(np.clip(preds,0,10)).astype(int) ) )



        prediction += y_pred    

        

        if model_type == 'lgb' and plot_feature_importance:

            # feature importance

            fold_importance = pd.DataFrame()

            fold_importance["feature"] = columns

            fold_importance["importance"] = model.feature_importances_

            fold_importance["fold"] = fold_n + 1

            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)



    #prediction /= folds.n_splits

    

    print('FINAL score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

    

    result_dict['oof'] = oof

    result_dict['prediction'] = prediction

    result_dict['scores'] = scores

    result_dict['model'] = model

    

    if model_type == 'lgb':

        if plot_feature_importance:

            feature_importance["importance"] /= folds.n_splits

            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(

                by="importance", ascending=False)[:50].index



            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]



            plt.figure(figsize=(16, 12));

            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));

            plt.title('LGB Features (avg over folds)');

            

            result_dict['feature_importance'] = feature_importance

        

    return result_dict
good_columns = [c for c in train.columns if c not in ['time', 'signal','open_channels', 'batch', 'batch_index', 'batch_slices', 'batch_slices2']]



X = train[good_columns].copy()

y = train['open_channels']

X_test = test[good_columns].copy()



# del train, test
%%time

params_xgb = {'colsample_bytree': 0.375, 'learning_rate': 0.05,  'max_depth':10, 'subsample': 1, 

              'colsample_bylevel':0.5, 'colsample_bynode':0.5, 'tree_method' : 'hist',

              'objective':'reg:squarederror', 'eval_metric':'logloss'}



result_dict_xgb = train_model_classification(X=X[0:500000*10-1], X_test=X_test, y=y[0:500000*10-1], params=params_xgb, model_type='xgb', eval_metric='logloss', plot_feature_importance=True,

                                                      verbose=50, early_stopping_rounds=200)  
%%time

params_lgb = {'learning_rate': 0.1, 'max_depth': 10, 'num_leaves': 400, 'boosting_type': 'dart', 'colsample_bytree': 0.7, 

              'max_bin': 300, 'metric': 'logloss', 'random_state': 7, 'n_jobs':-1}  



result_dict_lgb = train_model_classification(X=X[0:500000*10-1], X_test=X_test, y=y[0:500000*10-1], params=params_lgb, model_type='lgb', eval_metric='f1score', plot_feature_importance=False,

                                                      verbose=50, early_stopping_rounds=100, n_estimators=2000)

preds_ensemble = 0.50 * result_dict_lgb['prediction'] + 0.50 * result_dict_xgb['prediction']    
sub = pd.read_csv('/kaggle/input/liverpool-ion-switching/sample_submission.csv')

sub['open_channels'] =  np.array(np.round(preds_ensemble,0), np.int) 



sub.to_csv('submission.csv', index=False, float_format='%.4f')

sub.head(10)