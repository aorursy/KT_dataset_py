import numpy as np
import pandas as pd 
from tqdm import tqdm_notebook

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
from plotly import tools
import plotly.figure_factory as ff

from scipy.signal import hilbert, hann, convolve

import matplotlib.pyplot as plt
%matplotlib inline
from tqdm import tqdm_notebook
from sklearn.preprocessing import StandardScaler
pd.options.display.precision = 15

import lightgbm as lgb
import time
import datetime

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import gc
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import os
print(os.listdir("../input"))
%%time
train = pd.read_csv('../input/train.csv', dtype={'acoustic_data': np.float32, 'time_to_failure': np.float32})
train.head()
def classic_sta_lta_py(a, nsta, nlta):
    """
    Computes the standard STA/LTA from a given input array a. The length of
    the STA is given by nsta in samples, respectively is the length of the
    LTA given by nlta in samples. Written in Python.
    .. note::
        There exists a faster version of this trigger wrapped in C
        called :func:`~obspy.signal.trigger.classic_sta_lta` in this module!
    :type a: NumPy :class:`~numpy.ndarray`
    :param a: Seismic Trace
    :type nsta: int
    :param nsta: Length of short time average window in samples
    :type nlta: int
    :param nlta: Length of long time average window in samples
    :rtype: NumPy :class:`~numpy.ndarray`
    :return: Characteristic function of classic STA/LTA
    """
    # The cumulative sum can be exploited to calculate a moving average (the
    # cumsum function is quite efficient)
    sta = np.cumsum(a ** 2)

    # Convert to float
    sta = np.require(sta, dtype=np.float)

    # Copy for LTA
    lta = sta.copy()

    # Compute the STA and the LTA
    sta[nsta:] = sta[nsta:] - sta[:-nsta]
    sta /= nsta
    lta[nlta:] = lta[nlta:] - lta[:-nlta]
    lta /= nlta

    # Pad zeros
    sta[:nlta - 1] = 0

    # Avoid division by zero by setting zero values to tiny float
    dtiny = np.finfo(0.0).tiny
    idx = lta < dtiny
    lta[idx] = dtiny

    return sta / lta
per = 0.000005
#Calculate Hilbert transform
signal = train.acoustic_data[:int(len(train)*per)]
analytic_signal = hilbert(signal)
amplitude_envelope = np.abs(analytic_signal)

#Calculate Hann func
win = hann(50)
filtered = convolve(signal, win, mode='same') / sum(win)

#Calculate STA/LTA
sta_lta = classic_sta_lta_py(signal, 50, 1000)
trace0 = go.Scatter(
    y = signal,
    name = 'signal'
)

trace1 = go.Scatter(
    y = amplitude_envelope,
    name = 'amplitude_envelope'
)


trace3 = go.Scatter(
    y = filtered,
    name= 'filtered'
) 

trace4 = go.Scatter(
    y = sta_lta,
    name= 'sta_lta'
) 


trace_time = go.Scatter(
    y = train.time_to_failure[:int(len(train)*per)],
)

data = [trace0, trace1, trace3,trace4]

layout = go.Layout(
    title = "Part acoustic_data"
)

fig = go.Figure(data=data,layout=layout)
py.iplot(fig, filename = "Part acoustic_data")
col_feat = ['ave','med', 'std', 'max', 'min',
                               'av_change_abs', 'av_change_rate', 'abs_max', 'abs_min',
                               'std_first_50000', 'std_last_50000', 'std_first_10000', 'std_last_10000',
                               'avg_first_50000', 'avg_last_50000', 'avg_first_10000', 'avg_last_10000',
                               'min_first_50000', 'min_last_50000', 'min_first_10000', 'min_last_10000',
                               'max_first_50000', 'max_last_50000', 'max_first_10000', 'max_last_10000',
                               ]
name_lines = ['','amp_env','filt','sta_lta']
columns = []
for cf in col_feat:
    for name in name_lines:
        columns.append(cf+'_'+name)
        
len(columns)
rows = 150_000
segments = int(np.floor(train.shape[0] / rows))

y_tr = pd.DataFrame(index=range(segments), dtype=np.float64,
                       columns=['time_to_failure'])

X_tr = pd.DataFrame()

def get_feature(x,segment, name_lines):
    X_tr1 = pd.DataFrame(dtype=np.float64)
    X_tr1.loc[segment, 'ave'+'_'+name_lines] = x.mean()
    X_tr1.loc[segment, 'ave'+'_'+name_lines] = np.median(x)
    X_tr1.loc[segment, 'std'+'_'+name_lines] = x.std()
    X_tr1.loc[segment, 'max'+'_'+name_lines] = x.max()
    X_tr1.loc[segment, 'min'+'_'+name_lines] = x.min()
    
    X_tr1.loc[segment, 'av_change_abs'+'_'+name_lines] = np.mean(np.diff(x))
    X_tr1.loc[segment, 'av_change_rate'+'_'+name_lines] = np.mean(np.nonzero((np.diff(x) / x[:-1]))[0])
    X_tr1.loc[segment, 'abs_max'+'_'+name_lines] = np.abs(x).max()
    X_tr1.loc[segment, 'abs_min'+'_'+name_lines] = np.abs(x).min()
    
    X_tr1.loc[segment, 'std_first_50000'+'_'+name_lines] = x[:50000].std()
    X_tr1.loc[segment, 'std_last_50000'+'_'+name_lines] = x[50000:].std()
    X_tr1.loc[segment, 'std_first_10000'+'_'+name_lines] = x[:10000].std()
    X_tr1.loc[segment, 'std_last_10000'+'_'+name_lines] = x[10000:].std()
    
    X_tr1.loc[segment, 'avg_first_50000'+'_'+name_lines] = x[:50000].mean()
    X_tr1.loc[segment, 'avg_last_50000'+'_'+name_lines] = x[50000:].mean()
    X_tr1.loc[segment, 'avg_first_10000'+'_'+name_lines] = x[:10000].mean()
    X_tr1.loc[segment, 'avg_last_10000'+'_'+name_lines] = x[10000:].mean()
    
    X_tr1.loc[segment, 'min_first_50000'+'_'+name_lines] = x[:50000].min()
    X_tr1.loc[segment, 'min_last_50000'+'_'+name_lines] = x[50000:].min()
    X_tr1.loc[segment, 'min_first_10000'+'_'+name_lines] = x[:10000].min()
    X_tr1.loc[segment, 'min_last_10000'+'_'+name_lines] = x[10000:].min()
    
    X_tr1.loc[segment, 'max_first_50000'+'_'+name_lines] = x[:50000].max()
    X_tr1.loc[segment, 'max_last_50000'+'_'+name_lines] = x[50000:].max()
    X_tr1.loc[segment, 'max_first_10000'+'_'+name_lines] = x[:10000].max()
    X_tr1.loc[segment, 'max_last_10000'+'_'+name_lines] = x[10000:].max()
    return X_tr1
np.random.seed(42)
for segment in tqdm_notebook(range(segments)):
    seg = train.iloc[segment*rows:segment*rows+rows]
    x = seg['acoustic_data'].values
    y = seg['time_to_failure'].values[-1]
    y_tr.loc[segment, 'time_to_failure'] = y
    
    x_ac_signal = get_feature(x,segment, 'data')
    
    analytic_signal = hilbert(x)
    amplitude_envelope = np.abs(analytic_signal)
    x_amp_env = get_feature(amplitude_envelope,segment, 'amp_env')
    
    win = hann(100)
    filtered = convolve(x, win, mode='same') / sum(win)
    x_filt_hann = get_feature(filtered,segment, 'filt')
    
    sta_lta = classic_sta_lta_py(x, 1000, 5000)
    x_sta_lta = get_feature(sta_lta,segment, 'sta_lta')
    
    df_loc = pd.concat([x_ac_signal, x_amp_env, x_filt_hann, x_sta_lta], axis = 1)
    X_tr = X_tr.append(df_loc)
segments = 10000

y_tr_more = pd.DataFrame(index=range(segments), dtype=np.float64,
                       columns=['time_to_failure'])
X_tr_more = pd.DataFrame()
np.random.seed(42)
for segment in tqdm_notebook(range(segments)):
    ind = np.random.randint(0, train.shape[0]-150001)
    seg = train.iloc[ind:ind+rows]
    x = seg['acoustic_data'].values
    y = seg['time_to_failure'].values[-1]
    
    y_tr_more.loc[segment, 'time_to_failure'] = y
    x_ac_signal = get_feature(x,segment, 'data')
    
    analytic_signal = hilbert(x)
    amplitude_envelope = np.abs(analytic_signal)
    x_amp_env = get_feature(amplitude_envelope,segment, 'amp_env')
    
    win = hann(100)
    filtered = convolve(x, win, mode='same') / sum(win)
    x_filt_hann = get_feature(filtered,segment, 'filt')
    
    sta_lta = classic_sta_lta_py(x, 1000, 5000)
    x_sta_lta = get_feature(sta_lta,segment, 'sta_lta')
    
    df_loc = pd.concat([x_ac_signal, x_amp_env, x_filt_hann, x_sta_lta], axis = 1)
    X_tr_more = X_tr_more.append(df_loc)
X_tr = X_tr.append(X_tr_more)
y_tr = y_tr.append(y_tr_more)
print(f'{X_tr.shape[0]} samples in new train data now.')
X_tr.tail()
scaler = StandardScaler()
scaler.fit(X_tr)
X_train_scaled = pd.DataFrame(scaler.transform(X_tr), columns=X_tr.columns)
submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')
X_test = pd.DataFrame()
for i, seg_id in enumerate(tqdm_notebook(submission.index)):
    seg = pd.read_csv('../input/test/' + seg_id + '.csv')
    
    x = seg['acoustic_data'].values
    
    x_ac_signal = get_feature(x,seg_id, 'data')
    
    analytic_signal = hilbert(x)
    amplitude_envelope = np.abs(analytic_signal)
    x_amp_env = get_feature(amplitude_envelope,seg_id, 'amp_env')
    
    win = hann(100)
    filtered = convolve(x, win, mode='same') / sum(win)
    x_filt_hann = get_feature(filtered,seg_id, 'filt')
    
    sta_lta = classic_sta_lta_py(x, 1000, 5000)
    x_sta_lta = get_feature(sta_lta,seg_id, 'sta_lta')
    
    df_loc = pd.concat([x_ac_signal, x_amp_env, x_filt_hann, x_sta_lta], axis = 1)
    X_test = X_test.append(df_loc)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
len(X_test_scaled)
len(X_train_scaled)
X_train_scaled = X_train_scaled.fillna(0)
X_test_scaled = X_test_scaled.fillna(0)
n_fold = 5
folds = KFold(n_splits=n_fold, shuffle=True, random_state=11)
def train_model(X=X_train_scaled, X_test=X_test_scaled, y=y_tr, params=None, folds=folds, model_type='lgb', plot_feature_importance=False, model=None):

    oof = np.zeros(len(X))
    prediction = np.zeros(len(X_test))
    scores = []
    feature_importance = pd.DataFrame()
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
        print('Fold', fold_n, 'started at', time.ctime())
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        
        if model_type == 'lgb':
            model = lgb.LGBMRegressor(**params, n_estimators = 20000, nthread = 4, n_jobs = -1)
            model.fit(X_train, y_train, 
                    eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='mae',
                    verbose=10000, early_stopping_rounds=200)
            
            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test, num_iteration=model.best_iteration_)
        
        oof[valid_index] = y_pred_valid.reshape(-1,)
        scores.append(mean_absolute_error(y_valid, y_pred_valid))

        prediction += y_pred
        
        if model_type == 'lgb':
            # feature importance
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = X.columns
            fold_importance["importance"] = model.feature_importances_
            fold_importance["fold"] = fold_n + 1
            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

    prediction /= n_fold
    
    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))
    
    if model_type == 'lgb':
        feature_importance["importance"] /= n_fold
        if plot_feature_importance:
            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
                by="importance", ascending=False)[:50].index

            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

            plt.figure(figsize=(16, 12));
            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));
            plt.title('LGB Features (avg over folds)');
        
            return oof, prediction, feature_importance
        return oof, prediction
    
    else:
        return oof, prediction
params = {'num_leaves': 54,
         'min_data_in_leaf': 79,
         'objective': 'huber',
         'max_depth': -1,
         'learning_rate': 0.01,
         "boosting": "gbdt",
         # "feature_fraction": 0.8354507676881442,
         "bagging_freq": 3,
         "bagging_fraction": 0.8126672064208567,
         "bagging_seed": 11,
         "metric": 'mae',
         "verbosity": -1,
         'reg_alpha': 1.1302650970728192,
         'reg_lambda': 0.3603427518866501
         }
oof_lgb, prediction_lgb, feature_importance = train_model(params=params, model_type='lgb', plot_feature_importance=True)
len(prediction_lgb)
submission['time_to_failure'] = prediction_lgb
print(submission.head())
submission.to_csv('submission.csv')