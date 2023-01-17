import datetime

import numpy as np

import scipy as sp

import scipy.fftpack

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from scipy.signal import butter,filtfilt,freqz, iirnotch

from sklearn import *

from sklearn.metrics import f1_score

import lightgbm as lgb

import xgboost as xgb

from catboost import Pool,CatBoostRegressor

import time

import datetime

from sklearn.model_selection import KFold

from sklearn.metrics import f1_score, accuracy_score

from pykalman import KalmanFilter



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('../input/data-without-drift/train_clean.csv')

test = pd.read_csv('../input/data-without-drift/test_clean.csv')

train.head()
def create_axes_grid(numplots_x, numplots_y, plotsize_x=6, plotsize_y=3):

    fig, axes = plt.subplots(numplots_y, numplots_x)

    fig.set_size_inches(plotsize_x * numplots_x, plotsize_y * numplots_y)

    fig.subplots_adjust(wspace=0.05, hspace=0.05)

    return fig, axes



def set_axes(axes, use_grid=True, x_val = [0,100,10,5], y_val = [-50,50,10,5]):

    axes.grid(use_grid)

    axes.tick_params(which='both', direction='inout', top=True, right=True, labelbottom=True, labelleft=True)

    axes.set_xlim(x_val[0], x_val[1])

    axes.set_ylim(y_val[0], y_val[1])

    axes.set_xticks(np.linspace(x_val[0], x_val[1], np.around((x_val[1] - x_val[0]) / x_val[2] + 1).astype(int)))

    axes.set_xticks(np.linspace(x_val[0], x_val[1], np.around((x_val[1] - x_val[0]) / x_val[3] + 1).astype(int)), minor=True)

    axes.set_yticks(np.linspace(y_val[0], y_val[1], np.around((y_val[1] - y_val[0]) / y_val[2] + 1).astype(int)))

    axes.set_yticks(np.linspace(y_val[0], y_val[1], np.around((y_val[1] - y_val[0]) / y_val[3] + 1).astype(int)), minor=True)
def calc_markov_p_trans(states):

    max_state = np.max(states)

    states_next = np.roll(states, -1)

    matrix = []

    for i in range(max_state + 1):

        current_row = np.histogram(states_next[states == i], bins=np.arange(max_state + 2))[0]

        if np.sum(current_row) == 0: # if a state doesn't appear in states...

            current_row = np.ones(max_state + 1) / (max_state + 1) # ...use uniform probability

        else:

            current_row = current_row / np.sum(current_row) # normalize to 1

        matrix.append(current_row)

    return np.array(matrix)



def calc_markov_p_signal(state, signal, num_bins = 1000):

    states_range = np.arange(state.min(), state.max() + 1)

    signal_bins = np.linspace(signal.min(), signal.max(), num_bins + 1)

    p_signal = np.array([ np.histogram(signal[state == s], bins=signal_bins)[0] for s in states_range ])

    p_signal = np.array([ p / np.sum(p) for p in p_signal ]) # normalize to 1

    return p_signal, signal_bins



def digitize_signal(signal, signal_bins):

    signal_dig = np.digitize(signal, bins=signal_bins) - 1 # these -1 and -2 are necessary because of the way...

    signal_dig = np.minimum(signal_dig, len(signal_bins) - 2) # ... numpy.digitize works

    return signal_dig



def viterbi(p_trans, p_signal, p_in, signal):



    offset = 10**(-20) # added to values to avoid problems with log2(0)

    

    p_trans_tlog  = np.transpose(np.log2(p_trans  + offset)) # p_trans, logarithm + transposed

    p_signal_tlog = np.transpose(np.log2(p_signal + offset)) # p_signal, logarithm + transposed

    p_in_log      =              np.log2(p_in     + offset)  # p_in, logarithm

    p_state_log = [ p_in_log + p_signal_tlog[signal[0]] ] # initial state probabilities for signal element 0 



    for s in signal[1:]:

        p_state_log.append(np.max(p_state_log[-1] + p_trans_tlog, axis=1) + p_signal_tlog[s]) # the Viterbi algorithm



    states = np.argmax(p_state_log, axis=1) # finding the most probable states

    

    return states
def Kalman1D(observations,damping=1):

    # To return the smoothed time series data

    observation_covariance = damping

    initial_value_guess = observations[0]

    transition_matrix = 1

    transition_covariance = 0.1

    initial_value_guess

    kf = KalmanFilter(

            initial_state_mean=initial_value_guess,

            initial_state_covariance=observation_covariance,

            observation_covariance=observation_covariance,

            transition_covariance=transition_covariance,

            transition_matrices=transition_matrix

        )

    pred_state, state_cov = kf.smooth(observations)

    return pred_state
batch_size = 500000

num_batches = 10

res = 1000 # Resolution of signal plots



fs = 10000       # sample rate, 10kHz

nyq = 0.5 * fs  # Nyquist Frequency
plt.figure(figsize=(20,5));

plt.plot(range(0,train.shape[0],res),train.signal[0::res])

for i in range(num_batches+1): plt.plot([i*batch_size,i*batch_size],[-5,12.5],'r')

for j in range(num_batches): plt.text(j*batch_size+200000,num_batches,str(j+1),size=20)

plt.xlabel('Row',size=16); plt.ylabel('Signal',size=16); 

plt.title('Training Data Signal - 10 batches',size=20)

plt.show()
plt.figure(figsize=(20,5));

plt.plot(range(0,train.shape[0],res),train.open_channels[0::res])

for i in range(num_batches+1): plt.plot([i*batch_size,i*batch_size],[-5,12.5],'r')

for j in range(num_batches): plt.text(j*batch_size+200000,num_batches,str(j+1),size=20)

plt.xlabel('Row',size=16); plt.ylabel('Signal',size=16); 

plt.title('Training Data Open Channels - 10 batches',size=20)

plt.show()
batch = 5



fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(20, 10))

ax[0].plot(range(0,batch_size,res),train.open_channels[batch_size*(batch-1):batch_size*batch:res],color='g')

ax[1].plot(range(0,batch_size,res),train.signal[batch_size*(batch-1):batch_size*batch:res])



ax[0].legend(['open_channels'])

ax[1].legend(['signal'])
observation_covariance = .0015

signal_kalman_train = Kalman1D(train.signal.values,observation_covariance)

kalman = pd.DataFrame(signal_kalman_train,columns=['signal'])
f1_unfiltered = []

f1_unfiltered_avg = []

f1_filtered = []

f1_filtered_avg = []



for batch in range(1,10,1):

    # Unfiltered Viterbi F1 Macro

    p_trans = calc_markov_p_trans(train.open_channels[batch_size*(batch-1):batch_size*batch])

    p_signal_unfiltered, signal_bins = calc_markov_p_signal(train.open_channels[batch_size*(batch-1):batch_size*batch], train.signal[batch_size*(batch-1):batch_size*batch])

    signal_dig = digitize_signal(train.signal[batch_size*(batch-1):batch_size*batch], signal_bins)

    p_in = np.ones(len(p_trans)) / len(p_trans)

    viterbi_state = viterbi(p_trans, p_signal_unfiltered, p_in, signal_dig)

    print(f'Batch = {batch}')

    f1_unfiltered = f1_score(y_pred=viterbi_state, y_true=train.open_channels[batch_size*(batch-1):batch_size*batch], average='macro')

    print("Unfiltered - F1 macro =", f1_unfiltered)

    f1_unfiltered_avg += f1_unfiltered

    

    # Kalman Filtered Viterbi F1 Macro

    p_trans = calc_markov_p_trans(train.open_channels[batch_size*(batch-1):batch_size*batch])

    p_signal_filtered, signal_bins = calc_markov_p_signal(train.open_channels[batch_size*(batch-1):batch_size*batch], kalman.signal[batch_size*(batch-1):batch_size*batch])

    signal_dig = digitize_signal(kalman.signal[batch_size*(batch-1):batch_size*batch], signal_bins)

    p_in = np.ones(len(p_trans)) / len(p_trans)

    viterbi_state = viterbi(p_trans, p_signal_filtered, p_in, signal_dig)

    f1_filtered = f1_score(y_pred=viterbi_state, y_true=train.open_channels[batch_size*(batch-1):batch_size*batch], average='macro')

    print("Kalman Filtered - F1 macro =", f1_filtered)

    f1_filtered_avg += f1_filtered
fig, axes = create_axes_grid(1,1,30,7)

set_axes(axes, x_val=[0, 1000, 100, 10], y_val=[0,0.02,0.005,0.001])

axes.set_title('Signal probability distribution for each state (not normalized)')

for s,p in enumerate(p_signal_unfiltered):

    axes.plot(p, label="Unfiltered - State "+str(s));

for s,p in enumerate(p_signal_filtered):

    axes.plot(p, label="Filtered - State "+str(s));    

axes.legend();
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(20, 10))

ax[0].plot(range(0,batch_size,res),train.open_channels[batch_size*(batch-1):batch_size*batch:res],color='g')

ax[1].plot(range(0,batch_size,res),train.signal[batch_size*(batch-1):batch_size*batch:res])

ax[1].plot(range(0,batch_size,res),kalman.signal[batch_size*(batch-1):batch_size*batch:res])



ax[0].legend(['open_channels'])

ax[1].legend(['signal', 'filtered signal'])
fft = sp.fftpack.fft(train.signal[batch_size*(batch-1):batch_size*batch])

psd = np.abs(fft) ** 2

fftfreq = sp.fftpack.fftfreq(len(psd),1/fs)

i = fftfreq > 0



fig, ax = plt.subplots(2, 1, figsize=(10, 6))

fig.subplots_adjust(hspace = .5)

ax[0].plot(fftfreq[i], 10 * np.log10(psd[i]))

ax[0].set_xlabel('Frequency (1/10000 seconds)')

ax[0].set_ylabel('PSD (dB)')

ax[0].set_title('Unfiltered')





fft = sp.fftpack.fft(kalman.signal)

psd = np.abs(fft) ** 2

fftfreq = sp.fftpack.fftfreq(len(psd),1/fs)

i = fftfreq > 0



ax[1].plot(fftfreq[i], 10 * np.log10(psd[i]))

ax[1].set_xlabel('Frequency (1/10000 seconds)')

ax[1].set_ylabel('PSD (dB)')

ax[1].set_title('Kalman Filter')
fig, axes = create_axes_grid(1,1,10,5)

axes.set_title('Markov Transition Matrix P_trans')

sns.heatmap(

    p_trans,

    annot=True, fmt='.3f', cmap='Blues', cbar=False,

    ax=axes, vmin=0, vmax=0.5, linewidths=2);
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(20, 10))

ax[0].plot(range(0,batch_size,res),train.open_channels[batch_size*(batch-1):batch_size*batch:res],color='g')

ax[1].plot(range(0,batch_size,res),viterbi_state[::res])



ax[0].legend(['open_channels'])

ax[1].legend(['Viterbi State Prediction'])
# Use the filtered signal for train 'signal'

train['signal_undrifted'] = signal_kalman_train



# Apply Kalman filter to test data and set as test 'signal'

signal_kalman_test = Kalman1D(test.signal.values,observation_covariance)

test['signal_undrifted'] = signal_kalman_test
def features(df):

    df = df.sort_values(by=['time']).reset_index(drop=True)

    df.index = ((df.time * 10_000) - 1).values

    df['batch'] = df.index // 50_000

    df['batch_index'] = df.index  - (df.batch * 50_000)

    df['batch_slices'] = df['batch_index']  // 5_000

    df['batch_slices2'] = df.apply(lambda r: '_'.join([str(r['batch']).zfill(3), str(r['batch_slices']).zfill(3)]), axis=1)

    

    for c in ['batch','batch_slices2']:

        d = {}

        d['mean'+c] = df.groupby([c])['signal_undrifted'].mean()

        d['median'+c] = df.groupby([c])['signal_undrifted'].median()

        d['max'+c] = df.groupby([c])['signal_undrifted'].max()

        d['min'+c] = df.groupby([c])['signal_undrifted'].min()

        d['std'+c] = df.groupby([c])['signal_undrifted'].std()

        d['mean_abs_chg'+c] = df.groupby([c])['signal_undrifted'].apply(lambda x: np.mean(np.abs(np.diff(x))))

        d['abs_max'+c] = df.groupby([c])['signal_undrifted'].apply(lambda x: np.max(np.abs(x)))

        d['abs_min'+c] = df.groupby([c])['signal_undrifted'].apply(lambda x: np.min(np.abs(x)))

        for v in d:

            df[v] = df[c].map(d[v].to_dict())

        df['range'+c] = df['max'+c] - df['min'+c]

        df['maxtomin'+c] = df['max'+c] / df['min'+c]

        df['abs_avg'+c] = (df['abs_min'+c] + df['abs_max'+c]) / 2

    

    #add shifts

    df['signal_shift_+1'] = [0,] + list(df['signal_undrifted'].values[:-1])

    df['signal_shift_-1'] = list(df['signal_undrifted'].values[1:]) + [0]

    for i in df[df['batch_index']==0].index:

        df['signal_shift_+1'][i] = np.nan

    for i in df[df['batch_index']==49999].index:

        df['signal_shift_-1'][i] = np.nan



    # add shifts_2

    df['signal_shift_+2'] = [0,] + [1,] + list(df['signal_undrifted'].values[:-2])

    df['signal_shift_-2'] = list(df['signal_undrifted'].values[2:]) + [0] + [1]

    for i in df[df['batch_index']==0].index:

        df['signal_shift_+2'][i] = np.nan

    for i in df[df['batch_index']==1].index:

        df['signal_shift_+2'][i] = np.nan

    for i in df[df['batch_index']==49999].index:

        df['signal_shift_-2'][i] = np.nan

    for i in df[df['batch_index']==49998].index:

        df['signal_shift_-2'][i] = np.nan 

        

    for c in [c1 for c1 in df.columns if c1 not in ['time', 'signal_undrifted', 'open_channels', 'batch', 'batch_index', 'batch_slices', 'batch_slices2']]:

        df[c+'_msignal'] = df[c] - df['signal_undrifted']

        

    return df



train = features(train)

test = features(test)
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



del train, test
params_xgb = {'colsample_bytree': 0.375,'learning_rate': 0.1,'max_depth': 10, 'subsample': 1, 'objective':'reg:squarederror',

          'eval_metric':'rmse'}



result_dict_xgb = train_model_classification(X=X[0:500000*8-1], X_test=X_test, y=y[0:500000*8-1], params=params_xgb, model_type='xgb', eval_metric='f1score', plot_feature_importance=False,

                                                      verbose=50, early_stopping_rounds=250)
params_lgb = {'learning_rate': 0.1, 'max_depth': 7, 'num_leaves':2**7+1, 'metric': 'rmse', 'random_state': 7, 'n_jobs':-1}



result_dict_lgb = train_model_classification(X=X[0:500000*8-1], X_test=X_test, y=y[0:500000*8-1], params=params_lgb, model_type='lgb', eval_metric='f1score', plot_feature_importance=False,

                                                      verbose=50, early_stopping_rounds=250, n_estimators=3000)
preds_ensemble = 0.50 * result_dict_lgb['prediction'] + 0.50 * result_dict_xgb['prediction']
sub = pd.read_csv('/kaggle/input/liverpool-ion-switching/sample_submission.csv')

sub['open_channels'] =  np.array(np.round(preds_ensemble,0), np.int) 



sub.to_csv('submission.csv', index=False, float_format='%.4f')

sub.head(10)