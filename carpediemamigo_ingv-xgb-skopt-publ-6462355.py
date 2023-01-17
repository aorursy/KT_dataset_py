import os

import gc

import numpy as np

import pandas as pd



from time import time

from time import ctime



from sklearn.linear_model import LinearRegression

import scipy.stats as spstats

from scipy.signal import hilbert

from scipy.stats import iqr, gmean, sem, entropy



import warnings

warnings.filterwarnings("ignore")

from tqdm import tqdm_notebook

from tqdm import tqdm



import joblib

from joblib import Parallel, delayed

import multiprocessing

num_cores = multiprocessing.cpu_count()-1



from sklearn.metrics import mean_absolute_error

from sklearn.metrics import r2_score



RANDOM_STATE = 12061985



import matplotlib.pyplot as plt



import xgboost

import lightgbm as lgbm

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import mean_absolute_error

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import KFold,StratifiedKFold, RepeatedKFold



from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_validate

from sklearn.model_selection import learning_curve

from sklearn.model_selection import validation_curve

from sklearn.model_selection import ShuffleSplit

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.model_selection import KFold



# HPO

from skopt.space import Integer, Categorical, Real

from skopt.utils import use_named_args

from skopt import gp_minimize, gbrt_minimize, forest_minimize

from skopt.plots import plot_convergence

from skopt.callbacks import DeltaXStopper, DeadlineStopper, DeltaYStopper

from skopt.callbacks import EarlyStopper
def ChangeRate(x):

    change = (np.diff(x) / x[:-1]).values

    change = change[np.nonzero(change)[0]]

    change = change[~np.isnan(change)]

    change = change[change != -np.inf]

    change = change[change != np.inf]

    return np.mean(change)



def basic_statistics(t_X, x, s, sensor, postfix=''):

    """Computes basic statistics for the training feature set.

    

    Args:

        t_X (pandas.DataFrame): The feature set being built.

        x (pandas.Series): The signal values.

        s (int): The integer number of the segment.

        postfix (str): The postfix string value.

    Return:

        t_X (pandas.DataFrame): The feature set being built.

    """

    t_X.loc[s, f'{sensor}_sum{postfix}']       = x.sum()

    t_X.loc[s, f'{sensor}_mean{postfix}']      = x.mean()

    t_X.loc[s, f'{sensor}_std{postfix}']       = x.std()

    t_X.loc[s, f'{sensor}_var{postfix}']       = x.var() 

    t_X.loc[s, f'{sensor}_max{postfix}']       = x.max()

    t_X.loc[s, f'{sensor}_min{postfix}']       = x.min()

    t_X.loc[s, f'{sensor}_median{postfix}']    = x.median()

    t_X.loc[s, f'{sensor}_skew{postfix}']      = x.skew()

    t_X.loc[s, f'{sensor}_mad{postfix}']       = x.mad()

    t_X.loc[s, f'{sensor}_kurtosis{postfix}']  = x.kurtosis()

    

    t_X.loc[s, f'{sensor}_ptp{postfix}']  = x.values.ptp()

    t_X.loc[s, f'{sensor}_prod{postfix}'] = x.prod()

    t_X.loc[s, f'{sensor}_iqr{postfix}']  = iqr(x)

    t_X.loc[s, f'{sensor}_sem{postfix}']  = x.sem()

    t_X.loc[s, f'{sensor}_gmean{postfix}']  = gmean(x)

    t_X.loc[s, f'{sensor}_entropy{postfix}']  = entropy(x)

    t_X.loc[s, f'{sensor}_chrate{postfix}']  = ChangeRate(x)



    t_X.loc[s, f'{sensor}_hilmean{postfix}']  = np.abs(hilbert(x)).mean()

    t_X.loc[s, f'{sensor}_countbig{postfix}']  = len(x[np.abs(x) > x.mean()])

    t_X.loc[s, f'{sensor}_maxmindiff{postfix}']  = x.max() - np.abs(x.min())

    t_X.loc[s, f'{sensor}_maxtomin{postfix}']  = x.max() / np.abs(x.min())

    t_X.loc[s, f'{sensor}_meanchabs{postfix}']  = np.mean(np.diff(x))   



    return t_X





def quantiles(t_X, x, s, sensor, postfix=''):

    """Calculates quantile features for the training feature set.

    Args:

        t_X (pandas.DataFrame): The feature set being built.

        x (pandas.Series): The signal values.

        s (int): The integer number of the segment.

        postfix (str): The postfix string value.

    Return:

        t_X (pandas.DataFrame): The feature set being built.

    """

    t_X.loc[s, f'{sensor}_q999{postfix}']     = np.quantile(x ,0.999)

    t_X.loc[s, f'{sensor}_q99{postfix}']      = np.quantile(x, 0.99)

    t_X.loc[s, f'{sensor}_q95{postfix}']      = np.quantile(x, 0.95)

    t_X.loc[s, f'{sensor}_q87{postfix}']      = np.quantile(x, 0.87)

    t_X.loc[s, f'{sensor}_q13{postfix}']      = np.quantile(x, 0.13)  

    t_X.loc[s, f'{sensor}_q05{postfix}']      = np.quantile(x, 0.05)

    t_X.loc[s, f'{sensor}_q01{postfix}']      = np.quantile(x, 0.01)

    t_X.loc[s, f'{sensor}_q001{postfix}']     = np.quantile(x ,0.001)

    

    x_abs = np.abs(x)

    t_X.loc[s, f'{sensor}_q999_abs{postfix}'] = np.quantile(x_abs, 0.999)

    t_X.loc[s, f'{sensor}_q99_abs{postfix}']  = np.quantile(x_abs, 0.99)

    t_X.loc[s, f'{sensor}_q95_abs{postfix}']  = np.quantile(x_abs, 0.95)

    t_X.loc[s, f'{sensor}_q87_abs{postfix}']  = np.quantile(x_abs, 0.87)

    t_X.loc[s, f'{sensor}_q13_abs{postfix}']  = np.quantile(x_abs, 0.13)

    t_X.loc[s, f'{sensor}_q05_abs{postfix}']  = np.quantile(x_abs, 0.05)

    t_X.loc[s, f'{sensor}_q01_abs{postfix}']  = np.quantile(x_abs, 0.01)

    t_X.loc[s, f'{sensor}_q001_abs{postfix}'] = np.quantile(x_abs, 0.001)

    

    t_X.loc[s, f'{sensor}_iqr']     = np.subtract(*np.percentile(x, [75, 25]))

    t_X.loc[s, f'{sensor}_iqr_abs'] = np.subtract(*np.percentile(x_abs, [75, 25]))



    return t_X



def __linear_regression(arr, abs_v=False):

    """

    """

    idx = np.array(range(len(arr)))

    if abs_v:

        arr = np.abs(arr)

    lr = LinearRegression()

    fit_X = idx.reshape(-1, 1)

    lr.fit(fit_X, arr)

    return lr.coef_[0]





def __classic_sta_lta(x, length_sta, length_lta):

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



def linear_regression(t_X, x, s, sensor, postfix=''):

    t_X.loc[s, f'{sensor}_lr_coef{postfix}'] = __linear_regression(x)

    t_X.loc[s, f'{sensor}_lr_coef_abs{postfix}'] = __linear_regression(x, True)

    return t_X





def classic_sta_lta(t_X, x, sensor, s):

    t_X.loc[s, f'{sensor}_classic_sta_lta1_mean'] = __classic_sta_lta(x, 500, 10000).mean()

    t_X.loc[s, f'{sensor}_classic_sta_lta2_mean'] = __classic_sta_lta(x, 5000, 100000).mean()

    t_X.loc[s, f'{sensor}_classic_sta_lta3_mean'] = __classic_sta_lta(x, 3333, 6666).mean()

    t_X.loc[s, f'{sensor}_classic_sta_lta4_mean'] = __classic_sta_lta(x, 10000, 25000).mean()

    return t_X





def fft(t_X, x, s, sensor, postfix=''):

    """Generates basic statistics over the fft of the signal"""

    z = np.fft.fft(x)

    fft_real = np.real(z)

    fft_imag = np.imag(z)



    t_X.loc[s, f'fft_A0']             = abs(z[0])

    t_X.loc[s, f'{sensor}_fft_real_mean{postfix}']      = fft_real.mean()

    t_X.loc[s, f'{sensor}_fft_real_std{postfix}']       = fft_real.std()

    t_X.loc[s, f'{sensor}_fft_real_max{postfix}']       = fft_real.max()

    t_X.loc[s, f'{sensor}_fft_real_min{postfix}']       = fft_real.min()

    t_X.loc[s, f'{sensor}_fft_real_median{postfix}']    = np.median(fft_real)

    t_X.loc[s, f'{sensor}_fft_real_skew{postfix}']      = spstats.skew(fft_real)

    t_X.loc[s, f'{sensor}_fft_real_kurtosis{postfix}']  = spstats.kurtosis(fft_real)

    

    t_X.loc[s, f'{sensor}_fft_imag_mean{postfix}']      = fft_imag.mean()

    t_X.loc[s, f'{sensor}_fft_imag_std{postfix}']       = fft_imag.std()

    t_X.loc[s, f'{sensor}_fft_imag_max{postfix}']       = fft_imag.max()

    t_X.loc[s, f'{sensor}_fft_imag_min{postfix}']       = fft_imag.min()

    t_X.loc[s, f'{sensor}_fft_imag_median{postfix}']    = np.median(fft_imag)

    t_X.loc[s, f'{sensor}_fft_imag_skew{postfix}']      = spstats.skew(fft_imag)

    t_X.loc[s, f'{sensor}_fft_imag_kurtosis{postfix}']  = spstats.kurtosis(fft_imag)

    

    return t_X
def get_params_SKopt(model, X, Y, space, cv_search, alg = 'catboost', cat_features = None, eval_dataset = None, UBM = False, opt_method =

                     'gbrt_minimize', verbose = True,  multi = False, scoring = 'neg_mean_squared_error', n_best = 50, total_time = 7200):

    """The method performs parameters tuning of an algorithm using scikit-optimize library.

    Parameters:

    1.

    2.

    3. multi - boolean, is used when a multioutput algorithm is tuned

    UPDATES:

    1. In this current version, the support of the catboost algorithms is added

    """

    if alg == 'catboost':

        fitparam = { 'eval_set' : eval_dataset,

                     'use_best_model' : UBM,

                     'cat_features' : cat_features,

                     'early_stopping_rounds': 10 }

    else:

        fitparam = {}

        

    @use_named_args(space)

    def objective(**params):

        model.set_params(**params)

        return -np.mean(cross_val_score(model, 

                                        X, Y, 

                                        cv=cv_search, 

                                        scoring= scoring,

                                        fit_params=fitparam))

    

    if opt_method == 'gbrt_minimize':

        

        HPO_PARAMS = {'n_calls':1000,

                      'n_random_starts':20,

                      'acq_func':'EI',}

        

        reg_gp = gbrt_minimize(objective, 

                               space, 

                               n_jobs = -1,

                               verbose = verbose,

                               callback = [DeltaYStopper(delta = 0.01, n_best = 5), RepeatedMinStopper(n_best = n_best), DeadlineStopper(total_time = total_time)],

                               **HPO_PARAMS,

                               random_state = RANDOM_STATE)

        



    elif opt_method == 'forest_minimize':

        

        HPO_PARAMS = {'n_calls':1000,

                      'n_random_starts':20,

                      'acq_func':'EI',}

        

        reg_gp = forest_minimize(objective, 

                               space, 

                               n_jobs = -1,

                               verbose = verbose,

                               callback = [RepeatedMinStopper(n_best = n_best), DeadlineStopper(total_time = total_time)],

                               **HPO_PARAMS,

                               random_state = RANDOM_STATE)

        

    elif opt_method == 'gp_minimize':

        

        HPO_PARAMS = {'n_calls':1000,

                      'n_random_starts':20,

                      'acq_func':'gp_hedge',}        

        

        reg_gp = gp_minimize(objective, 

                               space, 

                               n_jobs = -1,

                               verbose = verbose,

                               callback = [RepeatedMinStopper(n_best = n_best), DeadlineStopper(total_time = total_time)],

                               **HPO_PARAMS,

                               random_state = RANDOM_STATE)

    

    TUNED_PARAMS = {} 

    for i, item in enumerate(space):

        if multi:

            TUNED_PARAMS[item.name.split('__')[1]] = reg_gp.x[i]

        else:

            TUNED_PARAMS[item.name] = reg_gp.x[i]

    

    return [TUNED_PARAMS,reg_gp]



class RepeatedMinStopper(EarlyStopper):

    """Stop the optimization when there is no improvement in the minimum.

    Stop the optimization when there is no improvement in the minimum

    achieved function evaluation after `n_best` iterations.

    """

    def __init__(self, n_best=50):

        super(EarlyStopper, self).__init__()

        self.n_best = n_best

        self.count = 0

        self.minimum = np.finfo(np.float).max



    def _criterion(self, result):

        if result.fun < self.minimum:

            self.minimum = result.fun

            self.count = 0

        elif result.fun > self.minimum:

            self.count = 0

        else:

            self.count += 1



        return self.count >= self.n_best



def plotfig (ypred, yactual, strtitle, y_max):

    plt.scatter(ypred, yactual.values.ravel())

    plt.title(strtitle)

    plt.plot([(0, 0), (y_max, y_max)], [(0, 0), (y_max, y_max)])

    plt.xlim(0, y_max)

    plt.ylim(0, y_max)

    plt.xlabel('Predicted', fontsize=12)

    plt.ylabel('Actual', fontsize=12)

    plt.show()
n_files = None

gc.collect()
def features_generator(path_to_file):

    signals = pd.read_csv(path_to_file)

    seg = int(path_to_file.split('/')[-1].split('.')[0])

    row = pd.DataFrame([])

    

    signals = signals.rolling(10).mean().iloc[list(np.arange(10,60001,10))]

    

    for i in range(1, 11):

        sensor_id = f'sensor_{i}'



#         row = basic_statistics(row, signals[sensor_id].interpolate(method='polynomial', order=2), seg, sensor_id, postfix='')

#         row = basic_statistics(row, signals[sensor_id].interpolate(method='polynomial', order=2), seg, sensor_id, postfix='')

#         row = basic_statistics(row, signals[sensor_id].interpolate(method='polynomial', order=2), seg, sensor_id, postfix='')

#         row = basic_statistics(row, signals[sensor_id].interpolate(method='polynomial', order=2), seg, sensor_id, postfix='')

        

        

        row = basic_statistics(row, signals[sensor_id].fillna(0), seg, sensor_id, postfix='')

        row = basic_statistics(row, signals[sensor_id].fillna(0), seg, sensor_id, postfix='')

        row = basic_statistics(row, signals[sensor_id].fillna(0), seg, sensor_id, postfix='')

        row = basic_statistics(row, signals[sensor_id].fillna(0), seg, sensor_id, postfix='')

    return row
train_path_to_signals = '/kaggle/input/predict-volcanic-eruptions-ingv-oe/train/'

train_files_list = [os.path.join(train_path_to_signals, file) for file in os.listdir(train_path_to_signals)]

rows = Parallel(n_jobs=-1)(delayed(features_generator)(ex) for ex in tqdm(train_files_list[:n_files]))  

train_set = pd.concat(rows, axis=0)

train_set.head()
train = pd.read_csv('/kaggle/input/predict-volcanic-eruptions-ingv-oe/train.csv')

train = train.set_index('segment_id')

train_set = pd.concat([train_set, train], axis = 1)

train_set.head()
test_path_to_signals = '/kaggle/input/predict-volcanic-eruptions-ingv-oe/test/'

test_files_list = [os.path.join(test_path_to_signals, file) for file in os.listdir(test_path_to_signals)]

rows = Parallel(n_jobs=-1)(delayed(features_generator)(ex) for ex in tqdm(test_files_list[:n_files]))  

test_set = pd.concat(rows, axis=0)

test_set.head()
test = pd.read_csv('/kaggle/input/predict-volcanic-eruptions-ingv-oe/sample_submission.csv')

test = test.set_index('segment_id')

test_set = pd.concat([test_set, test], axis = 1)

test_set.head()
%%time



Y = train_set['time_to_eruption']

X = train_set.drop(['time_to_eruption'], axis = 1)



X_test = test_set.drop(['time_to_eruption'], axis = 1)



STATIC_PARAMS = {

                    'n_estimators': 250,

                    'objective' : 'reg:squarederror',

                    'random_state' : RANDOM_STATE,

                    'n_jobs': -1, 

                }



space_SKopt = [

                 Integer(2, 50, name='max_depth'),

                 Integer(2, 500, name='min_child_weight'),

                 Real(0.005, .05, name='learning_rate'),

                 Real(0.1, 1, name='subsample'),

                 Real(0.1, 1, name='colsample_bytree'),

                 Real(0.1, 10, name='reg_alpha'),

                 Real(0.1, 10, name='reg_lambda')

               ]



model = xgboost.XGBRegressor(**STATIC_PARAMS)



X_train_tune, X_test_tune, y_train_tune, y_test_tune = train_test_split(X, Y, 

                                                                        test_size=0.3, 

                                                                        shuffle = True,

                                                                        random_state=RANDOM_STATE)

best_alg_params = {}



eval_dataset = [(X_test_tune, y_test_tune)]



n_fold = 5

cv_tune = KFold(n_splits=n_fold, shuffle=True, random_state=RANDOM_STATE)



start_time = time()

[TUNED_alg_PARAMS,reg_gp] = get_params_SKopt(

                                                model, 

                                                 X_train_tune, y_train_tune, 

                                                 space_SKopt, 

                                                 cv_tune,

                                                 alg = 'xgboost',

                                                 cat_features = [],

                                                 eval_dataset = eval_dataset,

                                                 UBM = True,

                                                 opt_method = 'forest_minimize',

                                                 verbose = True,

                                                 multi = False, 

                                                 scoring = 'neg_mean_absolute_error', 

                                                 n_best = 30,

                                                 total_time = 10800,                                       

                                            )



print('\nTime for tuning: {0:.2f} minutes'.format((time() - start_time)/60))

STATIC_PARAMS['n_estimators'] = 2000

NEW_PARAMS = {**STATIC_PARAMS, **TUNED_alg_PARAMS} 

print(NEW_PARAMS)



print('Достигнутое значение метрики: ', reg_gp.fun)



n_fold = 5

cv = KFold(n_splits=n_fold, shuffle=True, random_state=RANDOM_STATE)



oof = np.zeros(len(X))

prediction = np.zeros(len(X_test))

mae, r2 = [], []



for fold_n, (train_index, valid_index) in enumerate(cv.split(X)):

    print('\nFold', fold_n, 'started at', ctime())



    X_train = X.iloc[train_index,:]

    X_valid = X.iloc[valid_index,:]

    

    Y_train = Y.iloc[train_index]

    Y_valid = Y.iloc[valid_index]

          

    best_model = xgboost.XGBRegressor(**NEW_PARAMS)

    

    best_model.fit(X_train, Y_train, 

           eval_metric='mae',    

           eval_set=[(X_train, Y_train), (X_valid, Y_valid)],

           verbose=False,

           early_stopping_rounds = 100)

      

    y_pred = best_model.predict(X_valid, 

                               ntree_limit = best_model.best_ntree_limit)



    mae.append(mean_absolute_error(Y_valid, y_pred))

    r2.append(r2_score(Y_valid, y_pred))



    print('MAE: ', mean_absolute_error(Y_valid, y_pred))

    print('R2: ', r2_score(Y_valid, y_pred))



    prediction += best_model.predict(X_test,

                                    ntree_limit = best_model.best_ntree_limit)

        

prediction /= n_fold



print('='*45)

print('CV mean MAE: {0:.4f}, std: {1:.4f}.'.format(np.mean(mae), np.std(mae)))

print('CV mean R2:  {0:.4f}, std: {1:.4f}.'.format(np.mean(r2), np.std(r2)))



plotfig(best_model.predict(X), Y, 'Predicted vs. Actual responses for XGB', max(Y) + 0.1*max(Y))

submission = pd.DataFrame()

submission['segment_id'] = test_set.index

submission['time_to_eruption'] = prediction

submission.to_csv('submission.csv', header=True, index=False)