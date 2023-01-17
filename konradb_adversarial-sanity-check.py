from statsmodels.tsa.ar_model import AutoReg, ar_select_order

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from statsmodels.tsa.api import acf, pacf, graphics

from typing import List, Tuple, Union, NoReturn

from plotly.subplots import make_subplots

import plotly.graph_objects as go

import plotly.offline as py

import plotly.express as px

import cufflinks as cf

import plotly

from statsmodels.robust import mad

import matplotlib.pyplot as plt

from scipy.signal import butter

from scipy import signal

import seaborn as sns

from sklearn import *

import pandas as pd 

import numpy as np

import warnings

import scipy

import pywt

import os

import gc



import lightgbm as lgb

import xgboost as xgb



from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import cohen_kappa_score, mean_squared_error, f1_score, roc_auc_score

from functools import partial

import scipy as sp

import datetime



cf.go_offline()

py.init_notebook_mode()

cf.getThemes()

cf.set_config_file(theme='ggplot')

warnings.simplefilter('ignore')

pd.plotting.register_matplotlib_converters()

sns.mpl.rc('figure',figsize=(16, 6))

plt.style.use('ggplot')

sns.set_style('darkgrid')



from sklearn.linear_model import Ridge, ElasticNet

from sklearn.preprocessing import StandardScaler

# from bayes_opt import BayesianOptimization
nfolds = 5
def add_bathing_to_data(df : pd.DataFrame) -> pd.DataFrame :

    batches = df.shape[0] // 500000

    df['batch'] = 0

    for i in range(batches):

        idx = np.arange(i*500000, (i+1)*500000)

        df.loc[idx, 'batch'] = i + 1

    return df



def p5( x : pd.Series) -> pd.Series : return x.quantile(0.05)

def p95(x : pd.Series) -> pd.Series : return x.quantile(0.95)

def maddest(d, axis=None):

    return np.mean(np.absolute(d - np.mean(d, axis)), axis)



def high_pass_filter(x, low_cutoff=1000, sample_rate=10000):



    nyquist = 0.5 * sample_rate

    norm_low_cutoff = low_cutoff / nyquist

    print(norm_low_cutoff)

    sos = butter(10, Wn=[norm_low_cutoff], btype='highpass', output='sos')

    filtered_sig = signal.sosfilt(sos, x)



    return filtered_sig



def denoise_signal( x, wavelet='db4', level=1):

    

    coeff = pywt.wavedec( x, wavelet, mode="per" )

    sigma = (1/0.6745) * maddest( coeff[-level] )

    uthresh = sigma * np.sqrt( 2*np.log( len( x ) ) )

    coeff[1:] = ( pywt.threshold( i, value=uthresh, mode='hard' ) for i in coeff[1:] )

    return pywt.waverec( coeff, wavelet, mode='per' )
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

        d['range'+c] = (d['max'+c] - d['min'+c])

        d['abs_avg'+c] = ((d['abs_min'+c] + d['abs_max'+c]) / 2)

        for v in d:

            df[v] = df[c].map(d[v].to_dict())



        

    for c in [c1 for c1 in df.columns if c1 not in ['time', 'signal', 'open_channels', 'batch', 'batch_index', 'batch_slices', 'batch_slices2']]:

        df[c] = df[c] - df['signal']

       

    # rolling features

    windows = [50, 500]



    for win in windows:

        xroll = df.groupby(['batch'])['signal'].rolling(win, min_periods= 3)

        df['min_r'+str(win)] = np.array(xroll.min().fillna(0))

        df['max_r'+str(win)] = np.array(xroll.max().fillna(0))

        df['mean_r'+str(win)] = np.array(xroll.max().fillna(0))

        df['std_r'+str(win)] = np.array(xroll.std().fillna(0))

        df['skew_r'+str(win)] = np.array(xroll.skew().fillna(0))

        df['kurt_r'+str(win)] = np.array(xroll.kurt().fillna(0))

        df['q01_r'+str(win)] = np.array(xroll.quantile(0.01).fillna(0))

        df['q99_r'+str(win)] = np.array(xroll.quantile(0.99).fillna(0))

    

    



            

    return df
def MacroF1Metric(preds, dtrain):

    labels = dtrain.get_label()

    preds = np.round(np.clip(preds, 0, 10)).astype(int)

    score = metrics.f1_score(labels, preds, average = 'macro')

    return ('MacroF1Metric', score, True)



class OptimizedRounder(object):



    def __init__(self):

        self.coef_ = 0



    def loss(self, coef, X, y):

        X_p = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        return -metrics.f1_score(y, X_p, average = 'macro')



    def fit(self, X, y):

        loss_partial = partial(self.loss, X=X, y=y)

        initial_coef = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]

        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')



    def predict(self, X, coef):

        return (pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])).astype(np.int8)



    def coefficients(self):

        return self.coef_['x']

def optimize_predictions(preds, coeffs):

    

    preds[preds <= coeffs[0]] = 0

    preds[np.where(np.logical_and(preds > coeffs[0], preds <= coeffs[1]))] = 1

    preds[np.where(np.logical_and(preds > coeffs[1], preds <= coeffs[2]))] = 2

    preds[np.where(np.logical_and(preds > coeffs[2], preds <= coeffs[3]))] = 3

    preds[np.where(np.logical_and(preds > coeffs[3], preds <= coeffs[4]))] = 4

    preds[np.where(np.logical_and(preds > coeffs[4], preds <= coeffs[5]))] = 5

    preds[np.where(np.logical_and(preds > coeffs[5], preds <= coeffs[6]))] = 6

    preds[np.where(np.logical_and(preds > coeffs[6], preds <= coeffs[7]))] = 7

    preds[np.where(np.logical_and(preds > coeffs[7], preds <= coeffs[8]))] = 8

    preds[np.where(np.logical_and(preds > coeffs[8], preds <= coeffs[9]))] = 9

    preds[preds > coeffs[9]] = 10

    preds = preds.astype(np.int8)

    

    return preds
def shrink_memory(df, xcols):

    for colname in xcols:

#        print(colname)

        c_min = df[colname].min()

        c_max = df[colname].max()

        if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

            df[colname] = df[colname].astype(np.float16)

        elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

            df[colname] = df[colname].astype(np.float32)

        else:

            df[colname] = df[colname].astype(np.float64)    

    return df
base = os.path.abspath('/kaggle/input/liverpool-ion-switching/')

# base = os.path.abspath('../input/')



train = pd.read_csv(os.path.join(base + '/train.csv'))

# create features and shrink memory



train = features(train)



start_memory = train.memory_usage(deep= True).sum() / 1024 ** 2

print(start_memory)



xcols = [f for f in train.columns if f not in ['time', 'open_channels', 'batch', 'batch_index', 'batch_slices', 'batch_slices2']]



train = shrink_memory(train, xcols) 



end_memory = train.memory_usage().sum() / 1024**2

percent = 100 * (start_memory - end_memory) / start_memory

print('Mem. usage decreased from {:5.2f} Mb to {:5.2f} Mb ({:.1f}% reduction)'.format(start_memory, end_memory, percent))  

test  = pd.read_csv(os.path.join(base + '/test.csv'))



test = features(test)



start_memory = test.memory_usage(deep= True).sum() / 1024 ** 2

print(start_memory)



xcols = [f for f in test.columns if f not in ['time', 'open_channels', 'batch', 'batch_index', 'batch_slices', 'batch_slices2']]



test = shrink_memory(test, xcols) 



end_memory = test.memory_usage().sum() / 1024**2

percent = 100 * (start_memory - end_memory) / start_memory

print('Mem. usage decreased from {:5.2f} Mb to {:5.2f} Mb ({:.1f}% reduction)'.format(start_memory, end_memory, percent))  

dropcols = ['batch', 'batch_index', 'batch_slices', 'batch_slices2']



batch_train = train['batch'] 



train.drop(dropcols, axis = 1, inplace = True)

test.drop(dropcols, axis = 1, inplace = True)

id_train = train['time']

id_test = test['time']

ytrain = train['open_channels']





train.drop(['time', 'open_channels'], axis = 1, inplace = True)

test.drop(['time'], axis = 1, inplace = True)
train['is_test'] = 0

test['is_test'] = 1



ntr = train.shape[0]
xdat = pd.concat([train, test], axis = 0)

del train, test

ydat = xdat['is_test']

xdat.drop('is_test', axis = 1, inplace = True)
skf = StratifiedKFold(n_splits= nfolds, shuffle= True, random_state=42)





mvalid = np.zeros((xdat.shape[0],1))

# for fold, (tr_ind, val_ind) in enumerate(skf.split(ytrain, groups = batch_train)):

for (fold, (id0, id1)) in enumerate(skf.split(xdat, ydat)):

    

    # prepare split

    x0, x1 = xdat.iloc[id0], xdat.iloc[id1]

    y0, y1 = ydat.iloc[id0], ydat.iloc[id1]

      

    sc0 = StandardScaler()

    sc0.fit(x0)

    

    x0 = sc0.transform(x0)

    x1 = sc0.transform(x1)



    model = Ridge(alpha = 30)

    model.fit(x0, y0)

            

    mvalid[id1,0] = model.predict(x1)

    

    print(roc_auc_score(y1, mvalid[id1,0]))



    del x0, x1, y0, y1

    

#     mvalid[id1,0] = model.predict(x1, num_iteration=model.best_iteration_)

#     mfull[:,fold] = model.predict(test, num_iteration = model.best_iteration_)

    
df_out = pd.DataFrame()

df_out['time'] = id_train

df_out['wgt'] = mvalid[0:ntr]

df_out.to_csv('av_weights.csv', index = False)
print(roc_auc_score(ydat, mvalid))