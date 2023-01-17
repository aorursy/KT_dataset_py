# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import gc

import time

import logging

import datetime

import warnings

import numpy as np

import pandas as pd

import seaborn as sns

import xgboost as xgb

import lightgbm as lgb

from scipy import stats

from scipy.signal import hann

from tqdm import tqdm_notebook

import matplotlib.pyplot as plt

from scipy.signal import hilbert

from scipy.signal import convolve

from catboost import CatBoostRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import mean_absolute_error

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import KFold,StratifiedKFold, RepeatedKFold

warnings.filterwarnings("ignore")



#from sklearn.linear_model import LinearRegression as lmr

from xgboost import XGBRegressor as xgbr

from sklearn.pipeline import make_pipeline as mpp

from sklearn.preprocessing import Imputer as imp

from sklearn.ensemble import AdaBoostRegressor as adr

from sklearn.model_selection import cross_val_score as cvs

from sklearn.linear_model import Lasso as lss

from sklearn.linear_model import Ridge as rdg

from sklearn.linear_model import ElasticNet as ecn



# Any results you write to the current directory are saved as output.
%%time

train = pd.read_csv('../input/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})
train.head()
rows =150_000

sgt = int(np.floor(train.shape[0] / rows))
#Feature Engeneering

def data_feature(arr,abs_values=False):

    idx = np.array(range(len(arr)))

    if abs_values:

        arr = np.abs(arr)

    lr = LinearRegression()

    lr.fit(idx.reshape(-1,1),arr)

    return lr.coef_[0]



def unknow_func(x,length_sta,length_lta):

    sta = np.cumsum( x ** 2)

    sta = np.require(sta,dtype=np.float)# convert to float

    lta = sta.copy()

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
def sigmoid(z):

    

    return 1 / (1 + np.exp(-z))
def cost(theta, e, d, lr) :

    theta = np.matrix(theta) 

    e = np.matrix(e)

    d = np.matrix(d)

        

    fst = np.multiply(-e, np.log(sigmoid(d * theta.T)))

    scd = np.multiply((1 - e), np.log(1 - sigmoid(d * theta.T)))

    reg = (lr / 2 * len(e)) * np.sum(np.power(theta[:, 1:theta.shape[1]], 2))

    

    return np.sum(fst - scd) / (len(e)) + reg 



e = train['acoustic_data'].values[-1]

f = train['time_to_failure'].values[-1]



ef = cost(- 0.04194, e, f, 0.33)

ee = cost(- 0.02624, e, f, 0.33)



print(ef)

print(ee)
def EvalutedHypothesis(theta, e, d, lr) :

    theta = np.matrix(theta) 

    e = np.matrix(e)

    d = np.matrix(d)

        

    fst = np.multiply(-e, np.log(sigmoid(d * theta.T)))

    scd = np.multiply((1 - e), np.log(1 - sigmoid(d * theta.T)))

    reg = (lr / 2 * len(e)) * np.sum(np.power(theta[:, 1:theta.shape[1]], 2))

    

    return ((np.sum(fst - scd) ** 2) / (len(e))) + reg 



aa = EvalutedHypothesis(- 0.04194, e, f, 0.33)

bb = EvalutedHypothesis(- 0.02624, e, f, 0.33)



print(aa)

print(bb)
xtr = pd.DataFrame(index = range(sgt), dtype = np.float64)

ytr = pd.DataFrame(index = range(sgt), dtype = np.float64, columns = ['time_to_failure'])



mn = train['acoustic_data'].mean()

sd = train['acoustic_data'].std()

mx = train['acoustic_data'].max()

mi = train['acoustic_data'].min()

tt = np.abs(train['acoustic_data']).sum() 
def create_features(seg_id, seg, xtr):

    X = pd.Series(seg['acoustic_data'].values)

    zc = np.fft.fft(X)

    

    realFFT = np.real(zc)

    imagFFT = np.imag(zc)

    

    xtr.loc[seg_id, 'Sigmoid_call_std'] = sigmoid(X.std())

    xtr.loc[seg_id, 'Sigmoid_call_mean'] = sigmoid(X.mean())

    xtr.loc[seg_id, 'Cost_func_trn_std'] = cost(-2.555912182174505, X.std(), sd, 0.33)

    xtr.loc[seg_id, 'Cost_func_trn_mean'] = cost(-2.555912182174505, X.mean(), mn, 0.33)

    xtr.loc[seg_id, 'Cost_func_tst_std'] = cost(-1.8537597012091438, X.std(), sd, 0.33)

    xtr.loc[seg_id, 'Cost_func_tst_mean'] = cost(-1.8537597012091438, X.mean(), mn, 0.33)

    

    xtr.loc[seg_id, 'EvalutedHypothesis_func_trn_std'] = EvalutedHypothesis(-6.53268708298804, X.std(), sd, 0.33)

    xtr.loc[seg_id, 'EvalutedHypothesis_func_trn_mean'] = EvalutedHypothesis(-6.53268708298804, X.mean(), mn, 0.33)

    xtr.loc[seg_id, 'EvalutedHypothesis_func_tst_std'] = EvalutedHypothesis(-3.436425029827014, X.std(), sd, 0.33)

    xtr.loc[seg_id, 'EvalutedHypothesis_tst_mean'] = EvalutedHypothesis(-3.436425029827014, X.mean(), mn, 0.33)

    

    xtr.loc[seg_id, 'mean'] = X.mean()

    xtr.loc[seg_id, 'std'] = X.std()

    xtr.loc[seg_id, 'max'] = X.max()

    xtr.loc[seg_id, 'min'] = X.min()

    xtr.loc[seg_id, 'kurtosis'] = X.kurtosis()

    xtr.loc[seg_id, 'skew'] = X.skew()

    

    xtr.loc[seg_id, 'quantile0'] = X.quantile()

    xtr.loc[seg_id, 'quantile1'] = np.count_nonzero(X < np.quantile(X,0.05))

    xtr.loc[seg_id, 'quantile2'] = np.count_nonzero(X < np.quantile(X,0.10))

    xtr.loc[seg_id, 'quantile3'] = np.count_nonzero(X > np.quantile(X,0.15))

    xtr.loc[seg_id, 'quantile4'] = np.count_nonzero(X > np.quantile(X,0.20))

    xtr.loc[seg_id, 'quantile5'] = np.count_nonzero(X < np.quantile(X,0.25))

    xtr.loc[seg_id, 'quantile6'] = np.count_nonzero(X < np.quantile(X,0.30))

    xtr.loc[seg_id, 'quantile7 '] = np.count_nonzero(X > np.quantile(X,0.35))

    xtr.loc[seg_id, 'quantile8'] = np.count_nonzero(X > np.quantile(X,0.40))

    xtr.loc[seg_id, 'quantile9'] = np.count_nonzero(X < np.quantile(X,0.45))

    xtr.loc[seg_id, 'quantile10'] = np.count_nonzero(X < np.quantile(X,0.50))

    

    xtr.loc[seg_id, 'data_feature0'] = data_feature(X)

    xtr.loc[seg_id, 'data_feature1'] = data_feature(X, abs_values = True)

    xtr.loc[seg_id, 'unknow_func0'] = unknow_func(X, 500, 10000).mean()

    xtr.loc[seg_id, 'unknow_func1'] = unknow_func(X, 625, 25000).mean()

    

    xtr.loc[seg_id, 'absolute_max'] = np.abs(X).max()

    xtr.loc[seg_id, 'absolute_min'] = np.abs(X).min()    

    

    xtr.loc[seg_id, 'real_mean'] = realFFT.mean()

    xtr.loc[seg_id, 'real_std'] = realFFT.std()

    xtr.loc[seg_id, 'real_max'] = realFFT.max()

    xtr.loc[seg_id, 'real_min'] = realFFT.min()

    xtr.loc[seg_id, 'image_mean'] = imagFFT.mean()

    xtr.loc[seg_id, 'image_std'] = imagFFT.std()

    xtr.loc[seg_id, 'image_max'] = imagFFT.max()

    xtr.loc[seg_id, 'image_min'] = imagFFT.min()    

    

    xtr.loc[seg_id, 'Roll_second_col_50000_std'] = X[:50000].std()

    xtr.loc[seg_id, 'Roll_first_col_-50000_std'] = X[-50000:].std()

    xtr.loc[seg_id, 'Roll_second_col_10000_std'] = X[:10000].std()

    xtr.loc[seg_id, 'Roll_first_col_-10000_std'] = X[-10000:].std()

    xtr.loc[seg_id, 'Roll_second_col_50000_mean'] = X[:50000].mean()

    xtr.loc[seg_id, 'Roll_first_col_-50000_mean'] = X[-50000:].mean()

    xtr.loc[seg_id, 'Roll_second_col_10000_mean'] = X[:10000].mean()

    xtr.loc[seg_id, 'Roll_first_col_-10000_mean'] = X[-10000:].mean()

    xtr.loc[seg_id, 'Roll_second_col_50000_min'] = X[:50000].min()

    xtr.loc[seg_id, 'Roll_first_col_-10000_min'] = X[-50000:].min()

    xtr.loc[seg_id, 'Roll_second_col_50000_min'] = X[:10000].min()

    xtr.loc[seg_id, 'Roll_first_col_-10000_min'] = X[-10000:].min()

    xtr.loc[seg_id, 'Roll_second_col_50000_max'] = X[:50000].max()

    xtr.loc[seg_id, 'Roll_first_col_-50000_max'] = X[-50000:].max()

    xtr.loc[seg_id, 'Roll_second_col_10000_max'] = X[:10000].max()

    xtr.loc[seg_id, 'Roll_first_col_-10000_max'] = X[-10000:].max()

    

    for win in [10, 100, 1000] :

        xrllsd = X.rolling(win).std().dropna().values 

        xrllmn = X.rolling(win).std().dropna().values

        

        xtr.loc[seg_id, 'WindowsPartition1.1' + str(win)] = xrllsd.mean()

        xtr.loc[seg_id, 'WindowsPartition1.2' + str(win)] = xrllsd.std()

        xtr.loc[seg_id, 'WindowsPartitio1.3' + str(win)] = xrllsd.max()

        xtr.loc[seg_id, 'WindowsPartition1.4' + str(win)] = xrllsd.min()

        xtr.loc[seg_id, 'WindowsPartition1.5' + str(win)] = np.quantile(xrllsd, 0.0125)

        xtr.loc[seg_id, 'WindowsPartition1.6' + str(win)] = np.quantile(xrllsd, 0.0750)

        xtr.loc[seg_id, 'WindowsPartition1.7' + str(win)] = np.quantile(xrllsd, 0.125)

        xtr.loc[seg_id, 'WindowsPartition1.8' + str(win)] = np.quantile(xrllsd, 0.105)

        xtr.loc[seg_id, 'WindowsPartition1.9' + str(win)] = np.mean(np.diff(xrllsd))

        xtr.loc[seg_id, 'WindowsPartition1.10' + str(win)] = np.mean(np.nonzero((np.diff(xrllsd) / xrllsd[:-1]))[0])

        xtr.loc[seg_id, 'WindowsPartition1.11' + str(win)] = np.abs(xrllsd).max()

        xtr.loc[seg_id, 'WindowsPartitio1.12' + str(win)] = xrllsd.mean() - xrllsd.std()

        xtr.loc[seg_id, 'WindowsPartition1.13' + str(win)] = xrllsd.max()  - xrllsd.min()

        xtr.loc[seg_id, 'WindowsPartition1.14' + str(win)] = np.quantile(xrllsd, 0.0250) - np.quantile(xrllsd, 0.0125)

        xtr.loc[seg_id, 'WindowsPartition1.15' + str(win)] = np.quantile(xrllsd, 0.0900) - np.quantile(xrllsd, 0.0750)

        xtr.loc[seg_id, 'WindowsPartition1.16' + str(win)] = np.quantile(xrllsd, 0.250) - np.quantile(xrllsd, 0.125)

        xtr.loc[seg_id, 'WindowsPartition1.17' + str(win)] = np.quantile(xrllsd, 0.210) - np.quantile(xrllsd, 0.105)

        xtr.loc[seg_id, 'WindowsPartition1.18' + str(win)] = np.quantile(xrllsd, 0.2575) 

        xtr.loc[seg_id, 'WindowsPartition1.19' + str(win)] = np.mean(np.nonzero((np.diff(xrllsd) / xrllsd[:-1]))[0]) - np.abs(xrllsd).max()

        xtr.loc[seg_id, 'WindowsPartition1.20' + str(win)] = np.mean(np.nonzero((np.diff(xrllsd) / xrllsd[:-1]))[0]) + np.abs(xrllsd).max()

        xtr.loc[seg_id, 'WindowsPartition1.21' + str(win)] = sigmoid(xrllsd.std())

        xtr.loc[seg_id, 'WindowsPartition1.22' + str(win)] = sigmoid(xrllsd.mean())

        xtr.loc[seg_id, 'WindowsPartition1.23' + str(win)] = sigmoid(xrllsd.std()) - sigmoid(xrllsd.mean())

        xtr.loc[seg_id, 'WindowsPartition1.24' + str(win)] = sigmoid(xrllsd.std()) + sigmoid(xrllsd.mean())

        xtr.loc[seg_id, 'WindowsPartition1.25' + str(win)] = cost(-2.555912182174505, xrllsd.std(), sd, 0.33)

        xtr.loc[seg_id, 'WindowsPartition1.26' + str(win)] = cost(-2.555912182174505, xrllsd.mean(), sd, 0.33)

        xtr.loc[seg_id, 'WindowsPartition1.27' + str(win)] = cost(-2.555912182174505, xrllsd.std(), sd, 0.25)

        xtr.loc[seg_id, 'WindowsPartition1.28' + str(win)] = cost(-2.555912182174505, xrllsd.mean(), mn, 0.25)

        xtr.loc[seg_id, 'WindowsPartition1.29' + str(win)] = cost(-1.8537597012091438, xrllsd.std(), sd, 0.33)

        xtr.loc[seg_id, 'WindowsPartition1.30' + str(win)] = cost(-1.8537597012091438, xrllsd.mean(), mn, 0.33)

        xtr.loc[seg_id, 'WindowsPartition1.31' + str(win)] = cost(-1.8537597012091438, xrllsd.std(), sd, 0.25)

        xtr.loc[seg_id, 'WindowsPartition1.32' + str(win)] = cost(-1.8537597012091438, xrllsd.mean(), mn, 0.25)

        xtr.loc[seg_id, 'WindowsPartition1.33' + str(win)] = EvalutedHypothesis(-6.53268708298804, xrllsd.std(), sd, 0.33)

        xtr.loc[seg_id, 'WindowsPartition1.34' + str(win)] = EvalutedHypothesis(-6.53268708298804, xrllsd.std(), sd, 0.25)

        xtr.loc[seg_id, 'WindowsPartition1.35' + str(win)] = EvalutedHypothesis(-6.53268708298804, xrllsd.mean(), mn, 0.33)

        xtr.loc[seg_id, 'WindowsPartition1.36' + str(win)] = EvalutedHypothesis(-6.53268708298804, xrllsd.mean(), mn, 0.25)

        xtr.loc[seg_id, 'WindowsPartition1.37' + str(win)] = EvalutedHypothesis(-3.436425029827014, xrllsd.std(), sd, 0.33)

        xtr.loc[seg_id, 'WindowsPartition1.38' + str(win)] = EvalutedHypothesis(-3.436425029827014, xrllsd.std(), sd, 0.25)

        xtr.loc[seg_id, 'WindowsPartition1.39' + str(win)] = EvalutedHypothesis(-3.436425029827014, xrllsd.mean(), mn, 0.33)

        xtr.loc[seg_id, 'WindowsPartition1.40' + str(win)] = EvalutedHypothesis(-3.436425029827014, xrllsd.mean(), mn, 0.25)

        xtr.loc[seg_id, 'WindowsPartition1.41' + str(win)] = EvalutedHypothesis(-6.53268708298804, xrllsd.std(), sd, 0.33) - sigmoid(xrllsd.std())

        xtr.loc[seg_id, 'WindowsPartition1.42' + str(win)] = EvalutedHypothesis(-3.436425029827014, xrllsd.std(), sd, 0.33) - sigmoid(xrllsd.std())

        xtr.loc[seg_id, 'WindowsPartition1.43' + str(win)] = EvalutedHypothesis(-6.53268708298804, xrllsd.mean(), mn, 0.33) - sigmoid(xrllsd.mean())

        xtr.loc[seg_id, 'WindowsPartition1.44' + str(win)] = EvalutedHypothesis(-3.436425029827014, xrllsd.mean(), mn, 0.33) - sigmoid(xrllsd.mean())

        xtr.loc[seg_id, 'WindowsPartition1.45' + str(win)] = EvalutedHypothesis(-6.53268708298804, xrllsd.std(), sd, 0.25) - sigmoid(xrllsd.std())

        xtr.loc[seg_id, 'WindowsPartition1.46' + str(win)] = EvalutedHypothesis(-3.436425029827014, xrllsd.std(), sd, 0.25) - sigmoid(xrllsd.std())

        xtr.loc[seg_id, 'WindowsPartition1.47' + str(win)] = EvalutedHypothesis(-6.53268708298804, xrllsd.mean(), mn, 0.25) - sigmoid(xrllsd.mean())

        xtr.loc[seg_id, 'WindowsPartition1.48' + str(win)] = EvalutedHypothesis(-3.436425029827014, xrllsd.mean(), mn, 0.25) - sigmoid(xrllsd.mean())

        

        xtr.loc[seg_id, 'WindowsPartition2.1' + str(win)] = xrllmn.mean()

        xtr.loc[seg_id, 'WindowsPartition2.2' + str(win)] = xrllmn.std()

        xtr.loc[seg_id, 'WindowsPartition2.3' + str(win)] = xrllmn.max()

        xtr.loc[seg_id, 'WindowsPartition2.4' + str(win)] = xrllmn.min()

        xtr.loc[seg_id, 'WindowsPartition2.5' + str(win)] = np.quantile(xrllmn, 0.0125)

        xtr.loc[seg_id, 'WindowsPartition2.6' + str(win)] = np.quantile(xrllmn, 0.0750)

        xtr.loc[seg_id, 'WindowsPartition2.7' + str(win)] = np.quantile(xrllmn, 0.125)

        xtr.loc[seg_id, 'WindowsPartition2.8' + str(win)] = np.quantile(xrllmn, 0.105)

        xtr.loc[seg_id, 'WindowsPartition2.9' + str(win)] = np.mean(np.diff(xrllsd))

        xtr.loc[seg_id, 'WindowsPartition2.10' + str(win)] = np.mean(np.nonzero((np.diff(xrllmn) / xrllmn[:-1]))[0])

        xtr.loc[seg_id, 'WindowsPartition2.11' + str(win)] = np.abs(xrllmn).max()

        xtr.loc[seg_id, 'WindowsPartition2.12' + str(win)] = xrllmn.mean() - xrllmn.std()

        xtr.loc[seg_id, 'WindowsPartition2.13' + str(win)] = xrllmn.max()  - xrllmn.min()

        xtr.loc[seg_id, 'WindowsPartition2.14' + str(win)] = np.quantile(xrllmn, 0.0250) - np.quantile(xrllmn, 0.0125)

        xtr.loc[seg_id, 'WindowsPartition2.15' + str(win)] = np.quantile(xrllmn, 0.0900) - np.quantile(xrllmn, 0.0750)

        xtr.loc[seg_id, 'WindowsPartition2.16' + str(win)] = np.quantile(xrllmn, 0.250) - np.quantile(xrllmn, 0.125)

        xtr.loc[seg_id, 'WindowsPartition2.17' + str(win)] =  np.quantile(xrllmn, 0.210) - np.quantile(xrllmn, 0.105)

        xtr.loc[seg_id, 'WindowsPartition2.18' + str(win)] = np.quantile(xrllmn, 0.2575)

        xtr.loc[seg_id, 'WindowsPartition2.19' + str(win)] = np.mean(np.nonzero((np.diff(xrllmn) / xrllmn[:-1]))[0]) - np.abs(xrllmn).max()

        xtr.loc[seg_id, 'WindowsPartition2.20' + str(win)] = np.mean(np.nonzero((np.diff(xrllmn) / xrllmn[:-1]))[0]) + np.abs(xrllmn).max()

        xtr.loc[seg_id, 'WindowsPartition2.21' + str(win)] = sigmoid(xrllmn.std())

        xtr.loc[seg_id, 'WindowsPartition2.22' + str(win)] = sigmoid(xrllmn.mean())

        xtr.loc[seg_id, 'WindowsPartition2.23' + str(win)] = sigmoid(xrllmn.std()) - sigmoid(xrllmn.mean())

        xtr.loc[seg_id, 'WindowsPartition2.24' + str(win)] = sigmoid(xrllmn.std()) + sigmoid(xrllmn.mean())

        xtr.loc[seg_id, 'WindowsPartition2.25' + str(win)] = cost(-2.555912182174505, xrllmn.std(), sd, 0.33)

        xtr.loc[seg_id, 'WindowsPartition2.26' + str(win)] = cost(-2.555912182174505, xrllmn.mean(), sd, 0.33)

        xtr.loc[seg_id, 'WindowsPartition2.27' + str(win)] = cost(-2.555912182174505, xrllmn.std(), sd, 0.25)

        xtr.loc[seg_id, 'WindowsPartition2.28' + str(win)] = cost(-2.555912182174505, xrllmn.mean(), mn, 0.25)

        xtr.loc[seg_id, 'WindowsPartition2.29' + str(win)] = cost(-1.8537597012091438, xrllmn.std(), sd, 0.33)

        xtr.loc[seg_id, 'WindowsPartition2.30' + str(win)] = cost(-1.8537597012091438, xrllmn.mean(), mn, 0.33)

        xtr.loc[seg_id, 'WindowsPartition2.31' + str(win)] = cost(-1.8537597012091438, xrllmn.std(), sd, 0.25)

        xtr.loc[seg_id, 'WindowsPartition2.32' + str(win)] = cost(-1.8537597012091438, xrllmn.mean(), mn, 0.25)

        xtr.loc[seg_id, 'WindowsPartition2.33' + str(win)] = EvalutedHypothesis(-6.53268708298804, xrllmn.std(), sd, 0.33)

        xtr.loc[seg_id, 'WindowsPartition2.34' + str(win)] = EvalutedHypothesis(-6.53268708298804, xrllmn.std(), sd, 0.25)

        xtr.loc[seg_id, 'WindowsPartition2.35' + str(win)] = EvalutedHypothesis(-6.53268708298804, xrllmn.mean(), mn, 0.33)

        xtr.loc[seg_id, 'WindowsPartition2.36' + str(win)] = EvalutedHypothesis(-6.53268708298804, xrllmn.mean(), mn, 0.25)

        xtr.loc[seg_id, 'WindowsPartition2.37' + str(win)] = EvalutedHypothesis(-3.436425029827014, xrllmn.std(), sd, 0.33)

        xtr.loc[seg_id, 'WindowsPartition2.38' + str(win)] = EvalutedHypothesis(-3.436425029827014, xrllmn.std(), sd, 0.25)

        xtr.loc[seg_id, 'WindowsPartition2.39' + str(win)] = EvalutedHypothesis(-3.436425029827014, xrllmn.mean(), mn, 0.33)

        xtr.loc[seg_id, 'WindowsPartition2.40' + str(win)] = EvalutedHypothesis(-3.436425029827014, xrllmn.mean(), mn, 0.25)

        xtr.loc[seg_id, 'WindowsPartition2.41' + str(win)] = EvalutedHypothesis(-6.53268708298804, xrllmn.std(), sd, 0.33) - sigmoid(xrllmn.std())

        xtr.loc[seg_id, 'WindowsPartition2.42' + str(win)] = EvalutedHypothesis(-3.436425029827014, xrllmn.std(), sd, 0.33) - sigmoid(xrllmn.std())

        xtr.loc[seg_id, 'WindowsPartition2.43' + str(win)] = EvalutedHypothesis(-6.53268708298804, xrllmn.mean(), mn, 0.33) - sigmoid(xrllmn.mean())

        xtr.loc[seg_id, 'WindowsPartition2.44' + str(win)] = EvalutedHypothesis(-3.436425029827014, xrllmn.mean(), mn, 0.33) - sigmoid(xrllmn.mean())

        xtr.loc[seg_id, 'WindowsPartition2.45' + str(win)] = EvalutedHypothesis(-6.53268708298804, xrllmn.std(), sd, 0.25) - sigmoid(xrllmn.std())

        xtr.loc[seg_id, 'WindowsPartition2.46' + str(win)] = EvalutedHypothesis(-3.436425029827014, xrllmn.std(), sd, 0.25) - sigmoid(xrllmn.std())

        xtr.loc[seg_id, 'WindowsPartition2.47' + str(win)] = EvalutedHypothesis(-6.53268708298804, xrllmn.mean(), mn, 0.25) - sigmoid(xrllmn.mean())

        xtr.loc[seg_id, 'WindowsPartition2.48' + str(win)] = EvalutedHypothesis(-3.436425029827014, xrllmn.mean(), mn, 0.25) - sigmoid(xrllmn.mean())

        

        

    for win in [15, 125, 1125] :

        xrllsd = X.rolling(win).std().dropna().values 

        xrllmn = X.rolling(win).std().dropna().values

        

        xtr.loc[seg_id, 'WindowsPartition3.1' + str(win)] = xrllsd.mean()

        xtr.loc[seg_id, 'WindowsPartition3.2' + str(win)] = xrllsd.std()

        xtr.loc[seg_id, 'WindowsPartition3.3' + str(win)] = xrllsd.max()

        xtr.loc[seg_id, 'WindowsPartition3.4' + str(win)] = xrllsd.min()

        xtr.loc[seg_id, 'WindowsPartition3.5' + str(win)] = np.quantile(xrllsd, 0.0125)

        xtr.loc[seg_id, 'WindowsPartition3.6' + str(win)] = np.quantile(xrllsd, 0.0750)

        xtr.loc[seg_id, 'WindowsPartition3.7' + str(win)] = np.quantile(xrllsd, 0.125)

        xtr.loc[seg_id, 'WindowsPartition3.8' + str(win)] = np.quantile(xrllsd, 0.105)

        xtr.loc[seg_id, 'WindowsPartition3.9' + str(win)] = np.mean(np.diff(xrllsd))

        xtr.loc[seg_id, 'WindowsPartition3.10' + str(win)] = np.mean(np.nonzero((np.diff(xrllsd) / xrllsd[:-1]))[0])

        xtr.loc[seg_id, 'WindowsPartition3.11' + str(win)] = np.abs(xrllsd).max()

        xtr.loc[seg_id, 'WindowsPartition3.12' + str(win)] = xrllsd.mean() - xrllsd.std()

        xtr.loc[seg_id, 'WindowsPartition3.13' + str(win)] = xrllsd.max()  - xrllsd.min()

        xtr.loc[seg_id, 'WindowsPartition3.14' + str(win)] = np.quantile(xrllsd, 0.0250) - np.quantile(xrllsd, 0.0125)

        xtr.loc[seg_id, 'WindowsPartition3.15' + str(win)] = np.quantile(xrllsd, 0.0900) - np.quantile(xrllsd, 0.0750)

        xtr.loc[seg_id, 'WindowsPartition3.16' + str(win)] = np.quantile(xrllsd, 0.250) - np.quantile(xrllsd, 0.125)

        xtr.loc[seg_id, 'WindowsPartition3.17' + str(win)] = np.quantile(xrllsd, 0.210) - np.quantile(xrllsd, 0.105)

        xtr.loc[seg_id, 'WindowsPartition3.18' + str(win)] = np.quantile(xrllsd, 0.2575) 

        xtr.loc[seg_id, 'WindowsPartition3.19' + str(win)] = np.mean(np.nonzero((np.diff(xrllsd) / xrllsd[:-1]))[0]) - np.abs(xrllsd).max()

        xtr.loc[seg_id, 'WindowsPartition3.20' + str(win)] = np.mean(np.nonzero((np.diff(xrllsd) / xrllsd[:-1]))[0]) + np.abs(xrllsd).max()

        xtr.loc[seg_id, 'WindowsPartition3.21' + str(win)] = sigmoid(xrllsd.std())

        xtr.loc[seg_id, 'WindowsPartition3.22' + str(win)] = sigmoid(xrllsd.mean())

        xtr.loc[seg_id, 'WindowsPartition3.23' + str(win)] = sigmoid(xrllsd.std()) - sigmoid(xrllsd.mean())

        xtr.loc[seg_id, 'WindowsPartition3.24' + str(win)] = sigmoid(xrllsd.std()) + sigmoid(xrllsd.mean())

        xtr.loc[seg_id, 'WindowsPartition3.25' + str(win)] = cost(-2.555912182174505, xrllsd.std(), sd, 0.33)

        xtr.loc[seg_id, 'WindowsPartition3.26' + str(win)] = cost(-2.555912182174505, xrllsd.mean(), sd, 0.33)

        xtr.loc[seg_id, 'WindowsPartition3.27' + str(win)] = cost(-2.555912182174505, xrllsd.std(), sd, 0.25)

        xtr.loc[seg_id, 'WindowsPartition3.28' + str(win)] = cost(-2.555912182174505, xrllsd.mean(), mn, 0.25)

        xtr.loc[seg_id, 'WindowsPartition3.29' + str(win)] = cost(-1.8537597012091438, xrllsd.std(), sd, 0.33)

        xtr.loc[seg_id, 'WindowsPartition3.30' + str(win)] = cost(-1.8537597012091438, xrllsd.mean(), mn, 0.33)

        xtr.loc[seg_id, 'WindowsPartition3.31' + str(win)] = cost(-1.8537597012091438, xrllsd.std(), sd, 0.25)

        xtr.loc[seg_id, 'WindowsPartition3.32' + str(win)] = cost(-1.8537597012091438, xrllsd.mean(), mn, 0.25)

        xtr.loc[seg_id, 'WindowsPartition3.33' + str(win)] = EvalutedHypothesis(-6.53268708298804, xrllsd.std(), sd, 0.33)

        xtr.loc[seg_id, 'WindowsPartition3.34' + str(win)] = EvalutedHypothesis(-6.53268708298804, xrllsd.std(), sd, 0.25)

        xtr.loc[seg_id, 'WindowsPartition3.35' + str(win)] = EvalutedHypothesis(-6.53268708298804, xrllsd.mean(), mn, 0.33)

        xtr.loc[seg_id, 'WindowsPartition3.36' + str(win)] = EvalutedHypothesis(-6.53268708298804, xrllsd.mean(), mn, 0.25)

        xtr.loc[seg_id, 'WindowsPartition3.37' + str(win)] = EvalutedHypothesis(-3.436425029827014, xrllsd.std(), sd, 0.33)

        xtr.loc[seg_id, 'WindowsPartition3.38' + str(win)] = EvalutedHypothesis(-3.436425029827014, xrllsd.std(), sd, 0.25)

        xtr.loc[seg_id, 'WindowsPartition3.39' + str(win)] = EvalutedHypothesis(-3.436425029827014, xrllsd.mean(), mn, 0.33)

        xtr.loc[seg_id, 'WindowsPartition3.40' + str(win)] = EvalutedHypothesis(-3.436425029827014, xrllsd.mean(), mn, 0.25)

        xtr.loc[seg_id, 'WindowsPartition3.41' + str(win)] = EvalutedHypothesis(-6.53268708298804, xrllsd.std(), sd, 0.33) - sigmoid(xrllsd.std())

        xtr.loc[seg_id, 'WindowsPartition3.42' + str(win)] = EvalutedHypothesis(-3.436425029827014, xrllsd.std(), sd, 0.33) - sigmoid(xrllsd.std())

        xtr.loc[seg_id, 'WindowsPartition3.43' + str(win)] = EvalutedHypothesis(-6.53268708298804, xrllsd.mean(), mn, 0.33) - sigmoid(xrllsd.mean())

        xtr.loc[seg_id, 'WindowsPartition3.44' + str(win)] = EvalutedHypothesis(-3.436425029827014, xrllsd.mean(), mn, 0.33) - sigmoid(xrllsd.mean())

        xtr.loc[seg_id, 'WindowsPartition3.45' + str(win)] = EvalutedHypothesis(-6.53268708298804, xrllsd.std(), sd, 0.25) - sigmoid(xrllsd.std())

        xtr.loc[seg_id, 'WindowsPartition3.46' + str(win)] = EvalutedHypothesis(-3.436425029827014, xrllsd.std(), sd, 0.25) - sigmoid(xrllsd.std())

        xtr.loc[seg_id, 'WindowsPartition3.47' + str(win)] = EvalutedHypothesis(-6.53268708298804, xrllsd.mean(), mn, 0.25) - sigmoid(xrllsd.mean())

        xtr.loc[seg_id, 'WindowsPartition3.48' + str(win)] = EvalutedHypothesis(-3.436425029827014, xrllsd.mean(), mn, 0.25) - sigmoid(xrllsd.mean())

        

        xtr.loc[seg_id, 'WindowsPartition4.1' + str(win)] = xrllmn.mean()

        xtr.loc[seg_id, 'WindowsPartition4.2' + str(win)] = xrllmn.std()

        xtr.loc[seg_id, 'WindowsPartition4.3' + str(win)] = xrllmn.max()

        xtr.loc[seg_id, 'WindowsPartition4.4' + str(win)] = xrllmn.min()

        xtr.loc[seg_id, 'WindowsPartition4.5' + str(win)] = np.quantile(xrllmn, 0.0125)

        xtr.loc[seg_id, 'WindowsPartition4.6' + str(win)] = np.quantile(xrllmn, 0.0750)

        xtr.loc[seg_id, 'WindowsPartition4.7' + str(win)] = np.quantile(xrllmn, 0.125)

        xtr.loc[seg_id, 'WindowsPartition4.8' + str(win)] = np.quantile(xrllmn, 0.105)

        xtr.loc[seg_id, 'WindowsPartition4.9' + str(win)] = np.mean(np.diff(xrllsd))

        xtr.loc[seg_id, 'WindowsPartition4.10' + str(win)] = np.mean(np.nonzero((np.diff(xrllmn) / xrllmn[:-1]))[0])

        xtr.loc[seg_id, 'WindowsPartition4.11' + str(win)] = np.abs(xrllmn).max()

        xtr.loc[seg_id, 'WindowsPartition4.12' + str(win)] = xrllmn.mean() - xrllmn.std()

        xtr.loc[seg_id, 'WindowsPartition4.13' + str(win)] = xrllmn.max()  - xrllmn.min()

        xtr.loc[seg_id, 'WindowsPartition4.14' + str(win)] = np.quantile(xrllmn, 0.0250) - np.quantile(xrllmn, 0.0125)

        xtr.loc[seg_id, 'WindowsPartition4.15' + str(win)] = np.quantile(xrllmn, 0.0900) - np.quantile(xrllmn, 0.0750)

        xtr.loc[seg_id, 'WindowsPartition4.16' + str(win)] = np.quantile(xrllmn, 0.250) - np.quantile(xrllmn, 0.125)

        xtr.loc[seg_id, 'WindowsPartition4.17' + str(win)] =  np.quantile(xrllmn, 0.210) - np.quantile(xrllmn, 0.105)

        xtr.loc[seg_id, 'WindowsPartition4.18' + str(win)] = np.quantile(xrllmn, 0.2575)

        xtr.loc[seg_id, 'WindowsPartition4.19' + str(win)] = np.mean(np.nonzero((np.diff(xrllmn) / xrllmn[:-1]))[0]) - np.abs(xrllmn).max()

        xtr.loc[seg_id, 'WindowsPartition4.20' + str(win)] = np.mean(np.nonzero((np.diff(xrllmn) / xrllmn[:-1]))[0]) + np.abs(xrllmn).max()

        xtr.loc[seg_id, 'WindowsPartition4.21' + str(win)] = sigmoid(xrllmn.std())

        xtr.loc[seg_id, 'WindowsPartition4.22' + str(win)] = sigmoid(xrllmn.mean())

        xtr.loc[seg_id, 'WindowsPartition4.23' + str(win)] = sigmoid(xrllmn.std()) - sigmoid(xrllmn.mean())

        xtr.loc[seg_id, 'WindowsPartition4.24' + str(win)] = sigmoid(xrllmn.std()) + sigmoid(xrllmn.mean())

        xtr.loc[seg_id, 'WindowsPartition4.25' + str(win)] = cost(-2.555912182174505, xrllmn.std(), sd, 0.33)

        xtr.loc[seg_id, 'WindowsPartition4.26' + str(win)] = cost(-2.555912182174505, xrllmn.mean(), sd, 0.33)

        xtr.loc[seg_id, 'WindowsPartition4.27' + str(win)] = cost(-2.555912182174505, xrllmn.std(), sd, 0.25)

        xtr.loc[seg_id, 'WindowsPartition4.28' + str(win)] = cost(-2.555912182174505, xrllmn.mean(), mn, 0.25)

        xtr.loc[seg_id, 'WindowsPartition4.29' + str(win)] = cost(-1.8537597012091438, xrllmn.std(), sd, 0.33)

        xtr.loc[seg_id, 'WindowsPartition4.30' + str(win)] = cost(-1.8537597012091438, xrllmn.mean(), mn, 0.33)

        xtr.loc[seg_id, 'WindowsPartition4.31' + str(win)] = cost(-1.8537597012091438, xrllmn.std(), sd, 0.25)

        xtr.loc[seg_id, 'WindowsPartition4.32' + str(win)] = cost(-1.8537597012091438, xrllmn.mean(), mn, 0.25)

        xtr.loc[seg_id, 'WindowsPartition4.33' + str(win)] = EvalutedHypothesis(-6.53268708298804, xrllmn.std(), sd, 0.33)

        xtr.loc[seg_id, 'WindowsPartition4.34' + str(win)] = EvalutedHypothesis(-6.53268708298804, xrllmn.std(), sd, 0.25)

        xtr.loc[seg_id, 'WindowsPartition4.35' + str(win)] = EvalutedHypothesis(-6.53268708298804, xrllmn.mean(), mn, 0.33)

        xtr.loc[seg_id, 'WindowsPartition.36' + str(win)] = EvalutedHypothesis(-6.53268708298804, xrllmn.mean(), mn, 0.25)

        xtr.loc[seg_id, 'WindowsPartition4.37' + str(win)] = EvalutedHypothesis(-3.436425029827014, xrllmn.std(), sd, 0.33)

        xtr.loc[seg_id, 'WindowsPartition4.38' + str(win)] = EvalutedHypothesis(-3.436425029827014, xrllmn.std(), sd, 0.25)

        xtr.loc[seg_id, 'WindowsPartition4.39' + str(win)] = EvalutedHypothesis(-3.436425029827014, xrllmn.mean(), mn, 0.33)

        xtr.loc[seg_id, 'WindowsPartition4.40' + str(win)] = EvalutedHypothesis(-3.436425029827014, xrllmn.mean(), mn, 0.25)

        xtr.loc[seg_id, 'WindowsPartition4.41' + str(win)] = EvalutedHypothesis(-6.53268708298804, xrllmn.std(), sd, 0.33) - sigmoid(xrllmn.std())

        xtr.loc[seg_id, 'WindowsPartition4.42' + str(win)] = EvalutedHypothesis(-3.436425029827014, xrllmn.std(), sd, 0.33) - sigmoid(xrllmn.std())

        xtr.loc[seg_id, 'WindowsPartition4.43' + str(win)] = EvalutedHypothesis(-6.53268708298804, xrllmn.mean(), mn, 0.33) - sigmoid(xrllmn.mean())

        xtr.loc[seg_id, 'WindowsPartition4.44' + str(win)] = EvalutedHypothesis(-3.436425029827014, xrllmn.mean(), mn, 0.33) - sigmoid(xrllmn.mean())

        xtr.loc[seg_id, 'WindowsPartition4.45' + str(win)] = EvalutedHypothesis(-6.53268708298804, xrllmn.std(), sd, 0.25) - sigmoid(xrllmn.std())

        xtr.loc[seg_id, 'WindowsPartition4.46' + str(win)] = EvalutedHypothesis(-3.436425029827014, xrllmn.std(), sd, 0.25) - sigmoid(xrllmn.std())

        xtr.loc[seg_id, 'WindowsPartition4.47' + str(win)] = EvalutedHypothesis(-6.53268708298804, xrllmn.mean(), mn, 0.25) - sigmoid(xrllmn.mean())

        xtr.loc[seg_id, 'WindowsPartition4.48' + str(win)] = EvalutedHypothesis(-3.436425029827014, xrllmn.mean(), mn, 0.25) - sigmoid(xrllmn.mean())

        

    for win in [20, 175, 1375] :

        xrllsd = X.rolling(win).std().dropna().values 

        xrllmn = X.rolling(win).std().dropna().values

        

        xtr.loc[seg_id, 'WindowsPartition5.1' + str(win)] = xrllsd.mean()

        xtr.loc[seg_id, 'WindowsPartition5.2' + str(win)] = xrllsd.std()

        xtr.loc[seg_id, 'WindowsPartitio5.3' + str(win)] = xrllsd.max()

        xtr.loc[seg_id, 'WindowsPartition5.4' + str(win)] = xrllsd.min()

        xtr.loc[seg_id, 'WindowsPartition5.5' + str(win)] = np.quantile(xrllsd, 0.0125)

        xtr.loc[seg_id, 'WindowsPartition5.6' + str(win)] = np.quantile(xrllsd, 0.0750)

        xtr.loc[seg_id, 'WindowsPartition5.7' + str(win)] = np.quantile(xrllsd, 0.125)

        xtr.loc[seg_id, 'WindowsPartition5.8' + str(win)] = np.quantile(xrllsd, 0.105)

        xtr.loc[seg_id, 'WindowsPartition5.9' + str(win)] = np.mean(np.diff(xrllsd))

        xtr.loc[seg_id, 'WindowsPartition5.10' + str(win)] = np.mean(np.nonzero((np.diff(xrllsd) / xrllsd[:-1]))[0])

        xtr.loc[seg_id, 'WindowsPartition5.11' + str(win)] = np.abs(xrllsd).max()

        xtr.loc[seg_id, 'WindowsPartition5.12' + str(win)] = xrllsd.mean() - xrllsd.std()

        xtr.loc[seg_id, 'WindowsPartition5.13' + str(win)] = xrllsd.max()  - xrllsd.min()

        xtr.loc[seg_id, 'WindowsPartition5.14' + str(win)] = np.quantile(xrllsd, 0.0250) - np.quantile(xrllsd, 0.0125)

        xtr.loc[seg_id, 'WindowsPartition5.15' + str(win)] = np.quantile(xrllsd, 0.0900) - np.quantile(xrllsd, 0.0750)

        xtr.loc[seg_id, 'WindowsPartition5.16' + str(win)] = np.quantile(xrllsd, 0.250) - np.quantile(xrllsd, 0.125)

        xtr.loc[seg_id, 'WindowsPartition5.17' + str(win)] = np.quantile(xrllsd, 0.210) - np.quantile(xrllsd, 0.105)

        xtr.loc[seg_id, 'WindowsPartition5.18' + str(win)] = np.quantile(xrllsd, 0.2575) 

        xtr.loc[seg_id, 'WindowsPartition5.19' + str(win)] = np.mean(np.nonzero((np.diff(xrllsd) / xrllsd[:-1]))[0]) - np.abs(xrllsd).max()

        xtr.loc[seg_id, 'WindowsPartition5.20' + str(win)] = np.mean(np.nonzero((np.diff(xrllsd) / xrllsd[:-1]))[0]) + np.abs(xrllsd).max()

        xtr.loc[seg_id, 'WindowsPartition5.21' + str(win)] = sigmoid(xrllsd.std())

        xtr.loc[seg_id, 'WindowsPartition5.22' + str(win)] = sigmoid(xrllsd.mean())

        xtr.loc[seg_id, 'WindowsPartition5.23' + str(win)] = sigmoid(xrllsd.std()) - sigmoid(xrllsd.mean())

        xtr.loc[seg_id, 'WindowsPartition5.24' + str(win)] = sigmoid(xrllsd.std()) + sigmoid(xrllsd.mean())

        xtr.loc[seg_id, 'WindowsPartition5.25' + str(win)] = cost(-2.555912182174505, xrllsd.std(), sd, 0.33)

        xtr.loc[seg_id, 'WindowsPartition5.26' + str(win)] = cost(-2.555912182174505, xrllsd.mean(), sd, 0.33)

        xtr.loc[seg_id, 'WindowsPartition5.27' + str(win)] = cost(-2.555912182174505, xrllsd.std(), sd, 0.25)

        xtr.loc[seg_id, 'WindowsPartition5.28' + str(win)] = cost(-2.555912182174505, xrllsd.mean(), mn, 0.25)

        xtr.loc[seg_id, 'WindowsPartition5.29' + str(win)] = cost(-1.8537597012091438, xrllsd.std(), sd, 0.33)

        xtr.loc[seg_id, 'WindowsPartition5.30' + str(win)] = cost(-1.8537597012091438, xrllsd.mean(), mn, 0.33)

        xtr.loc[seg_id, 'WindowsPartition5.31' + str(win)] = cost(-1.8537597012091438, xrllsd.std(), sd, 0.25)

        xtr.loc[seg_id, 'WindowsPartition5.32' + str(win)] = cost(-1.8537597012091438, xrllsd.mean(), mn, 0.25)

        xtr.loc[seg_id, 'WindowsPartition5.33' + str(win)] = EvalutedHypothesis(-6.53268708298804, xrllsd.std(), sd, 0.33)

        xtr.loc[seg_id, 'WindowsPartition.34' + str(win)] = EvalutedHypothesis(-6.53268708298804, xrllsd.std(), sd, 0.25)

        xtr.loc[seg_id, 'WindowsPartition5.35' + str(win)] = EvalutedHypothesis(-6.53268708298804, xrllsd.mean(), mn, 0.33)

        xtr.loc[seg_id, 'WindowsPartition5.36' + str(win)] = EvalutedHypothesis(-6.53268708298804, xrllsd.mean(), mn, 0.25)

        xtr.loc[seg_id, 'WindowsPartition5.37' + str(win)] = EvalutedHypothesis(-3.436425029827014, xrllsd.std(), sd, 0.33)

        xtr.loc[seg_id, 'WindowsPartition5.38' + str(win)] = EvalutedHypothesis(-3.436425029827014, xrllsd.std(), sd, 0.25)

        xtr.loc[seg_id, 'WindowsPartition5.39' + str(win)] = EvalutedHypothesis(-3.436425029827014, xrllsd.mean(), mn, 0.33)

        xtr.loc[seg_id, 'WindowsPartition5.40' + str(win)] = EvalutedHypothesis(-3.436425029827014, xrllsd.mean(), mn, 0.25)

        xtr.loc[seg_id, 'WindowsPartition5.41' + str(win)] = EvalutedHypothesis(-6.53268708298804, xrllsd.std(), sd, 0.33) - sigmoid(xrllsd.std())

        xtr.loc[seg_id, 'WindowsPartition5.42' + str(win)] = EvalutedHypothesis(-3.436425029827014, xrllsd.std(), sd, 0.33) - sigmoid(xrllsd.std())

        xtr.loc[seg_id, 'WindowsPartition5.43' + str(win)] = EvalutedHypothesis(-6.53268708298804, xrllsd.mean(), mn, 0.33) - sigmoid(xrllsd.mean())

        xtr.loc[seg_id, 'WindowsPartition5.44' + str(win)] = EvalutedHypothesis(-3.436425029827014, xrllsd.mean(), mn, 0.33) - sigmoid(xrllsd.mean())

        xtr.loc[seg_id, 'WindowsPartition5.45' + str(win)] = EvalutedHypothesis(-6.53268708298804, xrllsd.std(), sd, 0.25) - sigmoid(xrllsd.std())

        xtr.loc[seg_id, 'WindowsPartition5.46' + str(win)] = EvalutedHypothesis(-3.436425029827014, xrllsd.std(), sd, 0.25) - sigmoid(xrllsd.std())

        xtr.loc[seg_id, 'WindowsPartition5.47' + str(win)] = EvalutedHypothesis(-6.53268708298804, xrllsd.mean(), mn, 0.25) - sigmoid(xrllsd.mean())

        xtr.loc[seg_id, 'WindowsPartition5.48' + str(win)] = EvalutedHypothesis(-3.436425029827014, xrllsd.mean(), mn, 0.25) - sigmoid(xrllsd.mean())

        

        xtr.loc[seg_id, 'WindowsPartition6.1' + str(win)] = xrllmn.mean()

        xtr.loc[seg_id, 'WindowsPartition6.2' + str(win)] = xrllmn.std()

        xtr.loc[seg_id, 'WindowsPartition6.3' + str(win)] = xrllmn.max()

        xtr.loc[seg_id, 'WindowsPartition6.4' + str(win)] = xrllmn.min()

        xtr.loc[seg_id, 'WindowsPartition6.5' + str(win)] = np.quantile(xrllmn, 0.0125)

        xtr.loc[seg_id, 'WindowsPartition6.6' + str(win)] = np.quantile(xrllmn, 0.0750)

        xtr.loc[seg_id, 'WindowsPartition6.7' + str(win)] = np.quantile(xrllmn, 0.125)

        xtr.loc[seg_id, 'WindowsPartition6.8' + str(win)] = np.quantile(xrllmn, 0.105)

        xtr.loc[seg_id, 'WindowsPartition6.9' + str(win)] = np.mean(np.diff(xrllsd))

        xtr.loc[seg_id, 'WindowsPartition6.10' + str(win)] = np.mean(np.nonzero((np.diff(xrllmn) / xrllmn[:-1]))[0])

        xtr.loc[seg_id, 'WindowsPartition6.11' + str(win)] = np.abs(xrllmn).max()

        xtr.loc[seg_id, 'WindowsPartition6.12' + str(win)] = xrllmn.mean() - xrllmn.std()

        xtr.loc[seg_id, 'WindowsPartition6.13' + str(win)] = xrllmn.max()  - xrllmn.min()

        xtr.loc[seg_id, 'WindowsPartition6.14' + str(win)] = np.quantile(xrllmn, 0.0250) - np.quantile(xrllmn, 0.0125)

        xtr.loc[seg_id, 'WindowsPartition6.15' + str(win)] = np.quantile(xrllmn, 0.0900) - np.quantile(xrllmn, 0.0750)

        xtr.loc[seg_id, 'WindowsPartition6.16' + str(win)] = np.quantile(xrllmn, 0.250) - np.quantile(xrllmn, 0.125)

        xtr.loc[seg_id, 'WindowsPartition6.17' + str(win)] =  np.quantile(xrllmn, 0.210) - np.quantile(xrllmn, 0.105)

        xtr.loc[seg_id, 'WindowsPartition6.18' + str(win)] = np.quantile(xrllmn, 0.2575)

        xtr.loc[seg_id, 'WindowsPartition6.19' + str(win)] = np.mean(np.nonzero((np.diff(xrllmn) / xrllmn[:-1]))[0]) - np.abs(xrllmn).max()

        xtr.loc[seg_id, 'WindowsPartition6.20' + str(win)] = np.mean(np.nonzero((np.diff(xrllmn) / xrllmn[:-1]))[0]) + np.abs(xrllmn).max()

        xtr.loc[seg_id, 'WindowsPartition6.21' + str(win)] = sigmoid(xrllmn.std())

        xtr.loc[seg_id, 'WindowsPartition6.22' + str(win)] = sigmoid(xrllmn.mean())

        xtr.loc[seg_id, 'WindowsPartition6.23' + str(win)] = sigmoid(xrllmn.std()) - sigmoid(xrllmn.mean())

        xtr.loc[seg_id, 'WindowsPartition6.24' + str(win)] = sigmoid(xrllmn.std()) + sigmoid(xrllmn.mean())

        xtr.loc[seg_id, 'WindowsPartition6.25' + str(win)] = cost(-2.555912182174505, xrllmn.std(), sd, 0.33)

        xtr.loc[seg_id, 'WindowsPartition6.26' + str(win)] = cost(-2.555912182174505, xrllmn.mean(), sd, 0.33)

        xtr.loc[seg_id, 'WindowsPartition6.27' + str(win)] = cost(-2.555912182174505, xrllmn.std(), sd, 0.25)

        xtr.loc[seg_id, 'WindowsPartition6.28' + str(win)] = cost(-2.555912182174505, xrllmn.mean(), mn, 0.25)

        xtr.loc[seg_id, 'WindowsPartition6.29' + str(win)] = cost(-1.8537597012091438, xrllmn.std(), sd, 0.33)

        xtr.loc[seg_id, 'WindowsPartition6.30' + str(win)] = cost(-1.8537597012091438, xrllmn.mean(), mn, 0.33)

        xtr.loc[seg_id, 'WindowsPartition6.31' + str(win)] = cost(-1.8537597012091438, xrllmn.std(), sd, 0.25)

        xtr.loc[seg_id, 'WindowsPartition6.32' + str(win)] = cost(-1.8537597012091438, xrllmn.mean(), mn, 0.25)

        xtr.loc[seg_id, 'WindowsPartition6.33' + str(win)] = EvalutedHypothesis(-6.53268708298804, xrllmn.std(), sd, 0.33)

        xtr.loc[seg_id, 'WindowsPartition6.34' + str(win)] = EvalutedHypothesis(-6.53268708298804, xrllmn.std(), sd, 0.25)

        xtr.loc[seg_id, 'WindowsPartition6.35' + str(win)] = EvalutedHypothesis(-6.53268708298804, xrllmn.mean(), mn, 0.33)

        xtr.loc[seg_id, 'WindowsPartition6.36' + str(win)] = EvalutedHypothesis(-6.53268708298804, xrllmn.mean(), mn, 0.25)

        xtr.loc[seg_id, 'WindowsPartition6.37' + str(win)] = EvalutedHypothesis(-3.436425029827014, xrllmn.std(), sd, 0.33)

        xtr.loc[seg_id, 'WindowsPartition6.38' + str(win)] = EvalutedHypothesis(-3.436425029827014, xrllmn.std(), sd, 0.25)

        xtr.loc[seg_id, 'WindowsPartition6.39' + str(win)] = EvalutedHypothesis(-3.436425029827014, xrllmn.mean(), mn, 0.33)

        xtr.loc[seg_id, 'WindowsPartition6.40' + str(win)] = EvalutedHypothesis(-3.436425029827014, xrllmn.mean(), mn, 0.25)

        xtr.loc[seg_id, 'WindowsPartition6.41' + str(win)] = EvalutedHypothesis(-6.53268708298804, xrllmn.std(), sd, 0.33) - sigmoid(xrllmn.std())

        xtr.loc[seg_id, 'WindowsPartition6.42' + str(win)] = EvalutedHypothesis(-3.436425029827014, xrllmn.std(), sd, 0.33) - sigmoid(xrllmn.std())

        xtr.loc[seg_id, 'WindowsPartition6.43' + str(win)] = EvalutedHypothesis(-6.53268708298804, xrllmn.mean(), mn, 0.33) - sigmoid(xrllmn.mean())

        xtr.loc[seg_id, 'WindowsPartition6.44' + str(win)] = EvalutedHypothesis(-3.436425029827014, xrllmn.mean(), mn, 0.33) - sigmoid(xrllmn.mean())

        xtr.loc[seg_id, 'WindowsPartition6.45' + str(win)] = EvalutedHypothesis(-6.53268708298804, xrllmn.std(), sd, 0.25) - sigmoid(xrllmn.std())

        xtr.loc[seg_id, 'WindowsPartition6.46' + str(win)] = EvalutedHypothesis(-3.436425029827014, xrllmn.std(), sd, 0.25) - sigmoid(xrllmn.std())

        xtr.loc[seg_id, 'WindowsPartition6.47' + str(win)] = EvalutedHypothesis(-6.53268708298804, xrllmn.mean(), mn, 0.25) - sigmoid(xrllmn.mean())

        xtr.loc[seg_id, 'WindowsPartition6.48' + str(win)] = EvalutedHypothesis(-3.436425029827014, xrllmn.mean(), mn, 0.25) - sigmoid(xrllmn.mean())

        

    for win in [25, 200, 1500] :

        xrllsd = X.rolling(win).std().dropna().values 

        xrllmn = X.rolling(win).std().dropna().values

        

        xtr.loc[seg_id, 'WindowsPartition7.1' + str(win)] = xrllsd.mean()

        xtr.loc[seg_id, 'WindowsPartition7.2' + str(win)] = xrllsd.std()

        xtr.loc[seg_id, 'WindowsPartition7.3' + str(win)] = xrllsd.max()

        xtr.loc[seg_id, 'WindowsPartition7.4' + str(win)] = xrllsd.min()

        xtr.loc[seg_id, 'WindowsPartition7.5' + str(win)] = np.quantile(xrllsd, 0.0125)

        xtr.loc[seg_id, 'WindowsPartition7.6' + str(win)] = np.quantile(xrllsd, 0.0750)

        xtr.loc[seg_id, 'WindowsPartition7.7' + str(win)] = np.quantile(xrllsd, 0.125)

        xtr.loc[seg_id, 'WindowsPartition7.8' + str(win)] = np.quantile(xrllsd, 0.105)

        xtr.loc[seg_id, 'WindowsPartition7.9' + str(win)] = np.mean(np.diff(xrllsd))

        xtr.loc[seg_id, 'WindowsPartition7.10' + str(win)] = np.mean(np.nonzero((np.diff(xrllsd) / xrllsd[:-1]))[0])

        xtr.loc[seg_id, 'WindowsPartition7.11' + str(win)] = np.abs(xrllsd).max()

        xtr.loc[seg_id, 'WindowsPartition7.12' + str(win)] = xrllsd.mean() - xrllsd.std()

        xtr.loc[seg_id, 'WindowsPartition7.13' + str(win)] = xrllsd.max()  - xrllsd.min()

        xtr.loc[seg_id, 'WindowsPartition7.14' + str(win)] = np.quantile(xrllsd, 0.0250) - np.quantile(xrllsd, 0.0125)

        xtr.loc[seg_id, 'WindowsPartition7.15' + str(win)] = np.quantile(xrllsd, 0.0900) - np.quantile(xrllsd, 0.0750)

        xtr.loc[seg_id, 'WindowsPartition7.16' + str(win)] = np.quantile(xrllsd, 0.250) - np.quantile(xrllsd, 0.125)

        xtr.loc[seg_id, 'WindowsPartition7.17' + str(win)] = np.quantile(xrllsd, 0.210) - np.quantile(xrllsd, 0.105)

        xtr.loc[seg_id, 'WindowsPartition7.18' + str(win)] = np.quantile(xrllsd, 0.2575) 

        xtr.loc[seg_id, 'WindowsPartition7.19' + str(win)] = np.mean(np.nonzero((np.diff(xrllsd) / xrllsd[:-1]))[0]) - np.abs(xrllsd).max()

        xtr.loc[seg_id, 'WindowsPartition7.20' + str(win)] = np.mean(np.nonzero((np.diff(xrllsd) / xrllsd[:-1]))[0]) + np.abs(xrllsd).max()

        xtr.loc[seg_id, 'WindowsPartition7.21' + str(win)] = sigmoid(xrllsd.std())

        xtr.loc[seg_id, 'WindowsPartition7.22' + str(win)] = sigmoid(xrllsd.mean())

        xtr.loc[seg_id, 'WindowsPartition7.23' + str(win)] = sigmoid(xrllsd.std()) - sigmoid(xrllsd.mean())

        xtr.loc[seg_id, 'WindowsPartition7.24' + str(win)] = sigmoid(xrllsd.std()) + sigmoid(xrllsd.mean())

        xtr.loc[seg_id, 'WindowsPartition7.25' + str(win)] = cost(-2.555912182174505, xrllsd.std(), sd, 0.33)

        xtr.loc[seg_id, 'WindowsPartition7.26' + str(win)] = cost(-2.555912182174505, xrllsd.mean(), sd, 0.33)

        xtr.loc[seg_id, 'WindowsPartition7.27' + str(win)] = cost(-2.555912182174505, xrllsd.std(), sd, 0.25)

        xtr.loc[seg_id, 'WindowsPartition7.28' + str(win)] = cost(-2.555912182174505, xrllsd.mean(), mn, 0.25)

        xtr.loc[seg_id, 'WindowsPartition7.29' + str(win)] = cost(-1.8537597012091438, xrllsd.std(), sd, 0.33)

        xtr.loc[seg_id, 'WindowsPartition7.30' + str(win)] = cost(-1.8537597012091438, xrllsd.mean(), mn, 0.33)

        xtr.loc[seg_id, 'WindowsPartition7.31' + str(win)] = cost(-1.8537597012091438, xrllsd.std(), sd, 0.25)

        xtr.loc[seg_id, 'WindowsPartition7.32' + str(win)] = cost(-1.8537597012091438, xrllsd.mean(), mn, 0.25)

        xtr.loc[seg_id, 'WindowsPartition7.33' + str(win)] = EvalutedHypothesis(-6.53268708298804, xrllsd.std(), sd, 0.33)

        xtr.loc[seg_id, 'WindowsPartition7.34' + str(win)] = EvalutedHypothesis(-6.53268708298804, xrllsd.std(), sd, 0.25)

        xtr.loc[seg_id, 'WindowsPartition7.35' + str(win)] = EvalutedHypothesis(-6.53268708298804, xrllsd.mean(), mn, 0.33)

        xtr.loc[seg_id, 'WindowsPartition7.36' + str(win)] = EvalutedHypothesis(-6.53268708298804, xrllsd.mean(), mn, 0.25)

        xtr.loc[seg_id, 'WindowsPartition7.37' + str(win)] = EvalutedHypothesis(-3.436425029827014, xrllsd.std(), sd, 0.33)

        xtr.loc[seg_id, 'WindowsPartition7.38' + str(win)] = EvalutedHypothesis(-3.436425029827014, xrllsd.std(), sd, 0.25)

        xtr.loc[seg_id, 'WindowsPartition7.39' + str(win)] = EvalutedHypothesis(-3.436425029827014, xrllsd.mean(), mn, 0.33)

        xtr.loc[seg_id, 'WindowsPartition7.40' + str(win)] = EvalutedHypothesis(-3.436425029827014, xrllsd.mean(), mn, 0.25)

        xtr.loc[seg_id, 'WindowsPartition7.41' + str(win)] = EvalutedHypothesis(-6.53268708298804, xrllsd.std(), sd, 0.33) - sigmoid(xrllsd.std())

        xtr.loc[seg_id, 'WindowsPartition7.42' + str(win)] = EvalutedHypothesis(-3.436425029827014, xrllsd.std(), sd, 0.33) - sigmoid(xrllsd.std())

        xtr.loc[seg_id, 'WindowsPartition7.43' + str(win)] = EvalutedHypothesis(-6.53268708298804, xrllsd.mean(), mn, 0.33) - sigmoid(xrllsd.mean())

        xtr.loc[seg_id, 'WindowsPartition7.44' + str(win)] = EvalutedHypothesis(-3.436425029827014, xrllsd.mean(), mn, 0.33) - sigmoid(xrllsd.mean())

        xtr.loc[seg_id, 'WindowsPartition7.45' + str(win)] = EvalutedHypothesis(-6.53268708298804, xrllsd.std(), sd, 0.25) - sigmoid(xrllsd.std())

        xtr.loc[seg_id, 'WindowsPartition7.46' + str(win)] = EvalutedHypothesis(-3.436425029827014, xrllsd.std(), sd, 0.25) - sigmoid(xrllsd.std())

        xtr.loc[seg_id, 'WindowsPartition7.47' + str(win)] = EvalutedHypothesis(-6.53268708298804, xrllsd.mean(), mn, 0.25) - sigmoid(xrllsd.mean())

        xtr.loc[seg_id, 'WindowsPartition7.48' + str(win)] = EvalutedHypothesis(-3.436425029827014, xrllsd.mean(), mn, 0.25) - sigmoid(xrllsd.mean())

        

        xtr.loc[seg_id, 'WindowsPartition8.1' + str(win)] = xrllmn.mean()

        xtr.loc[seg_id, 'WindowsPartition8.2' + str(win)] = xrllmn.std()

        xtr.loc[seg_id, 'WindowsPartition8.3' + str(win)] = xrllmn.max()

        xtr.loc[seg_id, 'WindowsPartition8.4' + str(win)] = xrllmn.min()

        xtr.loc[seg_id, 'WindowsPartition8.5' + str(win)] = np.quantile(xrllmn, 0.0125)

        xtr.loc[seg_id, 'WindowsPartition8.6' + str(win)] = np.quantile(xrllmn, 0.0750)

        xtr.loc[seg_id, 'WindowsPartition8.7' + str(win)] = np.quantile(xrllmn, 0.125)

        xtr.loc[seg_id, 'WindowsPartition8.8' + str(win)] = np.quantile(xrllmn, 0.105)

        xtr.loc[seg_id, 'WindowsPartition8.9' + str(win)] = np.mean(np.diff(xrllsd))

        xtr.loc[seg_id, 'WindowsPartition8.10' + str(win)] = np.mean(np.nonzero((np.diff(xrllmn) / xrllmn[:-1]))[0])

        xtr.loc[seg_id, 'WindowsPartition8.11' + str(win)] = np.abs(xrllmn).max()

        xtr.loc[seg_id, 'WindowsPartition8.12' + str(win)] = xrllmn.mean() - xrllmn.std()

        xtr.loc[seg_id, 'WindowsPartition8.13' + str(win)] = xrllmn.max()  - xrllmn.min()

        xtr.loc[seg_id, 'WindowsPartition8.14' + str(win)] = np.quantile(xrllmn, 0.0250) - np.quantile(xrllmn, 0.0125)

        xtr.loc[seg_id, 'WindowsPartition8.15' + str(win)] = np.quantile(xrllmn, 0.0900) - np.quantile(xrllmn, 0.0750)

        xtr.loc[seg_id, 'WindowsPartition8.16' + str(win)] = np.quantile(xrllmn, 0.250) - np.quantile(xrllmn, 0.125)

        xtr.loc[seg_id, 'WindowsPartition8.17' + str(win)] =  np.quantile(xrllmn, 0.210) - np.quantile(xrllmn, 0.105)

        xtr.loc[seg_id, 'WindowsPartition8.18' + str(win)] = np.quantile(xrllmn, 0.2575)

        xtr.loc[seg_id, 'WindowsPartition8.19' + str(win)] = np.mean(np.nonzero((np.diff(xrllmn) / xrllmn[:-1]))[0]) - np.abs(xrllmn).max()

        xtr.loc[seg_id, 'WindowsPartition8.20' + str(win)] = np.mean(np.nonzero((np.diff(xrllmn) / xrllmn[:-1]))[0]) + np.abs(xrllmn).max()

        xtr.loc[seg_id, 'WindowsPartition8.21' + str(win)] = sigmoid(xrllmn.std())

        xtr.loc[seg_id, 'WindowsPartition8.22' + str(win)] = sigmoid(xrllmn.mean())

        xtr.loc[seg_id, 'WindowsPartition8.23' + str(win)] = sigmoid(xrllmn.std()) - sigmoid(xrllmn.mean())

        xtr.loc[seg_id, 'WindowsPartition8.24' + str(win)] = sigmoid(xrllmn.std()) + sigmoid(xrllmn.mean())

        xtr.loc[seg_id, 'WindowsPartition8.25' + str(win)] = cost(-2.555912182174505, xrllmn.std(), sd, 0.33)

        xtr.loc[seg_id, 'WindowsPartition8.26' + str(win)] = cost(-2.555912182174505, xrllmn.mean(), sd, 0.33)

        xtr.loc[seg_id, 'WindowsPartition8.27' + str(win)] = cost(-2.555912182174505, xrllmn.std(), sd, 0.25)

        xtr.loc[seg_id, 'WindowsPartition8.28' + str(win)] = cost(-2.555912182174505, xrllmn.mean(), mn, 0.25)

        xtr.loc[seg_id, 'WindowsPartition8.29' + str(win)] = cost(-1.8537597012091438, xrllmn.std(), sd, 0.33)

        xtr.loc[seg_id, 'WindowsPartition8.30' + str(win)] = cost(-1.8537597012091438, xrllmn.mean(), mn, 0.33)

        xtr.loc[seg_id, 'WindowsPartition8.31' + str(win)] = cost(-1.8537597012091438, xrllmn.std(), sd, 0.25)

        xtr.loc[seg_id, 'WindowsPartition8.32' + str(win)] = cost(-1.8537597012091438, xrllmn.mean(), mn, 0.25)

        xtr.loc[seg_id, 'WindowsPartition8.33' + str(win)] = EvalutedHypothesis(-6.53268708298804, xrllmn.std(), sd, 0.33)

        xtr.loc[seg_id, 'WindowsPartition8.34' + str(win)] = EvalutedHypothesis(-6.53268708298804, xrllmn.std(), sd, 0.25)

        xtr.loc[seg_id, 'WindowsPartition8.35' + str(win)] = EvalutedHypothesis(-6.53268708298804, xrllmn.mean(), mn, 0.33)

        xtr.loc[seg_id, 'WindowsPartition8.36' + str(win)] = EvalutedHypothesis(-6.53268708298804, xrllmn.mean(), mn, 0.25)

        xtr.loc[seg_id, 'WindowsPartition8.37' + str(win)] = EvalutedHypothesis(-3.436425029827014, xrllmn.std(), sd, 0.33)

        xtr.loc[seg_id, 'WindowsPartition8.38' + str(win)] = EvalutedHypothesis(-3.436425029827014, xrllmn.std(), sd, 0.25)

        xtr.loc[seg_id, 'WindowsPartition8.39' + str(win)] = EvalutedHypothesis(-3.436425029827014, xrllmn.mean(), mn, 0.33)

        xtr.loc[seg_id, 'WindowsPartition8.40' + str(win)] = EvalutedHypothesis(-3.436425029827014, xrllmn.mean(), mn, 0.25)

        xtr.loc[seg_id, 'WindowsPartition8.41' + str(win)] = EvalutedHypothesis(-6.53268708298804, xrllmn.std(), sd, 0.33) - sigmoid(xrllmn.std())

        xtr.loc[seg_id, 'WindowsPartition8.42' + str(win)] = EvalutedHypothesis(-3.436425029827014, xrllmn.std(), sd, 0.33) - sigmoid(xrllmn.std())

        xtr.loc[seg_id, 'WindowsPartition8.43' + str(win)] = EvalutedHypothesis(-6.53268708298804, xrllmn.mean(), mn, 0.33) - sigmoid(xrllmn.mean())

        xtr.loc[seg_id, 'WindowsPartition8.44' + str(win)] = EvalutedHypothesis(-3.436425029827014, xrllmn.mean(), mn, 0.33) - sigmoid(xrllmn.mean())

        xtr.loc[seg_id, 'WindowsPartition8.45' + str(win)] = EvalutedHypothesis(-6.53268708298804, xrllmn.std(), sd, 0.25) - sigmoid(xrllmn.std())

        xtr.loc[seg_id, 'WindowsPartition8.46' + str(win)] = EvalutedHypothesis(-3.436425029827014, xrllmn.std(), sd, 0.25) - sigmoid(xrllmn.std())

        xtr.loc[seg_id, 'WindowsPartition8.47' + str(win)] = EvalutedHypothesis(-6.53268708298804, xrllmn.mean(), mn, 0.25) - sigmoid(xrllmn.mean())

        xtr.loc[seg_id, 'WindowsPartition8.48' + str(win)] = EvalutedHypothesis(-3.436425029827014, xrllmn.mean(), mn, 0.25) - sigmoid(xrllmn.mean())
# iterate over all segments

for seg_id in tqdm_notebook(range(sgt)):

    seg = train.iloc[seg_id * rows : seg_id * rows + rows]

    create_features(seg_id, seg, xtr)

    ytr.loc[seg_id, 'time_to_failure'] = seg['time_to_failure'].values[-1]
xtr.shape
np.abs(xtr.corrwith(ytr['time_to_failure'])).sort_values(ascending=False).head(12)
scl = StandardScaler()

scl.fit(xtr)

xtsc = pd.DataFrame(scl.transform(xtr), columns = xtr.columns)
lm = LinearRegression()

lm.fit(xtsc, ytr)



lmpr = lm.predict(xtsc)

s = mean_absolute_error(ytr, lmpr)

print(s)



xgbm = xgbr()

xgbm.fit(xtsc, ytr)



xgbpr = xgbm.predict(xtsc)

xgbs = mean_absolute_error(ytr, xgbpr)

print(xgbs)



laso = lss(alpha = 0.1)

laso.fit(xtsc, ytr)





lspr = laso.predict(xtsc)

lsss = mean_absolute_error(ytr, lspr)

print(lsss)



rdgm = rdg()

rdgm.fit(xtsc, ytr)



rdgr = rdgm.predict(xtsc)

rdgs = mean_absolute_error(ytr, rdgr)

print(rdgs)



ecnm = ecn()

ecnm.fit(xtsc, ytr)



ecnpr = ecnm.predict(xtsc)

ecnms = mean_absolute_error(ytr, ecnpr)

print(ecnms)



adrm = adr()

adrm.fit(xtsc, ytr)



adrpr = adrm.predict(xtsc)

adrms = mean_absolute_error(ytr, adrpr)

print(adrms)
sub = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')

xtst = pd.DataFrame(columns = xtr.columns, dtype = np.float64, index = sub.index)
sub.shape, xtst.shape
for seg_id in tqdm_notebook(xtst.index):

    seg = pd.read_csv('../input/test/' + seg_id + '.csv')

    create_features(seg_id, seg, xtst)
stx = pd.DataFrame(scl.transform(xtst), columns = xtst.columns)
stx.shape
combo = pd.concat([xtsc, stx])
nfd = 5

fds = KFold(n_splits = nfd, shuffle = True, random_state = 33)

tnc = stx.columns.values
prms = {'num_leaves': 51,

         'min_data_in_leaf': 10, 

         'objective':'regression',

         'max_depth': -1,

         'learning_rate': 0.001,

         "boosting": "gbdt",

         "feature_fraction": 0.91,

         "bagging_freq": 1,

         "bagging_fraction": 0.91,

         "bagging_seed": 42,

         "metric": 'mae',

         "lambda_l1": 0.1,

         "verbosity": -1,

         "nthread": -1,

         "random_state": 33}
oof = np.zeros(len(xtsc))

pred = np.zeros(len(stx))

feimpf = pd.DataFrame()

#run model

for fold_, (tidx, vidx) in enumerate(fds.split(xtsc,ytr.values)):

    strLog = "fold {}".format(fold_)

    print(strLog)

    

    xtrf, xvl = xtsc.iloc[tidx], xtsc.iloc[vidx]

    ytrf, yvl = ytr.iloc[tidx], ytr.iloc[vidx]



    model = lgb.LGBMRegressor(**prms, n_estimators = 20000, n_jobs = -1)

    model.fit(xtrf, ytrf, 

                    eval_set=[(xtrf, ytrf), (xvl, yvl)], eval_metric = 'mae',

                    verbose = 1000, early_stopping_rounds = 200)

    oof[vidx] = model.predict(xvl, num_iteration = model.best_iteration_)

    #feature importance

    fimpf = pd.DataFrame()

    fimpf["Feature"] = tnc

    fimpf["importance"] = model.feature_importances_[:len(tnc)]

    fimpf["fold"] = fold_ + 1

    feimpf = pd.concat([feimpf, fimpf], axis=0)

    #predictions

    pred += model.predict(stx, num_iteration = model.best_iteration_) / fds.n_splits
def GUIPred(dat) :

    return((((ab.mean() + cd.mean()) / 2) + ef) + (0.4194 * np.tanh(((dat["part20.0"]) - ef))) - (0.2624

           * np.sinh(((dat["part7.5"]) - ef)) + (0.2624 * np.cosh(((dat["part7.0"]) - ef)))) + (0.4194 * 

           np.tanh(((dat["part10.0"]) - ef)) - np.tanh(0.4194 - ((dat["part10.1"]) - ef))) + (0.2624 - 

           np.sinh(ef) + np.cosh(ef) - cd.std()) - ((0.2624 - np.sinh(ef) + np.cosh(ef) - ab.std())) - 

           sigmoid(0.4194 - 0.2624) + ((zz.std()) + zz.mean() / (zz.min() + zz.max())) - ((sigmoid(0.4194) * 

           (np.tanh(dat["part22.0"] + (dat["part22.1"] - dat["part22.2"]))) - ef) / pred.mean()) - 

           ((mppr1.std() + mppr1.mean()) - (mppr2.std() + mppr2.mean()) - (mppr3.std() + mppr3.mean()) - 

           (mppr4.std() + mppr4.mean()) - (lm.predict(stx).std() + lm.predict(stx).mean()) - 

           (ecnm.predict(stx).std() + ecnm.predict(stx).mean())) / 

           ((4194  * ttas * ttsm * (np.tanh(dat["part15"])) - (ttas * ttsm * 2624 * (np.tanh(dat["part15"])

           )))) + (ef + (pred.std() - pred.mean())))



print(mean_absolute_error(ytr,GUIPred(combo[:xtr.shape[0]])))
sub.time_to_failure = pred

sub.to_csv('sub.csv',index = True)