# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
%%time

import matplotlib.pyplot as plt



from tqdm import tqdm_notebook, tqdm_gui



from catboost import CatBoostRegressor

from sklearn.preprocessing import StandardScaler,LabelEncoder

from sklearn.linear_model import LinearRegression

from sklearn.svm import NuSVR, SVR

from sklearn.metrics import mean_absolute_error

from sklearn.kernel_ridge import KernelRidge

from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold

pd.options.display.precision = 15



%matplotlib inline

import lightgbm as lgb

import xgboost as xgb

import time

import datetime

import gc

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")



from scipy.signal import hilbert

from scipy.signal import hann

from scipy.signal import convolve

from scipy import stats



import tsfresh

from tsfresh import extract_features

from tsfresh import select_features

from tsfresh.utilities.dataframe_functions import impute



print('MODULES IMPORTED')


def add_trend_feature(arr, abs_values=False):

    idx = np.array(range(len(arr)))

    if abs_values:

        arr = np.abs(arr)

    lr = LinearRegression()

    lr.fit(idx.reshape(-1, 1), arr)

    return lr.coef_[0]





def calc_change_rate(x):

    change = (np.diff(x) / x[:-1]).values

    change = change[np.nonzero(change)[0]]

    change = change[~np.isnan(change)]

    change = change[change != -np.inf]

    change = change[change != np.inf]

    return np.mean(change)
from tsfresh.feature_extraction import ComprehensiveFCParameters, EfficientFCParameters, MinimalFCParameters

from tsfresh.feature_extraction.feature_calculators import *

 

features = ['abs_energy','absolute_sum_of_changes','count_above_mean','count_below_mean',

            'first_location_of_maximum', 'first_location_of_minimum','last_location_of_maximum','last_location_of_minimum',

            'longest_strike_above_mean','longest_strike_below_mean',

            'mean_change','mean_abs_change','median','mean_second_derivative_central']

#'autocorrelation','fft_coefficient','index_mass_quantile','number_peaks'



n_peaks = 5

cross_threshold = 0



def row_features(seg,x):

    df_temp=pd.DataFrame(index=[0])

    

    df_temp['sum'] = x.sum()

    df_temp['mean']= x.mean()

    df_temp['std'] =x.std()

    df_temp['var'] =x.var()

    df_temp['max'] =x.max()

    df_temp['min'] =x.min()

    

    

    df_temp['trend'] = add_trend_feature(x)

    df_temp['abs_trend'] = add_trend_feature(x, abs_values=True)

    df_temp['abs_sum'] = np.abs(x).sum()

    df_temp['abs_mean'] = np.abs(x).mean()

    df_temp['abs_std'] = np.abs(x).std()

    

    df_temp['mad'] =x.mad()

    df_temp['kurt'] =x.kurt()

    df_temp['skew'] =x.skew()

    

    df_temp['max_to_min'] = x.max() / np.abs(x.min())

    df_temp['max_to_min_diff'] = x.max() - np.abs(x.min())

    df_temp['count_big'] = len(x[np.abs(x) > 500])

    df_temp['sum'] = x.sum()

    

    df_temp['mean_change_rate_first_50000'] = calc_change_rate(x[:50000])

    df_temp['mean_change_rate_last_50000'] = calc_change_rate(x[-50000:])

    df_temp['mean_change_rate_first_10000'] = calc_change_rate(x[:10000])

    df_temp['mean_change_rate_last_10000'] = calc_change_rate(x[-10000:])

        

    df_temp['q95'] = np.quantile(x, 0.95)

    df_temp['q99'] = np.quantile(x, 0.99)

    df_temp['q05'] = np.quantile(x, 0.05)

    df_temp['q01'] = np.quantile(x, 0.01)

    

    df_temp['abs_q95'] = np.quantile(np.abs(x), 0.95)

    df_temp['abs_q99'] = np.quantile(np.abs(x), 0.99)

    df_temp['abs_q05'] = np.quantile(np.abs(x), 0.05)

    df_temp['abs_q01'] = np.quantile(np.abs(x), 0.01)

    

    df_temp['std_first_50000'] = x[:50000].std()

    df_temp['std_last_50000'] = x[-50000:].std()

    df_temp['std_first_10000'] = x[:10000].std()

    df_temp['std_last_10000'] = x[-10000:].std()

    

    df_temp['avg_first_50000'] = x[:50000].mean()

    df_temp['avg_last_50000'] = x[-50000:].mean()

    df_temp['avg_first_10000'] = x[:10000].mean()

    df_temp['avg_last_10000'] = x[-10000:].mean()

    

    df_temp['min_first_50000'] = x[:50000].min()

    df_temp['min_last_50000'] = x[-50000:].min()

    df_temp['min_first_10000'] = x[:10000].min()

    df_temp['min_last_10000'] = x[-10000:].min()

    

    df_temp['max_first_50000'] = x[:50000].max()

    df_temp['max_last_50000'] = x[-50000:].max()

    df_temp['max_first_10000'] = x[:10000].max()

    df_temp['max_last_10000'] = x[-10000:].max()

    

    z = np.fft.fft(x)

    realFFT = np.real(z)

    imagFFT = np.imag(z)

    

    df_temp['Real_mean'] = realFFT.mean()

    df_temp['Real_std'] = realFFT.std()

    df_temp['Real_max'] = realFFT.max()

    df_temp['Real_min'] = realFFT.min()

    df_temp['Imag_mean'] = imagFFT.mean()

    df_temp['Imag_std'] = imagFFT.std()

    df_temp['Imag_max'] = imagFFT.max()

    df_temp['Imag_min'] = imagFFT.min()

    

    df_temp['Hilbert_mean'] = np.abs(hilbert(x)).mean()

    df_temp['Hann_window_mean'] = (convolve(x, hann(150), mode='same') / sum(hann(150))).mean()

    df_temp['Moving_average_700_mean'] = x.rolling(window=700).mean().mean(skipna=True)

    ewma = pd.Series.ewm

    df_temp['exp_Moving_average_300_mean'] = (ewma(x, span=300).mean()).mean(skipna=True)

    df_temp['exp_Moving_average_3000_mean'] = ewma(x, span=3000).mean().mean(skipna=True)

    df_temp['exp_Moving_average_30000_mean'] = ewma(x, span=30000).mean().mean(skipna=True)

    no_of_std = 3

    df_temp['MA_700MA_std_mean'] = x.rolling(window=700).std().mean()

    df_temp['MA_700MA_BB_high_mean'] = (df_temp['Moving_average_700_mean'] + no_of_std * df_temp['MA_700MA_std_mean']).mean()

    df_temp['MA_700MA_BB_low_mean'] = (df_temp['Moving_average_700_mean'] - no_of_std * df_temp['MA_700MA_std_mean']).mean()

    df_temp['MA_400MA_std_mean'] = x.rolling(window=400).std().mean()

    df_temp['MA_400MA_BB_high_mean'] = (df_temp['Moving_average_700_mean'] + no_of_std * df_temp['MA_400MA_std_mean']).mean()

    df_temp['MA_400MA_BB_low_mean'] = (df_temp['Moving_average_700_mean'] - no_of_std * df_temp['MA_400MA_std_mean']).mean()

    df_temp['MA_1000MA_std_mean'] = x.rolling(window=1000).std().mean()

    df_temp.drop('Moving_average_700_mean', axis=1, inplace=True)

    

    df_temp['iqr'] = np.subtract(*np.percentile(x, [75, 25]))

    df_temp['q999'] = np.quantile(x,0.999)

    df_temp['q001'] = np.quantile(x,0.001)

    df_temp['ave10'] = stats.trim_mean(x, 0.1)



    for windows in [10, 100, 1000]:

        x_roll_std = x.rolling(windows).std().dropna().values

        x_roll_mean = x.rolling(windows).mean().dropna().values

        

        df_temp['ave_roll_std_' + str(windows)] = x_roll_std.mean()

        df_temp['std_roll_std_' + str(windows)] = x_roll_std.std()

        df_temp['max_roll_std_' + str(windows)] = x_roll_std.max()

        df_temp['min_roll_std_' + str(windows)] = x_roll_std.min()

        df_temp['q01_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.01)

        df_temp['q05_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.05)

        df_temp['q95_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.95)

        df_temp['q99_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.99)

        df_temp['av_change_abs_roll_std_' + str(windows)] = np.mean(np.diff(x_roll_std))

        df_temp['av_change_rate_roll_std_' + str(windows)] = np.mean(np.nonzero((np.diff(x_roll_std) / x_roll_std[:-1]))[0])

        df_temp['abs_max_roll_std_' + str(windows)] = np.abs(x_roll_std).max()

        

        df_temp['ave_roll_mean_' + str(windows)] = x_roll_mean.mean()

        df_temp['std_roll_mean_' + str(windows)] = x_roll_mean.std()

        df_temp['max_roll_mean_' + str(windows)] = x_roll_mean.max()

        df_temp['min_roll_mean_' + str(windows)] = x_roll_mean.min()

        df_temp['q01_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.01)

        df_temp['q05_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.05)

        df_temp['q95_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.95)

        df_temp['q99_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.99)

        df_temp['av_change_abs_roll_mean_' + str(windows)] = np.mean(np.diff(x_roll_mean))

        df_temp['av_change_rate_roll_mean_' + str(windows)] = np.mean(np.nonzero((np.diff(x_roll_mean) / x_roll_mean[:-1]))[0])

        df_temp['abs_max_roll_mean_' + str(windows)] = np.abs(x_roll_mean).max()

    

    for feature in features: #Features from tsfresh

        possibles = globals().copy()

        possibles.update(locals())

        method = possibles.get(feature)

        if not method:

             raise NotImplementedError("Method %s not implemented" % feature)

        df_temp[feature] = method(x)

        

    for lag in range (0,150_000,30_000):

        df_temp['autocorrelation_'+str(lag)] = autocorrelation(x,lag)

        

    df_temp['number_peaks'] = number_peaks(x, n_peaks)

    df_temp['number_crossing_m'] = number_crossing_m(x, cross_threshold)

    

    

    #print(df_temp.head())

    return(df_temp)



def feature_creation(pd_frame,type_ft):

    segments = int(np.floor(pd_frame.shape[0] / rows))

    df = pd.DataFrame()

    if type_ft == 'train' :

        y_last = pd.DataFrame()

        for segment in range(segments):

                #print(segment)

                seg = pd_frame.iloc[segment*rows:segment*rows+rows]

                #print("SEG IS ", seg)

                x = pd.Series(seg['acoustic_data'].values)

                y = pd.Series(seg['time_to_failure'].values)

                y_last_temp = pd.Series(seg['time_to_failure'].values[-1])

                #print("X IS :",x )  

                df_temp=row_features(seg,x)

                df=pd.concat([df,df_temp])

                y_last=pd.concat([y_last,pd.DataFrame([y_last_temp])])



        #print(df.head())

        return(df,y_last)

    elif type_ft == 'test':

        y_last = pd.DataFrame()

        for segment in range(segments):

                #print(segment)

                seg = pd_frame.iloc[segment*rows:segment*rows+rows]

                #print("SEG IS ", seg)

                x = pd.Series(seg['acoustic_data'].values)

                #print("X IS :",x )  

                df_temp=row_features(seg,x)

                df=pd.concat([df,df_temp])

        #print(df.head())

        return(df,y_last)
chunk_nb = 6

rows = 150_000

i=0

df_features_tr=pd.DataFrame()

y_train=pd.DataFrame()

for train in tqdm_notebook(pd.read_csv('../input/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32},chunksize=rows*chunk_nb),total=int(4194/chunk_nb)):  

    #print(train.head())

    a = train

    ft_tr,y_last = feature_creation(train,'train')

    #print("FT_TR HEAD IS :",ft_tr)

    print(df_features_tr.shape)

    df_features_tr = pd.concat([df_features_tr,ft_tr])

    y_train = pd.concat([y_train,y_last])

y_train.columns=['time_to_failure']

print('TRAINING IMPORTED')
print(y_train.head())

print(y_train.shape)

print('Before Scaling : \n',df_features_tr.head(7))

scale = StandardScaler()

X_train_scaled = pd.DataFrame(scale.fit_transform(df_features_tr),columns=df_features_tr.columns)

print('After Scaling : \n',X_train_scaled.head())
submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')

X_test = pd.DataFrame(columns=df_features_tr.columns, dtype=np.float64, index=submission.index)

plt.figure(figsize=(22, 16))

df_features_test=pd.DataFrame()



for i, seg_id in enumerate(tqdm_notebook(X_test.index)):

    seg = pd.read_csv('../input/test/' + seg_id + '.csv')

    #x = pd.Series(seg['acoustic_data'].values)

    ft_test,zero = feature_creation(seg,'test')

    df_features_test = pd.concat([df_features_test,ft_test])

print('TEST IMPORTED')
scaler = StandardScaler()

scaler.fit(df_features_test)

X_test_scaled = pd.DataFrame(scaler.transform(df_features_test), columns=df_features_test.columns)
X_train_scaled.to_csv('train_features.csv', index=False)

y_train.to_csv('y_train.csv', index=False)

X_test_scaled.to_csv('test_features.csv', index=False)