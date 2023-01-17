import numpy as np

import pandas as pd

import os



import matplotlib.pyplot as plt

%matplotlib inline

from tqdm import tqdm_notebook

from sklearn.preprocessing import StandardScaler

from sklearn.svm import NuSVR, SVR

from sklearn.metrics import mean_absolute_error

pd.options.display.precision = 15



import lightgbm as lgb

import xgboost as xgb

import time

import datetime

from catboost import CatBoostRegressor

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold

from sklearn.metrics import mean_absolute_error

from sklearn.linear_model import LinearRegression

import gc

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")



from scipy.signal import hilbert

from scipy.signal import hann

from scipy.signal import convolve

from scipy import stats

from sklearn.kernel_ridge import KernelRidge

from itertools import product



from tsfresh.feature_extraction import feature_calculators

from joblib import Parallel, delayed
# Create a training file with simple derived features



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



def calc_change_rate(x):

    change = (np.diff(x) / x[:-1]).values

    change = change[np.nonzero(change)[0]]

    change = change[~np.isnan(change)]

    change = change[change != -np.inf]

    change = change[change != np.inf]

    return np.mean(change)
class FeatureGenerator(object):

    def __init__(self, dtype, n_jobs=1, chunk_size=None):

        self.chunk_size = chunk_size

        self.dtype = dtype

        self.filename = None

        self.n_jobs = n_jobs

        self.test_files = []

        if self.dtype == 'train':

            self.filename = '../input/train.csv'

            self.total_data = int(629145481 / self.chunk_size)

        else:

            submission = pd.read_csv('../input/sample_submission.csv')

            for seg_id in submission.seg_id.values:

                self.test_files.append((seg_id, '../input/test/' + seg_id + '.csv'))

            self.total_data = int(len(submission))



    def read_chunks(self):

        if self.dtype == 'train':

            iter_df = pd.read_csv(self.filename, iterator=True, chunksize=self.chunk_size,

                                  dtype={'acoustic_data': np.float64, 'time_to_failure': np.float64})

            for counter, df in enumerate(iter_df):

                x = np.sign(df.acoustic_data.values)*np.log1p(np.abs(df.acoustic_data.values))

                y = df.time_to_failure.values[-1]

                seg_id = 'train_' + str(counter)

                del df

                yield seg_id, x, y

        else:

            for seg_id, f in self.test_files:

                df = pd.read_csv(f, dtype={'acoustic_data': np.float64})

                x = df.acoustic_data.values[-self.chunk_size:]

                x = np.sign(x)*np.log1p(np.abs(x))

                del df

                yield seg_id, x, -999

    

    def get_features(self, x, y, seg_id):

        """

        Gets three groups of features: from original data and from reald and imaginary parts of FFT.

        """

        

        x = pd.Series(x)

    

        zc = np.fft.fft(x)

        realFFT = pd.Series(np.real(zc))

        imagFFT = pd.Series(np.imag(zc))

        

        main_dict = self.features(x, y, seg_id)

        r_dict = self.features(realFFT, y, seg_id)

        i_dict = self.features(imagFFT, y, seg_id)

        

        for k, v in r_dict.items():

            if k in ['classic_sta_lta2_mean',

                     'exp_Moving_std_3000_mean',

                     'classic_sta_lta1_mean']:

                main_dict[f'fftr_{k}'] = v

                

        for k, v in i_dict.items():

            if k in ['classic_sta_lta2_mean']:

                main_dict[f'ffti_{k}'] = v

        

        return main_dict

        

    

    def features(self, x, y, seg_id):

        feature_dict = dict()

        feature_dict['target'] = y

        feature_dict['seg_id'] = seg_id

        for p in [50]:

            feature_dict[f'abs_percentile_{p}'] = np.percentile(np.abs(x), p)

        for autocorr_lag in [5]:

            feature_dict[f'autocorrelation_{autocorr_lag}'] = feature_calculators.autocorrelation(x, autocorr_lag)

        for p in [95,99]:

            feature_dict[f'binned_entropy_{p}'] = feature_calculators.binned_entropy(x, p)

        

        # calc_change_rate on slices of data

        for slice_length, direction in product([50000], ['first']):

            if direction == 'first':

                feature_dict[f'mean_change_rate_{direction}_{slice_length}'] = calc_change_rate(x[:slice_length])

            elif direction == 'last':

                feature_dict[f'mean_change_rate_{direction}_{slice_length}'] = calc_change_rate(x[-slice_length:])

        for peak in [10]:

            feature_dict[f'num_peaks_{peak}'] = feature_calculators.number_peaks(x, peak)

        

        for p in [20]:

            feature_dict[f'percentile_{p}'] = np.percentile(x, p)

        

        x_roll_std = x.rolling(1000).std().dropna().values

        feature_dict[f'percentile_roll_std_30_window_1000'] = np.percentile(x_roll_std, 30)   

        

        x_roll_std = x.rolling(500).std().dropna().values

        feature_dict[f'percentile_roll_std_75_window_500'] = np.percentile(x_roll_std, 75)   

        feature_dict[f'percentile_roll_std_80_window_500'] = np.percentile(x_roll_std, 80)    

        

        ewma = pd.Series.ewm

        feature_dict[f'exp_Moving_std_3000_mean'] = (ewma(x, span=3000).std(skipna=True)).mean(skipna=True)

        

        feature_dict['classic_sta_lta1_mean'] = classic_sta_lta(x, 500, 10000).mean()

        feature_dict['classic_sta_lta2_mean'] = classic_sta_lta(x, 5000, 100000).mean()

        feature_dict['skew'] = x.skew()    

        return feature_dict



    def generate(self):

        feature_list = []

        res = Parallel(n_jobs=self.n_jobs,

                       backend='threading')(delayed(self.get_features)(x, y, s)

                                            for s, x, y in tqdm_notebook(self.read_chunks(), total=self.total_data))

        for r in res:

            feature_list.append(r)

        return pd.DataFrame(feature_list)
training_fg = FeatureGenerator(dtype='train', n_jobs=4, chunk_size=150000)

training_data = training_fg.generate()

training_data.columns
testing_fg = FeatureGenerator(dtype='test', n_jobs=4, chunk_size=150000)

test_data = testing_fg.generate()
X = training_data.drop(['target', 'seg_id'], axis=1)

X_test = test_data.drop(['target', 'seg_id'], axis=1)

test_segs = test_data.seg_id

y = training_data.target
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

feats = X_test.columns

ss.fit(pd.concat([X[feats],X_test[feats]]))

X[feats] = ss.transform(X[feats])

X_test[feats] = ss.transform(X_test[feats])

X.insert(0,'seg_id',training_data.seg_id.values)

X['time_to_failure'] = y.values

X_test.insert(0,'seg_id',test_segs.values)

def GPI(data):

    return (5.683668 +

            1.0*np.tanh(((((data["abs_percentile_50"]) - (((data["percentile_roll_std_75_window_500"]) * 2.0)))) - (((((((((((data["percentile_roll_std_75_window_500"]) + (data["num_peaks_10"]))) * 2.0)) + (data["abs_percentile_50"]))) * 2.0)) * 2.0)))) +

            1.0*np.tanh(((-1.0) - (((((data["percentile_roll_std_30_window_1000"]) + ((((data["binned_entropy_95"]) + (data["num_peaks_10"]))/2.0)))) + (data["percentile_roll_std_75_window_500"]))))) +

            1.0*np.tanh(((((((((data["abs_percentile_50"]) + (data["binned_entropy_99"]))) * ((((data["percentile_roll_std_75_window_500"]) + (data["num_peaks_10"]))/2.0)))) * ((-1.0*((((data["percentile_roll_std_75_window_500"]) + (data["percentile_roll_std_30_window_1000"])))))))) * 2.0)) +

            1.0*np.tanh((((-1.0*((((((data["fftr_exp_Moving_std_3000_mean"]) * (((1.0) + (((data["percentile_roll_std_30_window_1000"]) * 2.0)))))) / 2.0))))) * ((((data["num_peaks_10"]) + (data["abs_percentile_50"]))/2.0)))) +

            1.0*np.tanh(((((data["exp_Moving_std_3000_mean"]) - (data["percentile_roll_std_75_window_500"]))) + (((data["num_peaks_10"]) * (((data["fftr_classic_sta_lta1_mean"]) - (data["fftr_exp_Moving_std_3000_mean"]))))))))



def GPII(data):

    return (5.683668 +

            1.0*np.tanh((((((((((-1.0*((((data["num_peaks_10"]) + (data["percentile_roll_std_75_window_500"])))))) * 2.0)) - (((data["num_peaks_10"]) + (data["abs_percentile_50"]))))) * 2.0)) * 2.0)) +

            1.0*np.tanh((((((((data["exp_Moving_std_3000_mean"]) + (-2.0))) + (((((((data["percentile_roll_std_80_window_500"]) + (data["autocorrelation_5"]))) / 2.0)) / 2.0)))/2.0)) - (((((data["percentile_roll_std_75_window_500"]) * 2.0)) * 2.0)))) +

            1.0*np.tanh((((-1.0*((data["fftr_exp_Moving_std_3000_mean"])))) * (((((data["num_peaks_10"]) + (data["abs_percentile_50"]))) * (((((data["percentile_roll_std_30_window_1000"]) * 2.0)) + (((1.0) + (data["percentile_roll_std_30_window_1000"]))))))))) +

            1.0*np.tanh(((data["exp_Moving_std_3000_mean"]) - (((data["percentile_roll_std_75_window_500"]) + ((((((((data["abs_percentile_50"]) + (data["percentile_roll_std_75_window_500"]))/2.0)) * ((((data["abs_percentile_50"]) + (data["percentile_roll_std_75_window_500"]))/2.0)))) * (data["fftr_exp_Moving_std_3000_mean"]))))))) +

            1.0*np.tanh((((((((((data["percentile_roll_std_80_window_500"]) - (data["fftr_exp_Moving_std_3000_mean"]))) + (((data["fftr_classic_sta_lta1_mean"]) - (data["fftr_classic_sta_lta2_mean"]))))/2.0)) + (((data["fftr_classic_sta_lta1_mean"]) - (data["fftr_exp_Moving_std_3000_mean"]))))) * (data["num_peaks_10"]))))
print(feats)
mean_absolute_error(X.time_to_failure,GPI(X))
mean_absolute_error(X.time_to_failure,GPII(X))
X_test['time_to_failure'] = (GPI(X_test)+GPII(X_test))/2.

X_test[['seg_id','time_to_failure']].to_csv('submission.csv',index=False)