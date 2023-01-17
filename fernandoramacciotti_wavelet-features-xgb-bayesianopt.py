import numpy as np

import pandas as pd

from scipy import stats

from scipy import signal

import pywt

import numba



from tqdm.auto import tqdm

from joblib import Parallel, delayed

import gc
@numba.jit(parallel=True)

def wavelet_coeffs(x, wavelet='db9', level=9):

    coeffs = pywt.wavedec(x, wavelet=wavelet, level=level)

    return coeffs



class FeatureGenerator(object):

    def __init__(self, dtype, n_jobs=1, chunk_size=None, fs=4e6, wavelet='db9', level=9):

        self.chunk_size = chunk_size

        self.dtype = dtype

        self.filename = None

        self.n_jobs = n_jobs

        self.fs = fs

        self.wavelet = wavelet

        self.level = level

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

                if df.time_to_failure.values[0] > df.time_to_failure.values[-1]:

                    x = df.acoustic_data.values

                    y = df.time_to_failure.values[-1]

                    seg_id = 'train_' + str(counter)

                    yield seg_id, x, y

        else:

            for seg_id, f in self.test_files:

                df = pd.read_csv(f, dtype={'acoustic_data': np.float64})

                x = df.acoustic_data.values

                yield seg_id, x, -999

                

    @numba.jit(parallel=True)

    def features(self, x, y, seg_id, fs=4e6):

        feature_dict = dict()

        feature_dict['target'] = y

        feature_dict['seg_id'] = seg_id

        coeffs = wavelet_coeffs(x, wavelet=self.wavelet, level=self.level)

        coeffs_diff = wavelet_coeffs(np.diff(x), wavelet=self.wavelet, level=self.level)

        percentiles_ranges = [99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 80, 75, 

                              50, 25, 20, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

        

        signals_list = {

            'regular': coeffs, 

            'diff': coeffs_diff,

        }

        for signal_name, signal_i in signals_list.items():

            for i, x_i in enumerate(signal_i):

                if i == 0:

                    name = '{}_cA'.format(signal_name)

                else:

                    name = '{}_cD{}'.format(signal_name, self.level - (i - 1))

                # statistics and centered moments

                feature_dict['rms_{}'.format(name)] = np.sqrt(np.mean(np.sum(x_i ** 2)))

                feature_dict['mean_{}'.format(name)] = np.mean(x_i)

                feature_dict['median_{}'.format(name)] = np.median(x_i)

                feature_dict['var{}'.format(name)] = np.var(x_i)

                feature_dict['skewnes_{}'.format(name)] = stats.skew(x_i)

                feature_dict['kurtosis_{}'.format(name)] = stats.kurtosis(x_i)

                # non-centered moments

                for m in range(2, 5):

                    feature_dict['moment_{}_{}'.format(m, name)] = np.mean(np.sum(x_i ** m))

                # percentile ranges

                for pct in percentiles_ranges:

                    feature_dict['percentile{}_{}'.format(str(pct), name)] = np.percentile(x_i, pct)

                # sum of energy of coefficients within bands

                chunks = 20

                step = len(x_i) // chunks

                for chunk_no, band in enumerate(range(0, len(x_i), step)):

                    feature_dict['energy_chunk{}_{}'.format(chunk_no, name)] = np.sum(x_i[band:band+step] ** 2)

                    feature_dict['energy_chunk_rms{}_{}'.format(chunk_no, name)] = np.sqrt(

                        np.mean(

                            np.sum(

                                feature_dict['energy_chunk{}_{}'.format(chunk_no, name)] ** 2)

                        )

                    )

                

        return feature_dict



    def generate(self):

        feature_list = []

        res = Parallel(n_jobs=self.n_jobs)(delayed(self.features)(x, y, s, fs=self.fs)

                                            for s, x, y in tqdm(self.read_chunks(), total=self.total_data))

        for r in res:

            feature_list.append(r)

        return pd.DataFrame(feature_list)
wavelet = 'db4'

level = 9

training_fg = FeatureGenerator(dtype='train', n_jobs=1, chunk_size=150000, wavelet=wavelet, level=level)

training_data = training_fg.generate()



test_fg = FeatureGenerator(dtype='test', n_jobs=1, chunk_size=None, wavelet=wavelet, level=level)

test_data = test_fg.generate()



training_data.to_csv("train_features.csv", index=False)

test_data.to_csv("test_features.csv", index=False)
training_data.sample(5)
training_data.dropna(axis=1, inplace=True)

test_data.dropna(axis=1, inplace=True)

print(training_data.shape, test_data.shape)
from sklearn.model_selection import KFold, TimeSeriesSplit

from sklearn.metrics import mean_absolute_error, make_scorer

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor
features_cols = [c for c in training_data.columns if (('target' not in c) and ('seg_id' not in c))]



X = training_data[features_cols].values

y = training_data['target'].values

X_test = test_data[features_cols].values
from skopt import BayesSearchCV

from skopt.space import Real, Categorical, Integer
xgb = XGBRegressor(random_state=11)

search_space = {

    'n_estimators': Integer(100, 1000),

    'learning_rate': Real(1e-6, 3e-1, 'log-uniform'),

    'min_child_weight': Integer(4, 10),

    'reg_alpha': Real(1e-6, 0.5, 'log-uniform'),

    'reg_lambda': Real(1e-6, 1.0, 'log-uniform'),

    'colsample_bytree': Real(0.2, 0.8, 'log-uniform'),

}
#folds = KFold(n_splits=3, random_state=11)

#cv = folds.split(X, y)



# weird error using KFold or TimeSeriesSplit



opt = BayesSearchCV(

    xgb,

    search_space,

    scoring=make_scorer(mean_absolute_error, greater_is_better=False),

    n_iter=32,

    cv=5,

    n_jobs=-1,

    random_state=11,

    refit=True,

)



opt.fit(X, y)
print('val. score: {:.3f}'.format(opt.best_score_))
print(opt.best_params_)
y_pred = opt.predict(X_test)



# submission

sub = pd.read_csv('../input/sample_submission.csv')

sub['time_to_failure'] = y_pred

sub.to_csv('submission.csv', index=False)