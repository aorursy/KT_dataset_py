from collections import ChainMap

from multiprocessing import cpu_count

from pathlib import Path
import numpy as np

import pandas as pd

import lightgbm as lgb

from sklearn.externals.joblib import Parallel, delayed

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split, StratifiedKFold

from tqdm import tqdm_notebook as tqdm

from tsfresh.feature_extraction.feature_calculators import *

from tsfresh.feature_selection.relevance import calculate_relevance_table
seed = 1

np.random.seed(seed)
ROOT = Path.cwd().parent/'input'

SAMPLE = ROOT/'sample_submission.csv'

TRAIN = ROOT/'X_train.csv'

TARGET = ROOT/'y_train.csv'

TEST = ROOT/'X_test.csv'



ID_COLS = ['series_id', 'measurement_number']



x_cols = {

    'series_id': np.uint32,

    'measurement_number': np.uint32,

    'orientation_X': np.float32,

    'orientation_Y': np.float32,

    'orientation_Z': np.float32,

    'orientation_W': np.float32,

    'angular_velocity_X': np.float32,

    'angular_velocity_Y': np.float32,

    'angular_velocity_Z': np.float32,

    'linear_acceleration_X': np.float32,

    'linear_acceleration_Y': np.float32,

    'linear_acceleration_Z': np.float32

}



y_cols = {

    'series_id': np.uint32,

    'group_id': np.uint32,

    'surface': str

}
x_trn = pd.read_csv(TRAIN, usecols=x_cols.keys(), dtype=x_cols)

x_tst = pd.read_csv(TEST, usecols=x_cols.keys(), dtype=x_cols)

y_trn = pd.read_csv(TARGET, usecols=y_cols.keys(), dtype=y_cols)
def part(f, **params):

    """Partially applies the function's keyword parameters."""

    def wrapper(x): return f(x, **params)

    wrapper.__name__ = f.__name__

    return wrapper
class StatsFeatures:

    """Applies list of functions to a single instance of measurements 

    and returns dictionary with computed features.

    """

    def __init__(self, funcs):

        self.funcs = funcs

    

    def __call__(self, data):

        features = {}

        for col in data.columns:

            for func in self.funcs:

                result = func(data[col].values) 

                if hasattr(result, '__len__'):

                    for key, value in result:

                        features[f'{col}__{func.__name__}__{key}'] = value

                else:

                    features[f'{col}__{func.__name__}'] = result

        return features
class SliceFeatures:

    """Takes a slice of values from the original sequence of 

    observations.

    

    There types of slicing are supported:

        * first: take N observations from the beginning of the sequence.

        * middle: take N observations from the middle of the sequence.

        * last: take last N observations from the sequence.

        

    """

    def __init__(self, mode='first', n=5):

        if mode not in {'first', 'middle', 'last'}:

            raise ValueError('unexpected mode')

        self.mode = mode

        self.n = n

        

    def __call__(self, data):

        if self.mode == 'first':

            start, end = 0, self.n

        elif self.mode == 'last':

            start, end = -self.n, len(data)

        elif self.mode == 'middle':

            mid = len(data) // 2

            div, mod = divmod(self.n, 2)

            start, end = mid-div, mid+div+mod

        cols = data.columns

        vec = data.iloc[start:end].values.T.ravel()

        new_cols = [

            f'{col}_{self.mode}{i}' 

            for i in range(self.n) for col in cols]

        return dict(zip(new_cols, vec))
def generate_features(data, features, ignore=None):

    """Extracts tsfresh features from the dataset."""

    

    with Parallel(n_jobs=cpu_count()) as parallel:

        extracted = parallel(delayed(generate_features_for_group)(

            group=group.drop(columns=ignore or []),

            features=features

        ) for _, group in tqdm(data.groupby('series_id')))

    return pd.DataFrame(extracted)
def generate_features_for_group(group, features):

    """Extract tsfresh features from a single measurements group."""

    

    return dict(ChainMap(*[feat(group) for feat in features]))
funcs = (

    mean, median, standard_deviation, variance, 

    skewness, kurtosis, maximum, minimum,

    mean_change, mean_abs_change, count_above_mean, count_below_mean,

    mean_second_derivative_central, sum_of_reoccurring_data_points, 

    abs_energy, sum_values, sample_entropy,

    longest_strike_above_mean, longest_strike_below_mean,

    first_location_of_minimum, first_location_of_maximum,

    *[part(large_standard_deviation, r=r*0.05) for r in range(1, 20)],

    *[part(autocorrelation, lag=lag) for lag in range(1, 25)], 

    *[part(number_peaks, n=n) for n in (1, 2, 3, 5, 7, 10, 25, 50)],

    *[part(c3, lag=lag) for lag in range(1, 5)],

    *[part(quantile, q=q) for q in (.1, .2, .3, .4, .5, .6, .7, .8, .9)],

    part(partial_autocorrelation, param=[

        {'lag': lag} for lag in range(25)

    ]),

    part(agg_autocorrelation, param=[

        {'f_agg': s, 'maxlag': 40} for s in ('mean', 'median', 'var')

    ]),

    part(linear_trend, param=[

        {'attr': a} for a in 

        ('pvalue', 'rvalue', 'intercept', 'slope', 'stderr')

    ])

)
features = [

    StatsFeatures(funcs),

    SliceFeatures('first'),

    SliceFeatures('middle'),

    SliceFeatures('last')

]
ignore = ['series_id', 'measurement_number']
print('Feature extraction on train dataset')

x_trn = generate_features(x_trn, features, ignore=ignore)
print('Feature extraction on test dataset')

x_tst = generate_features(x_tst, features, ignore=ignore)
enc = LabelEncoder()

y_trn = pd.Series(enc.fit_transform(y_trn['surface']))
def accuracy(y_true, y_pred):

    n = len(y_true)

    y_hat = y_pred.reshape(9, n).argmax(axis=0)

    value = (y_true == y_hat).mean()

    return 'accuracy', value, True
model = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.005,

                           colsample_bytree=0.4, objective='multiclass',

                           num_leaves=500, num_class=9)
x_train, x_valid, y_train, y_valid = train_test_split(x_trn, y_trn, test_size=0.1, random_state=seed)
model.fit(x_train, y_train,

          eval_set=[(x_valid, y_valid)],

          eval_metric=accuracy,

          early_stopping_rounds=300,

          verbose=150)
# You could also try to run K-folded validation instead.

#

# k = 5

# test = np.zeros((len(x_tst_rich), 9), dtype=np.float32)

# kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

# for i, (trn_idx, val_idx) in enumerate(kfold.split(x_trn_rich.index, y_trn)):

#     x_train = x_trn[x_trn.isin(trn_idx)]

#     x_valid = x_trn[x_trn.isin(val_idx)]

#     y_train = y_trn[y_trn.isin(trn_idx)]

#     y_valid = y_trn[y_trn.isin(val_idx)]

#     model = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.005,

#                                colsample_bytree=0.4, objective='multiclass',

#                                num_leaves=500, num_class=9)

#     model.fit(x_train, y_train,

#               eval_set=[(x_valid, y_valid)],

#               eval_metric=accuracy,

#               early_stopping_rounds=300,

#               verbose=150)

#     test += model.predict_proba(x_tst_rich)

# test /= k
test = enc.inverse_transform(model.predict(x_tst))
submit = pd.read_csv(SAMPLE)

submit['surface'] = test

submit.to_csv('submit.csv', index=None)