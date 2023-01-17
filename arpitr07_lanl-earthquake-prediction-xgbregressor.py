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
import time

import datetime

import warnings

import matplotlib.pyplot as plt

import seaborn as sns

import xgboost as xgb

import lightgbm as lgb

from tqdm import tqdm

import scipy as sp

from sklearn.svm import NuSVR, SVR

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error

from sklearn.linear_model import LassoCV, RidgeCV

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import KFold, cross_val_score

warnings.filterwarnings("ignore")

warnings.simplefilter(action='ignore', category=FutureWarning)

from numpy import inf

pd.options.display.precision = 15
import os

import gc

from joblib import Parallel, delayed
def classic_sta_lta(x, Ns, Nl):

    sta = np.cumsum(x ** 2)

    sta = np.require(sta, dtype=np.float)

    lta = sta.copy()

    lta[Nl:-Ns] = lta[Nl:-Ns] - lta[:-Nl-Ns]

    lta /= Nl

    sta[Nl+Ns-1:] = sta[Nl+Ns-1:] - sta[Nl-1:-Ns]

    sta /= Ns

    sta[:Nl - 1 + Ns] = 0

    dtiny = np.finfo(0.0).tiny

    idx = lta < dtiny

    lta[idx] = dtiny

    return sta / lta
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

                x = df.acoustic_data.values

                y = df.time_to_failure.values[-1]

                seg_id = 'train_' + str(counter)

                del df

                yield seg_id, x, y

        else:

            for seg_id, f in self.test_files:

                df = pd.read_csv(f, dtype={'acoustic_data': np.float64})

                x = df.acoustic_data.values[-self.chunk_size:]

                del df

                yield seg_id, x, -999



    def features(self, x, y, seg_id):

        feature_dict = dict()

        feature_dict['target'] = y

        feature_dict['seg_id'] = seg_id



        # create features here

        feature_dict['mean'] = np.mean(x)

        feature_dict['max'] = np.max(x)

        feature_dict['min'] = np.min(x)

        feature_dict['std'] = np.std(x)

        feature_dict['var'] = np.var(x)

        feature_dict['quantile_03'] = np.quantile(x, 0.03)

        feature_dict['skew'] = sp.stats.skew(x)

        feature_dict['kurtosis'] = sp.stats.kurtosis(x)

        feature_dict['moment_3'] = sp.stats.moment(x, 3)

        

        pct_change = pd.Series(x).pct_change()

        pct_change[pct_change == -inf] = 0

        pct_change[pct_change == inf] = 0

        feature_dict['pct_change_mean'] = pct_change.mean()

        rate_change = pd.Series(x).pct_change().pct_change()

        rate_change[rate_change == -inf] = 0

        rate_change[rate_change == inf] = 0

        feature_dict['rate_change_max'] = rate_change.max()

        feature_dict['rate_change_mean'] = rate_change.mean()

        feature_dict['classic_sta_lta_mean'] = classic_sta_lta(x, 100, 5000).mean()

        

        window_size = 10

        x_roll_std = pd.Series(x).rolling(window_size).std().dropna().values

        feature_dict['q03_roll_std_' + str(window_size)] = np.quantile(x_roll_std, 0.03)

        window_size = 150

        x_roll_std = pd.Series(x).rolling(window_size).std().dropna().values

        feature_dict['q03_roll_std_' + str(window_size)] = np.quantile(x_roll_std, 0.03)

        

        return feature_dict

    

    def generate(self):

        feature_list = []

        res = Parallel(n_jobs=self.n_jobs,

                       backend='threading')(delayed(self.features)(x, y, s)

                                            for s, x, y in tqdm(self.read_chunks(), total=self.total_data))

        for r in res:

            feature_list.append(r)

        return pd.DataFrame(feature_list)

    

training_fg = FeatureGenerator(dtype='train', n_jobs=10, chunk_size=150000)

training_data = training_fg.generate()



test_fg = FeatureGenerator(dtype='test', n_jobs=10, chunk_size=150000)

test_data = test_fg.generate()

        
X = training_data.drop(['target', 'seg_id'], axis=1)

X_test = test_data.drop(['target', 'seg_id'], axis=1)

test_segs = test_data.seg_id

y = training_data.target
folds = KFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros((len(X), 1))

test_preds = np.zeros((len(X_test), 1))
params = {

    "learning_rate": 0.01,

    "max_depth": 3,

    "n_estimators": 10000,

    "min_child_weight": 4,

    "colsample_bytree": 1,

    "subsample": 0.9,

    "nthread": 12,

    "random_state": 42

}
for fold_, (trn_, val_) in enumerate(folds.split(X)):

    print("Current Fold: {}".format(fold_))

    trn_x, trn_y = X.iloc[trn_], y.iloc[trn_]

    val_x, val_y = X.iloc[val_], y.iloc[val_]



    clf = xgb.XGBRegressor(**params)

    clf.fit(

        trn_x, trn_y,

        eval_set=[(trn_x, trn_y), (val_x, val_y)],

        eval_metric='mae',

        verbose=150,

        early_stopping_rounds=100

    )

    val_pred = clf.predict(val_x, ntree_limit=clf.best_ntree_limit)

    test_fold_pred = clf.predict(X_test, ntree_limit=clf.best_ntree_limit)

    print("MAE = {}".format(mean_absolute_error(val_y, val_pred)))

    oof_preds[val_, :] = val_pred.reshape((-1, 1))

    test_preds += test_fold_pred.reshape((-1, 1))

test_preds /= 5



oof_score = mean_absolute_error(y, oof_preds)

print("Mean MAE = {}".format(oof_score))
print(clf.feature_importances_)
print(training_data.columns)
feature_importance = pd.concat([pd.Series(list(set(list(training_data)) - set(['seg_id', 'target']))), pd.Series(clf.feature_importances_)], axis = 1, keys = ['feature', 'importance'])
feature_importance.sort_values(by = ['importance'], ascending = False, inplace = True)
feature_importance
submission = pd.DataFrame(columns=['seg_id', 'time_to_failure'])

submission.seg_id = test_segs

submission.time_to_failure = test_preds

submission.to_csv('submission.csv', index=False)