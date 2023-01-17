%matplotlib inline



from collections import Counter, OrderedDict

from os.path import join



import catboost as cb

import hyperopt

import hyperopt.pyll.stochastic

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns



from IPython.display import FileLink

from hyperopt import hp, fmin, tpe

from hyperopt.pyll.base import scope

from pandas_summary import DataFrameSummary

from sklearn import preprocessing

from sklearn.base import TransformerMixin

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn.pipeline import Pipeline

from tqdm import tqdm_notebook as tqdm
ROOT = '/kaggle/input/flight-delays-fall-2018/'

TARGET_COL = 'dep_delayed_15min'

SEED = 1



pd.options.display.max_columns = None



trn_df = pd.read_csv(join(ROOT, 'flight_delays_train.csv'))

trn_data = trn_df.drop(columns=[TARGET_COL])

trn_target = trn_df[TARGET_COL].map({'Y': 1, 'N': 0})

tst_data = pd.read_csv(join(ROOT, 'flight_delays_test.csv'))
class BaseTransform(TransformerMixin):

    def fit(self, df):

        return self

    def transform(self, df):

        return self.apply(df.copy())
class AddFlightsFeature(BaseTransform):

    def apply(self, df):

        df['Flight'] = df['Origin'] + '-->' + df['Dest']

        return df

    

class AddSeasonFeature(BaseTransform):

    def apply(self, df):

        df['Season'] = df['Month'].str.slice(2).astype(int).map(

            lambda x: 

            'winter' if x in (11, 12, 1) else 

            'spring' if x in (2, 3, 4) else

            'summer' if x in (5, 6, 7) else

            'fall'

        )

        return df

    

class AddWeekendFeature(BaseTransform):

    def __init__(self, weekend=(6, 7)):

        self.weekend = weekend

    def apply(self, df):

        df['Weekend'] = df['DayOfWeek'].str.slice(2).astype(int).map(

            lambda x: 'Yes' if x in self.weekend else 'No')

        return df

    

class AddDepartureFeatures(BaseTransform):

    def apply(self, df):

        df['DepHour'] = df['DepTime'].map(lambda x: (x // 100) % 24)

        df['DepMinute'] = df['DepTime'].map(lambda x: x % 100)

        df['DepDaytime'] = df['DepHour'].map(

            lambda x:

            'morning' if 6 <= x < 12 else

            'afternoon' if 12 <= x < 16 else

            'evening' if 16 <= x < 22 else

            'night'

        )

        return df

    

class AddDaySinceYearStart(BaseTransform):

    def __init__(self):

        from calendar import mdays

        self.year_days = np.cumsum(mdays)

        

    def apply(self, df):

        months = df['Month'].str.slice(2).astype(int)

        days = df['DayofMonth'].str.slice(2).astype(int)

        df['DaysSinceYearStart'] = months.map(lambda x: self.year_days[x - 1]) + days

        return df

    

class AddInteractionsFeatures(BaseTransform):

    def __init__(self, pairs):

        self.pairs = pairs

    def apply(self, df):

        for a, b in self.pairs:

            df[f'{a}_{b}'] = df[a].astype(str) + '_' + df[b].astype(str)

        return df

    

class LogDistance(BaseTransform):

    def apply(self, df):

        df['Distance_Log'] = np.log(df['Distance'])

        return df

    

class AddFrequencyFeatures(BaseTransform):

    def __init__(self, columns):

        self.columns = columns

    def apply(self, df):

        for col in self.columns:

            df[f'{col}_freq'] = df[col].map(df[col].value_counts(normalize=True))

        return df

    

class HarmonicFeatures(BaseTransform):

    def __init__(self, col, modulo):

        self.col = col

        self.modulo = modulo

    def apply(self, df):

        df[f'{self.col}_sin'] = np.sin(2*np.pi*df[self.col]/self.modulo)

        df[f'{self.col}_cos'] = np.cos(2*np.pi*df[self.col]/self.modulo)

        return df

    

class Bucketize(BaseTransform):

    def __init__(self, col, buckets, prep_fn=None):

        self.col = col

        self.buckets = buckets

        self.prep_fn = prep_fn

    def apply(self, df):

        col = df[self.col]

        if self.prep_fn:

            col = self.prep_fn(col)

        df[f'{self.col}_bucket'] = pd.qcut(

            col, self.buckets, labels=range(self.buckets)) 

        return df

    

class BinarizeThreshold(BaseTransform):

    def __init__(self, col, t, prep_fn=None):

        self.col = col

        self.t = t

        self.prep_fn = prep_fn

    def apply(self, df):

        col = df[self.col]

        if self.prep_fn:

            col = self.prep_fn(col)

        df[f'{self.col}_above_t={self.t}'] = (col > self.t).astype(int)

        return df

    

class DropColumns(BaseTransform):

    def __init__(self, cols):

        self.cols = cols

    def apply(self, df):

        return df.drop(columns=self.cols)
def to_int(x):

    return x.str.slice(2).astype(int)



def categorical_columns_indexes(df):

    return [i for i, col in enumerate(df.columns) 

            if df.dtypes[col] not in (np.float32, np.float64)]



pipeline = Pipeline([

    ('flight', AddFlightsFeature()),

    ('season', AddSeasonFeature()),

    ('weekend', AddWeekendFeature()),

    ('daytime', AddDepartureFeatures()),

    ('start_days', AddDaySinceYearStart()),

    ('interact', AddInteractionsFeatures([

        ['UniqueCarrier', 'Origin'], 

        ['UniqueCarrier', 'Dest']

    ])),

    ('logdist', LogDistance()),

    ('freq', AddFrequencyFeatures(['UniqueCarrier', 'Origin', 'Dest'])),

    ('harmonic_hour' , HarmonicFeatures('DepHour', modulo=24)),

    ('harmonic_minute', HarmonicFeatures('DepMinute', modulo=60)),

    ('bucket_dom', Bucketize('DayofMonth', 4, prep_fn=to_int)),

    ('binarize_dow', BinarizeThreshold('DayOfWeek', t=4, prep_fn=to_int)),

    ('binarize_dsy', BinarizeThreshold('DaysSinceYearStart', t=365//2)),

    ('dephour_t1', BinarizeThreshold('DepHour', t=5)),

    ('dephour_t2', BinarizeThreshold('DepHour', t=12)),

    ('dephour_t3', BinarizeThreshold('DepHour', t=18)),

    ('bucket_dist', Bucketize('Distance', 5)),



    # Note: some features are excluded from this pipeline to reduce kernel's score.



    ('drop', DropColumns(['Distance']))

])



eng_trn_data = pipeline.fit_transform(trn_data)

eng_tst_data = pipeline.transform(tst_data)

cat_idx = categorical_columns_indexes(eng_trn_data)



X = eng_trn_data.values

y = trn_target.values

X_test = eng_tst_data.values
X
counts = trn_target.value_counts()

class_weights = [1, counts[0]/counts[1]]
params = dict(

    depth=8,

    l2_leaf_reg=0.5,

    bagging_temperature=2.0,

    border_count=64,

    grow_policy='Lossguide',

    num_leaves=10,

    class_weights=class_weights,

    eval_metric='AUC',

    task_type='GPU',

    loss_function='Logloss')
X_train, X_valid, y_train, y_valid = train_test_split(X, y,

                                                      test_size=0.3,

                                                      random_state=SEED,

                                                      stratify=y)



trn_pool = cb.Pool(X_train, y_train, cat_features=cat_idx)

val_pool = cb.Pool(X_valid, y_valid, cat_features=cat_idx)



model = cb.train(params=params,

                 dtrain=trn_pool,

                 verbose=100, 

                 early_stopping_rounds=200,

                 eval_set=val_pool, 

                 iterations=1000)
def feature_importance(model):

    scores = model.feature_importances_

    indexes = [int(c) for c in model.feature_names_]

    ser = pd.Series(dict(zip(indexes, scores)))

    ser.sort_values(inplace=True)

    return ser
def plot_feature_importance(model, colnames):

    fig, ax = plt.subplots(figsize=(12, 10))

    importance = feature_importance(model)

    importance.index = [colnames[i] for i in importance.index]

    importance.plot.barh(ax=ax)
plot_feature_importance(model, eng_trn_data.columns)
importance = feature_importance(model)

relevant_cols = eng_trn_data.columns[importance[importance > 1].index]

cat_idx = categorical_columns_indexes(eng_trn_data[relevant_cols])

X = eng_trn_data[relevant_cols].values

y = trn_target.values

X_test = eng_tst_data[relevant_cols].values
def logit(x):

    return 1/(1 + np.exp(x))
space = dict()



def p(name, func, *args, scope_fn=None):

    distrib = func(name, *args)

    if scope_fn is not None:

        distrib = scope_fn(distrib)

    space[name] = distrib



print('Building search space...')

p('depth', hp.quniform, 3, 10, 1, scope_fn=scope.int)

p('l2_leaf_reg', hp.uniform, 0.01, 50.0)

p('random_strength', hp.uniform, 0.0, 100.0)

p('bagging_temperature', hp.uniform, 0, 20.0)

p('border_count', hp.quniform, 1, 255, 1, scope_fn=scope.int)

p('grow_policy', hp.choice, ['SymmetricTree', 'Depthwise', 'Lossguide'])



def catboost_search(params):

    params['silent'] = True

    params['loss_function'] = 'Logloss'

    params['class_weights'] = class_weights

    params['task_type'] = 'GPU'

    model = cb.train(

        params=params, dtrain=trn_pool,

        early_stopping_rounds=10,

        eval_set=val_pool, iterations=100)

    probs = 1 - logit(model.predict(X_valid))

    auc = roc_auc_score(y_valid, probs)

    return {'loss': -auc, 'status': hyperopt.STATUS_OK, 'params': params}



trials = hyperopt.Trials()



# Clone the kernel locally and un-comment the following section to perform search.

#

# best = fmin(catboost_search, 

#             space=space, 

#             algo=tpe.suggest, 

#             trials=trials,

#             max_queue_len=12,

#             max_evals=10)
best = dict(

    depth=8,

    l2_leaf_reg=25.922353989859875,

    bagging_temperature=1.6853010941877322,

    border_count=65.0,

    grow_policy='Lossguide',

    num_leaves=35,

    random_strength=0.8073770414011081,

    class_weights=class_weights,

    eval_metric='AUC',

    task_type='GPU',

    loss_function='Logloss')
from sklearn.model_selection import StratifiedKFold

k = 5

kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)

preds = np.zeros((len(X_test), k), dtype=np.float32)

for i, (trn, val) in enumerate(kfold.split(X, y)):

    print(f'Running k-fold: {i+1} of {k}')

    trn_pool = cb.Pool(X[trn], y[trn], cat_features=cat_idx)

    val_pool = cb.Pool(X[val], y[val], cat_features=cat_idx)

    model = cb.train(params=best,

                     dtrain=trn_pool,

                     verbose=100, 

                     early_stopping_rounds=200,

                     eval_set=val_pool, 

                     iterations=10000)

    fold_preds = 1 - logit(model.predict(X_test))

    preds[:, i] = fold_preds
y_test = preds.mean(axis=1)
filename = 'submit.csv'

sample_df = pd.read_csv(join(ROOT, 'sample_submission.csv'), index_col='id')

sample_df[TARGET_COL] = y_test

sample_df.to_csv(filename)

# FileLink(filename)