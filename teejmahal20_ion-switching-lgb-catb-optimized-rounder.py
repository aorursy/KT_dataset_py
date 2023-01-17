import numpy as np 

import pandas as pd

from sklearn import *

import lightgbm as lgb



train = pd.read_csv('/kaggle/input/liverpool-ion-switching/train.csv')

test = pd.read_csv('/kaggle/input/liverpool-ion-switching/test.csv')

train.shape, test.shape
from functools import partial

import scipy as sp

class OptimizedRounder(object):

    """

    An optimizer for rounding thresholds

    to maximize F1 (Macro) score

    # https://www.kaggle.com/naveenasaithambi/optimizedrounder-improved

    """

    def __init__(self):

        self.coef_ = 0



    def _f1_loss(self, coef, X, y):

        """

        Get loss according to

        using current coefficients

        

        :param coef: A list of coefficients that will be used for rounding

        :param X: The raw predictions

        :param y: The ground truth labels

        """

        X_p = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])



        return -metrics.f1_score(y, X_p, average = 'macro')



    def fit(self, X, y):

        """

        Optimize rounding thresholds

        

        :param X: The raw predictions

        :param y: The ground truth labels

        """

        loss_partial = partial(self._f1_loss, X=X, y=y)

        initial_coef = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]

        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')



    def predict(self, X, coef):

        """

        Make predictions with specified thresholds

        

        :param X: The raw predictions

        :param coef: A list of coefficients that will be used for rounding

        """

        return pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])





    def coefficients(self):

        """

        Return the optimized coefficients

        """

        return self.coef_['x']
def optimize_prediction(prediction, coefficients):

    prediction[prediction <= coefficients[0]] = 0

    prediction[np.where(np.logical_and(prediction > coefficients[0], prediction <= coefficients[1]))] = 1

    prediction[np.where(np.logical_and(prediction > coefficients[1], prediction <= coefficients[2]))] = 2

    prediction[np.where(np.logical_and(prediction > coefficients[2], prediction <= coefficients[3]))] = 3

    prediction[np.where(np.logical_and(prediction > coefficients[3], prediction <= coefficients[4]))] = 4

    prediction[np.where(np.logical_and(prediction > coefficients[4], prediction <= coefficients[5]))] = 5

    prediction[np.where(np.logical_and(prediction > coefficients[5], prediction <= coefficients[6]))] = 6

    prediction[np.where(np.logical_and(prediction > coefficients[6], prediction <= coefficients[7]))] = 7

    prediction[np.where(np.logical_and(prediction > coefficients[7], prediction <= coefficients[8]))] = 8

    prediction[np.where(np.logical_and(prediction > coefficients[8], prediction <= coefficients[9]))] = 9

    prediction[prediction > coefficients[9]] = 10

    

    return prediction
def features(df):

    df = df.sort_values(by=['time']).reset_index(drop=True)

    df.index = ((df.time * 10_000) - 1).values

    df['batch'] = df.index // 50_000

    df['batch_index'] = df.index  - (df.batch * 50_000)

    df['batch_slices'] = df['batch_index']  // 5_000

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

        for v in d:

            df[v] = df[c].map(d[v].to_dict())

        df['range'+c] = df['max'+c] - df['min'+c]

        df['maxtomin'+c] = df['max'+c] / df['min'+c]

        df['abs_avg'+c] = (df['abs_min'+c] + df['abs_max'+c]) / 2

    

    #add shifts

    df['signal_shift_+1'] = [0,] + list(df['signal'].values[:-1])

    df['signal_shift_-1'] = list(df['signal'].values[1:]) + [0]

    for i in df[df['batch_index']==0].index:

        df['signal_shift_+1'][i] = np.nan

    for i in df[df['batch_index']==49999].index:

        df['signal_shift_-1'][i] = np.nan



    for c in [c1 for c1 in df.columns if c1 not in ['time', 'signal', 'open_channels', 'batch', 'batch_index', 'batch_slices', 'batch_slices2']]:

        df[c+'_msignal'] = df[c] - df['signal']

        

    return df



train = features(train)

test = features(test)
col = [c for c in train.columns if c not in ['time', 'open_channels', 'batch', 'batch_index', 'batch_slices', 'batch_slices2']]

x1, x2, y1, y2 = model_selection.train_test_split(train[col], train['open_channels'], test_size=0.3, random_state=7)
def lgb_Metric(preds, dtrain):

    labels = dtrain.get_label()

    preds = np.round(np.clip(preds, 0, 10)).astype(int)

    score = metrics.f1_score(labels, preds, average='macro')

    return ('F1 Macro', score, True)

 

params = {'learning_rate': 0.8, 'max_depth': 7, 'num_leaves':2**7+1, 'metric': 'rmse', 'random_state': 7, 'n_jobs':-1} 

model = lgb.train(params, lgb.Dataset(x1, y1), 2000,  lgb.Dataset(x2, y2), verbose_eval=50, early_stopping_rounds=100, feval=lgb_Metric)

train_preds_lgb = model.predict(train[col], num_iteration=model.best_iteration)

preds_lgb = model.predict(test[col], num_iteration=model.best_iteration)
optR_lgb = OptimizedRounder()

optR_lgb.fit(train_preds_lgb.reshape(-1,), train['open_channels'])

coefficients_lgb = optR_lgb.coefficients()

print(coefficients_lgb)
test['open_channels'] = optimize_prediction(preds_lgb,coefficients_lgb).astype(int)

test[['time','open_channels']].to_csv('submission_lgb.csv', index=False, float_format='%.4f')
from catboost import Pool,CatBoostRegressor



# Initialize CatBoostRegressor

model = CatBoostRegressor(task_type = "CPU",

                          iterations=1000,

                          learning_rate=0.1,

                          random_seed = 42,

                          depth=2,

                         )

# Fit model

model.fit(x1, y1)

# Get predictions

train_preds_cat = model.predict(train[col])

preds_catb = model.predict(test[col])
optR_cat = OptimizedRounder()

optR_cat.fit(train_preds_cat.reshape(-1,), train['open_channels'])

coefficients_cat = optR_cat.coefficients()

print(coefficients_cat)
test['open_channels'] = optimize_prediction(preds_catb, coefficients_cat).astype(int)

test[['time','open_channels']].to_csv('submission_cat.csv', index=False, float_format='%.4f')
train_preds_comb = 0.75 * train_preds_lgb + 0.25 * train_preds_cat

optR = OptimizedRounder()

optR.fit(train_preds_comb.reshape(-1,), train['open_channels'])

coefficients = optR.coefficients()

print(coefficients)
preds_comb = 0.75 * preds_lgb + 0.25 * preds_catb


test['open_channels'] = optimize_prediction(preds_comb, coefficients).astype(int)

test[['time','open_channels']].to_csv('submission.csv', index=False, float_format='%.4f')
