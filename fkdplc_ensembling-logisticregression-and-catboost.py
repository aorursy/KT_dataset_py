import numpy as np

import pandas as pd

import os

import math

from tqdm.notebook import tqdm

from sklearn.exceptions import ConvergenceWarning

import warnings



warnings.simplefilter(action='ignore', category=FutureWarning)

warnings.simplefilter(action='ignore', category=ConvergenceWarning)



PATH = '/kaggle/input/cat-in-the-dat-ii/'

train = pd.read_csv(PATH + 'train.csv')

test = pd.read_csv(PATH + 'test.csv')



# separate target, remove id and target

test_ids = test['id']

target = train['target']

train.drop(columns=['id', 'target'], inplace=True)

test.drop(columns=['id'], inplace=True)



train.head()
import category_encoders as ce



te = ce.TargetEncoder(cols=train.columns.values, smoothing=0.3).fit(train, target)



train = te.transform(train)

train.head()
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV



# split train data

x_train, x_test, y_train, y_test = train_test_split(

    train, target,  

    test_size=0.2, 

    random_state=289

)



# hyperparams setting

def make_search(estimator, params, verbose=1):

    search = GridSearchCV(estimator, params, cv=5, scoring='roc_auc', verbose=verbose, n_jobs=-1)

    search.fit(x_train, y_train)

    results = pd.DataFrame()

    for k, v in search.cv_results_.items():

        results[k] = v

    results = results.sort_values(by='rank_test_score')

    best_params_row = results[results['rank_test_score'] == 1]

    mean, std = best_params_row['mean_test_score'].iloc[0], best_params_row['std_test_score'].iloc[0]

    best_params = best_params_row['params'].iloc[0]

    if verbose:

        print('%.4f (%.4f) with params' % (mean, std), best_params)

    return best_params



# make prediction

def predict(estimator, features):

    return estimator.predict_proba(features)[:, 1]



# calculate auc

def auc(estimator):

    y_pred = predict(estimator, x_test)

    return roc_auc_score(y_test, y_pred)



# precalculated best params (0.7979)

best_params_lr = {

    'solver': 'lbfgs',

    'C': 2,

    'max_iter': 200,

    'random_state': 289

}



SEARCH_NOW = False



# this is to reproduce search

if SEARCH_NOW:

    params = {

        'solver': ['lbfgs'],

        'C': [1, 1.5, 2, 2.5, 3, 3.5, 4, 5],

        'max_iter': [100, 150, 200, 250, 300],

        'random_state': [289]

    }

    best_params_lr = make_search(LogisticRegression(), params)   



# fit logistic regressor

print('training..')

lr = LogisticRegression()

lr.set_params(**best_params_lr)

lr.fit(x_train, y_train)



# check auc

print('validation..')

print('roc auc = %.4f' % auc(lr))
from catboost import CatBoostClassifier



# precalculated best params (0.7978)

best_params_cat = {

    'max_depth': 2,

    'n_estimators': 600,

    'random_state': 289,

    'verbose': 0

}



SEARCH_NOW = False



# this is to reproduce search

if SEARCH_NOW:

    params = {

        'max_depth': [2, 3, 4, 5],

        'n_estimators': [50, 100, 200, 400, 600],

        'random_state': [289],

        'verbose': [0]

    }

    best_params_cat = make_search(CatBoostClassifier(), params)   



# fit xgb

print('training..')

cat = CatBoostClassifier()

cat.set_params(**best_params_cat)

cat.fit(x_train, y_train)



# check auc

print('validation..')

print('roc auc = %.4f' % auc(cat))
lr_predict = predict(lr, x_test)

cat_predict = predict(cat, x_test)



print('lr auc: %.4f' % roc_auc_score(y_test, lr_predict))

print('cat auc: %.4f' % roc_auc_score(y_test, cat_predict))



best_auc = 0

best_off = 0

aucs = []

offsets = [i * .01 for i in range(101)]

for off in tqdm(offsets):

    off_predict = lr_predict * off + cat_predict * (1 - off)

    auc = roc_auc_score(y_test, off_predict)

    aucs.append(auc)

    if auc > best_auc:

        best_auc = auc

        best_off = off



print('best auc %.4f reached on offset %.2f' % (best_auc, best_off))        



from matplotlib import pyplot as plt

plt.figure(1, figsize=(8, 8))

plt.xlabel('<- cat ------------- lr ->')

plt.ylabel('auc')

plt.plot(offsets, aucs)

plt.plot([best_off], [best_auc], 'o', color='r', label='best score (%.4f) at offset %.2f' % (best_auc, best_off))

plt.legend()

plt.show()
# create submission file

test = te.transform(test)



res = pd.DataFrame()

res['id'] = test_ids

res['target'] = predict(lr, test) * best_off + predict(cat, test) * (1 - best_off)

res.to_csv('submission.csv', index=False)

res.head(20)