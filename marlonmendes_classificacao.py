%load_ext autoreload

%autoreload 2

%matplotlib inline
import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from math import *

import os

import requests

import re

from sklearn.preprocessing import LabelEncoder





pd.set_option('display.max_rows', 1000)

pd.set_option('display.max_columns', 1000)



print(os.listdir("../input"))

DATA_FOLDER = '../input'





def read_csv(name, index_col=None):

    df = pd.read_csv(os.path.join(DATA_FOLDER, name), low_memory=False, index_col=index_col)

    return df
ex = read_csv('exemplo_resultado.csv')

display(ex.head())
df_train = read_csv('train.csv', index_col='ID')



display(df_train.dtypes)

display(df_train.describe())

display(df_train.head())

display(df_train.shape)
from sklearn.model_selection import train_test_split



TEST_SIZE = 0.2

RANDOM_STATE = 42



def split_train_test(df, ycol):

    X = df.drop(ycol, axis=1)

    y = df[ycol]

    res = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE) # Random state will generate the same dataset

    return res

df_train = read_csv('train.csv', index_col='ID')

x_train, x_test, y_train, y_test = split_train_test(df_train, 'default payment next month')

display(x_train.shape)
from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics







def fitpredict_score(m, x_train, y_train, x_valid, y_valid):

    train_predict = m.predict(x_train)

    valid_predict = m.predict(x_valid)

    data = {

      'naive AUROC': [0.999162, 0.666576],

      'AUROC': [metrics.roc_auc_score(y_train, train_predict), metrics.roc_auc_score(y_valid, valid_predict)]

    }

    



    columns = ['train', 'test']

    df = pd.DataFrame.from_dict(data, orient='index', columns=columns)

    display(df)

    

N_ESTIMATORS = 100

    

clf = RandomForestClassifier(n_jobs=-1, 

                             random_state=RANDOM_STATE,

                             n_estimators=N_ESTIMATORS,

                             verbose=False)

clf.fit(x_train, y_train)

display(clf.feature_importances_)

fitpredict_score(clf, x_train, y_train, x_test, y_test)
display(df_train.hist('LIMIT_BAL', bins=100))

display()
from sklearn.preprocessing import MinMaxScaler



def process_dataset(df, columns_to_normalize=None, columns_tooneshot=None):

    if columns_to_normalize:

        scaler = MinMaxScaler()

        df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

    

    if columns_tooneshot:

        df = pd.get_dummies(df, columns=columns_tooneshot)

    return df
df_train = process_dataset(df_train, columns_to_normalize=['LIMIT_BAL', ])

display(df_train.describe())

display(df_train.hist('LIMIT_BAL', bins=100))
df_train = read_csv('train.csv', index_col='ID')

df_train = process_dataset(df_train, columns_to_normalize=['LIMIT_BAL', ])



x_train, x_test, y_train, y_test = split_train_test(df_train, 'default payment next month')

clf.fit(x_train, y_train)

display(clf.feature_importances_)

fitpredict_score(clf, x_train, y_train, x_test, y_test)
column = 'SEX'

df_train = read_csv('train.csv', index_col='ID')

display(df_train[column].describe()) 

df_train.hist('SEX', bins=100)

display(df_train[column].nunique())
column = 'SEX'

columns_tooneshot = [column]



df_train = read_csv('train.csv', index_col='ID')

df_train = process_dataset(df_train, columns_tooneshot=columns_tooneshot)

display(df_train.columns)



x_train, x_test, y_train, y_test = split_train_test(df_train, 'default payment next month')

clf.fit(x_train, y_train)

fitpredict_score(clf, x_train, y_train, x_test, y_test)
display(df_train['EDUCATION'].unique())

display(df_train['EDUCATION'].nunique())

df_train.hist('EDUCATION')
column = 'EDUCATION'

columns_tooneshot = [column]



df_train = read_csv('train.csv', index_col='ID')

df_train = process_dataset(df_train, columns_tooneshot=columns_tooneshot)





x_train, x_test, y_train, y_test = split_train_test(df_train, 'default payment next month')

clf.fit(x_train, y_train)

fitpredict_score(clf, x_train, y_train, x_test, y_test)
column = 'MARRIAGE'

columns_tooneshot = [column]



df_train = read_csv('train.csv', index_col='ID')

df_train = process_dataset(df_train, columns_tooneshot=columns_tooneshot)



x_train, x_test, y_train, y_test = split_train_test(df_train, 'default payment next month')

clf.fit(x_train, y_train)

fitpredict_score(clf, x_train, y_train, x_test, y_test)
from sklearn.model_selection import GridSearchCV

from pprint import pprint



n_estimators = [100]



max_features = ['auto', 'sqrt', 'log2', ]

criterion = ['gini', 'entropy', ]

max_depth = [82]

min_samples_split = [8]

min_samples_leaf = [4]

bootstrap = [True]

min_weight_fraction_leaf = [0]

# Create the random grid

param_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'criterion': criterion,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap,

               'min_weight_fraction_leaf': min_weight_fraction_leaf

             }

pprint(param_grid)



df_train = read_csv('train.csv', index_col='ID')



clf = RandomForestClassifier(n_jobs=-1, 

                             random_state=RANDOM_STATE,

                             n_estimators=N_ESTIMATORS,

                             verbose=False)

x_train, x_test, y_train, y_test = split_train_test(df_train, 'default payment next month')



scoring = metrics.make_scorer(metrics.roc_auc_score)



grid_search = GridSearchCV(estimator = clf, scoring = scoring, param_grid = param_grid, cv = 5, n_jobs = -1, verbose = 2)

grid_search.fit(x_train, y_train)



best_params = {

    **{

        'n_jobs': -1, 

        'random_state': RANDOM_STATE,

        'verbose': False,

    },

    **grid_search.best_params_,

}

pprint(best_params)

clf = RandomForestClassifier(**best_params)



df_train = read_csv('train.csv', index_col='ID')

x_train, x_test, y_train, y_test = split_train_test(df_train, 'default payment next month')

clf.fit(x_train, y_train)

fitpredict_score(clf, x_train, y_train, x_test, y_test)
import xgboost as xgb



params = {}

params['objective'] = 'binary:logistic'

params['silent'] = True

params['eval_metric'] = 'auc'

params['random_state'] = RANDOM_STATE

df_train = read_csv('train.csv', index_col='ID')

x_train, x_test, y_train, y_test = split_train_test(df_train, 'default payment next month')

dtrain = xgb.DMatrix(x_train, y_train)

dtest = xgb.DMatrix(x_test, y_test)



watchlist = [(dtrain, 'train'), (dtest, 'test')]

ROUNDS = 1000

EARLY_STOP = 50

model = xgb.train(params, 

                dtrain, 

                ROUNDS, 

                watchlist, 

                early_stopping_rounds=EARLY_STOP, 

                maximize=True, 

                verbose_eval=0)



y_predict_train = model.predict(xgb.DMatrix(x_train))

y_predict_test = model.predict(xgb.DMatrix(x_test))
thresholds = np.linspace(0, 1, num = 10000)

#print(f'thresholds = {thresholds}')

aucs_train = []

aucs_test = []

max_auc_train = -1

best_t_train = -1



max_auc_test = -1

best_t_test = -1



for t in thresholds:

    ytrain = np.where(y_predict_train >= t, 1, 0)

    auc_train = metrics.roc_auc_score(y_train, ytrain)

    aucs_train.append(auc_train)

    if auc_train > max_auc_train:

        max_auc_train = auc_train

        best_t_train = t

    

    ytest = np.where(y_predict_test >= t, 1, 0)

    auc_test = metrics.roc_auc_score(y_test, ytest)

    aucs_test.append(auc_test)

    if auc_test > max_auc_test:

        max_auc_test = auc_test

        best_t_test = t

#print(f'aucs = {aucs}')

print(f'max_auc_train = {max_auc_train}, t = {best_t_train}')

print(f'max_auc_test = {max_auc_test}, t = {best_t_test}')



#print(f'max_thresh = {max_thresh}')



f, ax = plt.subplots(1, 1, figsize = (15, 7))

line, = ax.plot(thresholds, aucs_train, label='Treino')

plt.axvline(x=best_t_train, color=line.get_color())



line, = ax.plot(thresholds, aucs_test, label='Teste')

plt.axvline(x=best_t_test, color=line.get_color())

ax.legend(fontsize=20)

plt.title("Threshold ótimo para maximizar AUC", fontsize=30)

plt.xlabel("Threshold", fontsize=20)

plt.ylabel("AUC", fontsize=20)
def validate_submission():

    submission_shape = (9000, 2)

    df = pd.read_csv('submission.csv')

    assert df.shape == submission_shape

    assert df['ID'].nunique() == submission_shape[0]

    

    display(df['Default'].unique())

    assert set(df['Default'].values) == set([0, 1])

    

    valid = read_csv('valid.csv', index_col='ID')

    test = read_csv('test.csv', index_col='ID')

    inp = pd.concat([valid, test])

    ids = set(inp.index)

    assert ids == set(df['ID'].values)



def get_submission(model, df_input, indexes):

    THRESHOLD = (best_t_train + best_t_test) / 2.0

    submission_shape = (9000, 2)

    df_submission = pd.DataFrame()

    df_submission['ID'] = indexes

    y_predict = model.predict(df_input)

    y_predict = np.where(y_predict >= THRESHOLD, 1, 0)

    df_submission['Default'] = y_predict

    assert df_submission.shape == submission_shape

    df_submission.to_csv('submission.csv', index=False)

    validate_submission()

    print('Submission is ready to go!')

    

df_train = read_csv('train.csv', index_col='ID')

ycol = 'default payment next month'

x_train = df_train.drop(ycol, axis=1)

y_train = df_train[ycol]

dtrain = xgb.DMatrix(x_train, y_train)



params = {}

params['objective'] = 'binary:logistic'

params['silent'] = True

params['eval_metric'] = 'auc'

params['random_state'] = RANDOM_STATE



watchlist = [(dtrain, 'train'), ]

model = xgb.train(params, 

                dtrain, 

                ROUNDS, 

                watchlist, 

                early_stopping_rounds=EARLY_STOP, 

                maximize=True, 

                verbose_eval=0)

df_valid = read_csv('valid.csv', index_col='ID')

df_test = read_csv('test.csv', index_col='ID')

df_input = pd.concat([df_valid, df_test])

indexes = df_input.index

display(df_input.shape)

df_input = xgb.DMatrix(df_input)

get_submission(model, df_input, indexes)