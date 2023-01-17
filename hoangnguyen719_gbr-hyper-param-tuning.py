import os, math, re



from datetime import datetime as dt

from itertools import product

import matplotlib.pyplot as plt

import matplotlib.style as style

import numpy as np

import pandas as pd

import seaborn as sns

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.compose import ColumnTransformer

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_squared_error, make_scorer

from sklearn.model_selection import cross_val_score, GridSearchCV

from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.pipeline import Pipeline

from tqdm import tqdm

import warnings



warnings.filterwarnings("ignore")

%matplotlib inline

pd.set_option('display.max_colwidth', -1)
P_PATH = '../input/house-prices-preprocessed-feature-selected/'

B_PATH = '../input/house-prices-advanced-regression-techniques/'

def reload(df, processed=True, index=False):

    path = P_PATH if processed else B_PATH

    return pd.read_csv(path + df)

train = reload('train_preprocessed.csv')

X_test = reload('test_preprocessed.csv')

X_train = train[[c for c in train.columns if c!='SalePrice']]

y_train = train.SalePrice.copy()

y_trans = np.log1p(y_train)

base_model = GradientBoostingRegressor(

    loss='lad'

    , learning_rate=0.1

    , n_estimators=5000

    , subsample=1

    , min_samples_split=10

    , min_samples_leaf=1

    , max_depth=2

    , max_features=0.4

    , n_iter_no_change=None

    , random_state=713

)



def RMSE(y_true, y_pred):

    return -math.sqrt(mean_squared_error(y_true, y_pred))

rmse_scorer = make_scorer(RMSE)

def timer(t):

    t = dt.now() - t

    return t.seconds + t.microseconds/1e6

def grid(params

         , model=base_model

         , X=X_train, y=y_trans, scorer=rmse_scorer

        ):

    complexity = np.product([len(params[l]) for l in params])

    print(f'Grid search complexity: {complexity}')

    grid = GridSearchCV(model, params, scoring=scorer

                        , n_jobs=-1, cv=3, return_train_score=True)

    t = dt.now()

    grid.fit(X, y)

    print(f'Search time: {timer(t)} secs')

    out = pd.DataFrame(grid.cv_results_)

    out = out.sort_values(by='rank_test_score').reset_index(drop=True)

    out['mean_test_scr_delt_pct'] = out.mean_test_score.pct_change(-1)*100

    return out
params = dict(

    learning_rate = [0.075, 0.08, 0.085, 0.095, 0.1]

    , n_estimators = np.logspace(3.4,3.9,6).astype(int) #array([2511, 3162, 3981, 5011, 6309, 7943])

    , max_features = [0.37, 0.39, 0.4, 0.41]

    , min_samples_split = [8, 9, 10]

    , max_depth = [2, 3]

    , min_samples_leaf = [1, 2]

)

test_param = params.keys()

test_params = {}

for p in test_param:

    test_params[p] = params[p]



output = grid(test_params)

attrbts = ['params', 'mean_fit_time', 'mean_train_score', 'mean_test_score', 'mean_test_scr_delt_pct']

output[attrbts].iloc[:20,:].round(4)
output.drop(columns=[

    'std_fit_time', 'std_score_time', 'split0_test_score'

    , 'split1_test_score', 'split2_test_score', 'std_test_score'

    , 'split0_train_score', 'split1_train_score', 'split2_train_score'

    , 'std_train_score'

], inplace=True)

output.to_csv('gridsearch_result.csv', index=False)