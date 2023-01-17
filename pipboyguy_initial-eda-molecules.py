!pip install --upgrade pip;pip install git+https://github.com/molmod/molmod;pip install pubchempy;pip install auto_ml
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.options.display.max_columns = None

import os

from molmod import Molecule

import pubchempy as pcp



import lightgbm as lgb

import xgboost as xgb

from catboost import CatBoostRegressor, CatBoostClassifier

import keras



from auto_ml import Predictor

from auto_ml.utils_models import load_ml_model



import seaborn as sns

sns.set()

import matplotlib.pyplot as plt

%matplotlib inline



import os

print(os.listdir("../input"))



import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

warnings.filterwarnings("ignore", category=UserWarning)

warnings.filterwarnings("ignore", category=RuntimeWarning)

warnings.filterwarnings("ignore", category=FutureWarning)
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

structures = pd.read_csv('../input/structures.csv')

DipoleMovements = pd.read_csv('../input/dipole_moments.csv')

magnetic_shielding_tensors = pd.read_csv('../input/magnetic_shielding_tensors.csv')

mulliken_charges = pd.read_csv('../input/mulliken_charges.csv')

potential_energy = pd.read_csv('../input/potential_energy.csv')

scalar_coupling_contributions = pd.read_csv('../input/scalar_coupling_contributions.csv')

# train_comb = train.iloc[:,:-1].append(test, ignore_index=True)

train_comb= train.copy()
train_comb.shape, train.shape, test.shape
train_comb = train_comb.merge(structures, how='left', left_on=['molecule_name', 'atom_index_0'],\

                                        right_on=['molecule_name', 'atom_index'],copy=False, validate="m:1").drop('atom_index',axis=1)

train_comb = train_comb.merge(structures, how='left', left_on=['molecule_name', 'atom_index_1']\

                                        ,right_on=['molecule_name', 'atom_index'],  suffixes=('_0', '_1'),copy=False, validate="m:1").drop('atom_index',axis=1)
train_comb.shape, train.shape, test.shape
train_comb.head()
train_comb = train_comb.merge(DipoleMovements, how='left',on='molecule_name',validate="m:1")

train_comb.rename(columns={'X': 'dipole_moment_X', 'Y': 'dipole_moment_Y','Z': 'dipole_moment_Z'}, inplace=True)

train_comb.head()
train_comb.shape, train.shape, test.shape
train_comb = train_comb.merge(magnetic_shielding_tensors, how='left', left_on=['molecule_name', 'atom_index_0'],\

                                        right_on=['molecule_name', 'atom_index'],copy=False, validate="m:1").drop('atom_index',axis=1)

train_comb = train_comb.merge(magnetic_shielding_tensors, how='left', left_on=['molecule_name', 'atom_index_1']\

                                        ,right_on=['molecule_name', 'atom_index'],  suffixes=('_0', '_1'),copy=False, validate="m:1").drop('atom_index',axis=1)

train_comb.head()
train_comb.shape, train.shape , test.shape
train_comb = train_comb.merge(mulliken_charges, how='left', left_on=['molecule_name', 'atom_index_0'],\

                                        right_on=['molecule_name', 'atom_index'],copy=False, validate="m:1").drop('atom_index',axis=1)

train_comb = train_comb.merge(mulliken_charges, how='left', left_on=['molecule_name', 'atom_index_1']\

                                        ,right_on=['molecule_name', 'atom_index'],  suffixes=('_0', '_1'),copy=False, validate="m:1").drop('atom_index',axis=1)

train_comb.head()
train_comb.shape, train.shape , test.shape
train_comb = train_comb.merge(potential_energy, how='left',on='molecule_name',validate="m:1")

train_comb.head()
train_comb.shape, train.shape , test.shape
train_comb = train_comb.merge(scalar_coupling_contributions, how='left', left_on=['molecule_name', 'atom_index_0','atom_index_1'],\

                                        right_on=['molecule_name', 'atom_index_0','atom_index_1'],copy=False, validate="1:1").drop('type_x',axis=1)

train_comb.rename(columns={'type_y': 'type'},inplace=True)
train_comb.head()
train_comb.shape, train.shape , test.shape
train_comb['#Atoms_in_Molecule'] = train_comb.groupby(['molecule_name'])['id'].transform('count')
train_comb.head()
xyzFiles = os.listdir('../input/structures/')
Molnames = [name[:-4] for name in xyzFiles]

molecule_formula_names = []



for name in xyzFiles:

    

    mol = Molecule.from_file("../input/structures/"+name)

    molecule_formula_names.append(mol.chemical_formula)

    

Molnames_df = pd.DataFrame({'Name' : Molnames, 'Chem_formula' : molecule_formula_names})    
Molnames_df.head()
train_comb = train_comb.merge(Molnames_df, how='left',left_on='molecule_name',right_on='Name',validate="m:1")

train_comb.head()
train_comb.drop(['id','molecule_name','Name'], axis=1, inplace=True)
train_comb.shape, train.shape
train_comb.drop(['atom_index_0','atom_index_1'], axis=1, inplace=True)
train_comb.head()
X = train_comb.iloc[:,1:]

y = train_comb.iloc[:,0]
list(X.dtypes[X.columns != 'Chem_formula'][X.dtypes == 'object'].index.values)
cat_names = list(X.dtypes[X.columns != 'Chem_formula'][X.dtypes == 'object'].index.values)

X = pd.get_dummies(data=X,columns=cat_names, prefix_sep='_')

X.head()
from sklearn.preprocessing import LabelEncoder

cats_to_encode = X.dtypes[X.dtypes == 'object'].index.values

le = LabelEncoder()

X[cats_to_encode] = X[cats_to_encode].apply(lambda col: le.fit_transform(col))



X.head()
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.2, random_state=42)
import os

import time

import datetime

import json

import gc

from numba import jit



import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns

from tqdm import tqdm_notebook

from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold



import lightgbm as lgb

import xgboost as xgb

from catboost import CatBoostRegressor, CatBoostClassifier

from sklearn import metrics



from itertools import product





def eval_auc(y_true, y_pred):

    """

    Fast auc eval function for lgb.

    """

    return 'auc', fast_auc(y_true, y_pred), True





def group_mean_log_mae(y_true, y_pred, types, floor=1e-9):

    """

    Fast metric computation for this competition: https://www.kaggle.com/c/champs-scalar-coupling

    Code is from this kernel: https://www.kaggle.com/uberkinder/efficient-metric

    """

    maes = (y_true-y_pred).abs().groupby(types).mean()

    return np.log(maes.map(lambda x: max(x, floor))).mean()



def train_model_regression(X, X_test, y, params, folds, model_type='lgb', eval_metric='mae', columns=None, plot_feature_importance=False, model=None,

                               verbose=10000, early_stopping_rounds=200, n_estimators=50000):

    """

    A function to train a variety of regression models.

    Returns dictionary with oof predictions, test predictions, scores and, if necessary, feature importances.

    

    :params: X - training data, can be pd.DataFrame or np.ndarray (after normalizing)

    :params: X_test - test data, can be pd.DataFrame or np.ndarray (after normalizing)

    :params: y - target

    :params: folds - folds to split data

    :params: model_type - type of model to use

    :params: eval_metric - metric to use

    :params: columns - columns to use. If None - use all columns

    :params: plot_feature_importance - whether to plot feature importance of LGB

    :params: model - sklearn model, works only for "sklearn" model type

    

    """

    columns = X.columns if columns is None else columns

    X_test = X_test[columns]

    

    # to set up scoring parameters

    metrics_dict = {'mae': {'lgb_metric_name': 'mae',

                        'catboost_metric_name': 'MAE',

                        'sklearn_scoring_function': metrics.mean_absolute_error},

                    'group_mae': {'lgb_metric_name': 'mae',

                        'catboost_metric_name': 'MAE',

                        'scoring_function': group_mean_log_mae},

                    'mse': {'lgb_metric_name': 'mse',

                        'catboost_metric_name': 'MSE',

                        'sklearn_scoring_function': metrics.mean_squared_error}

                    }



    

    result_dict = {}

    

    # out-of-fold predictions on train data

    oof = np.zeros(len(X))

    

    # averaged predictions on train data

    prediction = np.zeros(len(X_test))

    

    # list of scores on folds

    scores = []

    feature_importance = pd.DataFrame()

    

    # split and train on folds

    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):

        print(f'Fold {fold_n + 1} started at {time.ctime()}')

        if type(X) == np.ndarray:

            X_train, X_valid = X[columns][train_index], X[columns][valid_index]

            y_train, y_valid = y[train_index], y[valid_index]

        else:

            X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]

            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

            

        if model_type == 'lgb':

            model = lgb.LGBMRegressor(**params, n_estimators = n_estimators, n_jobs = -1)

            model.fit(X_train, y_train, 

                    eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric=metrics_dict[eval_metric]['lgb_metric_name'],

                    verbose=verbose, early_stopping_rounds=early_stopping_rounds)

            

            y_pred_valid = model.predict(X_valid)

            y_pred = model.predict(X_test, num_iteration=model.best_iteration_)

            

        if model_type == 'xgb':

            train_data = xgb.DMatrix(data=X_train, label=y_train, feature_names=X.columns)

            valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=X.columns)



            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]

            model = xgb.train(dtrain=train_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200, verbose_eval=verbose, params=params)

            y_pred_valid = model.predict(xgb.DMatrix(X_valid, feature_names=X.columns), ntree_limit=model.best_ntree_limit)

            y_pred = model.predict(xgb.DMatrix(X_test, feature_names=X.columns), ntree_limit=model.best_ntree_limit)

        

        if model_type == 'sklearn':

            model = model

            model.fit(X_train, y_train)

            

            y_pred_valid = model.predict(X_valid).reshape(-1,)

            score = metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid)

            print(f'Fold {fold_n}. {eval_metric}: {score:.4f}.')

            print('')

            

            y_pred = model.predict(X_test).reshape(-1,)

        

        if model_type == 'cat':

            model = CatBoostRegressor(iterations=20000,  eval_metric=metrics_dict[eval_metric]['catboost_metric_name'], **params,

                                      loss_function=metrics_dict[eval_metric]['catboost_metric_name'])

            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True, verbose=False)



            y_pred_valid = model.predict(X_valid)

            y_pred = model.predict(X_test)

        

        oof[valid_index] = y_pred_valid.reshape(-1,)

        if eval_metric != 'group_mae':

            scores.append(metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid))

        else:

            scores.append(metrics_dict[eval_metric]['scoring_function'](y_valid, y_pred_valid, X_valid['type']))



        prediction += y_pred    

        

        if model_type == 'lgb' and plot_feature_importance:

            # feature importance

            fold_importance = pd.DataFrame()

            fold_importance["feature"] = columns

            fold_importance["importance"] = model.feature_importances_

            fold_importance["fold"] = fold_n + 1

            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)



    prediction /= folds.n_splits

    

    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

    

    result_dict['oof'] = oof

    result_dict['prediction'] = prediction

    result_dict['scores'] = scores

    

    if model_type == 'lgb':

        if plot_feature_importance:

            feature_importance["importance"] /= folds.n_splits

            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(

                by="importance", ascending=False)[:50].index



            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]



            plt.figure(figsize=(16, 12));

            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));

            plt.title('LGB Features (avg over folds)');

            

            result_dict['feature_importance'] = feature_importance

        

    return result_dict
n_fold = 5

folds = KFold(n_splits=n_fold, shuffle=True, random_state=42)

params = {'objective': 'huber',

          'max_depth': -1,

          'learning_rate': 0.01,

          "boosting": "gbdt",

          "bagging_freq": 5,

          "bagging_seed": 11,

          "metric": 'mae',

          "verbosity": -1,

         }



#test our model framework on first 50000 training samo=ples first and first 10000 test samples





output = train_model_regression(X_train[:50000], X_test[:10000], y_train[:50000], params, folds, model_type='lgb', eval_metric='mae', columns=None, plot_feature_importance=False, model=None,

                               verbose=10000, early_stopping_rounds=200, n_estimators=50000)
# print(metrics.mean_absolute_error(y_test[:10000], prediction_lgb[:10000]))

output
print(f'Validation MAE for LGBM = {metrics.mean_absolute_error(y_test[:10000], output["prediction"][:10000])}')

print(f'Validation MAE for just predicting the mean of test set = {metrics.mean_absolute_error(y_test[:10000], y_test[:10000].mean()*np.ones(y_test[:10000].shape))}')