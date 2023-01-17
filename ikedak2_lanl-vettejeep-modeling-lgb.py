import os

import time

import warnings

import traceback

import numpy as np

import pandas as pd

from scipy import stats

import scipy.signal as sg

import multiprocessing as mp

from scipy.signal import hann

from scipy.signal import hilbert

from scipy.signal import convolve

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import StandardScaler



import xgboost as xgb

import lightgbm as lgb

from sklearn.model_selection import KFold

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error



from tqdm import tqdm_notebook

warnings.filterwarnings("ignore")
print(os.listdir("../input"))

print(os.listdir("../input/lanl-vettejeep-join"))
DATA_DIR = '../input/LANL-Earthquake-Prediction'

TRAIN_DIR = '../input/lanl-vettejeep-join'

TEST_DIR = '../input/lanl-vettejeep-join'

# TRAIN_DIR = '../input/vettejeep-train-1'

# TEST_DIR = '../input/vettejeep-test-1'
# submission = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'), index_col='seg_id')

submission = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))

scaled_train_X = pd.read_csv(os.path.join(TRAIN_DIR, 'scaled_train_X.csv'))

scaled_test_X = pd.read_csv(os.path.join(TEST_DIR, 'scaled_test_X.csv'))

train_y = pd.read_csv(os.path.join(TRAIN_DIR, 'train_y.csv'))
print(submission.shape)

print(scaled_train_X.shape)

print(scaled_test_X.shape)

print(train_y.shape)
# No need for '../input/lanl-vettejeep-join'

# scaled_test_X.drop(['time_to_failure'], axis=1, inplace=True)
params = {'num_leaves': 21,

         'min_data_in_leaf': 20,

         'objective':'regression',

         'learning_rate': 0.01,

         'max_depth': 108,

         "boosting": "gbdt",

         "feature_fraction": 0.91,

         "bagging_freq": 1,

         "bagging_fraction": 0.91,

         "bagging_seed": 42,

         "metric": 'mae',

         "lambda_l1": 0.1,

         "verbosity": -1,

         "random_state": 42}



# From my kernel

# params = {'num_leaves': 51,

#          'min_data_in_leaf': 10, 

#          'objective':'regression',

#          'max_depth': -1,

#          'learning_rate': 0.001,

#          "boosting": "gbdt",

#          "feature_fraction": 0.91,

#          "bagging_freq": 1,

#          "bagging_fraction": 0.91,

#          "bagging_seed": 42,

#          "metric": 'mae',

#          "lambda_l1": 0.1,

#          "verbosity": -1,

#          "nthread": -1,

#          "random_state": 42}
# TEMP

# n_fold = 8

# folds = KFold(n_splits=n_fold, shuffle=True, random_state=42)



# for fold_, (trn_idx, val_idx) in enumerate(folds.split(scaled_train_X, train_y.values)):

#         print('working fold %d' % fold_)

#         strLog = "fold {}".format(fold_)

#         print(trn_idx[:10])

#         print(val_idx[:10])

        
def lgb_base_model():

    maes = []

    rmses = []

    predictions = np.zeros(len(scaled_test_X))



    n_fold = 8

    folds = KFold(n_splits=n_fold, shuffle=True, random_state=42)



    fold_importance_df = pd.DataFrame()

    fold_importance_df["Feature"] = scaled_train_X.columns



    for fold_, (trn_idx, val_idx) in enumerate(folds.split(scaled_train_X, train_y.values)):

        print('working fold %d' % fold_)

        strLog = "fold {}".format(fold_)

        print(strLog)



        X_tr, X_val = scaled_train_X.iloc[trn_idx], scaled_train_X.iloc[val_idx]

        y_tr, y_val = train_y.iloc[trn_idx], train_y.iloc[val_idx]



        model = lgb.LGBMRegressor(**params, n_estimators=80000, n_jobs=-1)

        model.fit(X_tr, y_tr,

                  eval_set=[(X_tr, y_tr), (X_val, y_val)], eval_metric='mae',

                  verbose=1000, early_stopping_rounds=200)



        # predictions

        preds = model.predict(scaled_test_X, num_iteration=model.best_iteration_)

        predictions += preds / folds.n_splits

        preds = model.predict(X_val, num_iteration=model.best_iteration_)



        # mean absolute error

        mae = mean_absolute_error(y_val, preds)

        print('MAE: %.6f' % mae)

        maes.append(mae)



        # root mean squared error

        rmse = mean_squared_error(y_val, preds)

        print('RMSE: %.6f' % rmse)

        rmses.append(rmse)



        fold_importance_df['importance_%d' % fold_] = model.feature_importances_[:len(scaled_train_X.columns)]



    print('MAEs', maes)

    print('MAE mean: %.6f' % np.mean(maes))

    print('RMSEs', rmses)

    print('RMSE mean: %.6f' % np.mean(rmses))



    submission.time_to_failure = predictions

    submission.to_csv('submission.csv', index=False)

    fold_importance_df.to_csv('fold_imp.csv')  # index needed, it is seg id



# lgb_base_model()
# Verify

# temp = pd.read_csv('submission.csv')

# display(temp.head(5))
params = {'num_leaves': 21,

         'min_data_in_leaf': 20,

         'objective':'regression',

         'max_depth': 108,

         'learning_rate': 0.001,

         "boosting": "gbdt",

         "feature_fraction": 0.91,

         "bagging_freq": 1,

         "bagging_fraction": 0.91,

         "bagging_seed": 42,

         "metric": 'mae',

         "lambda_l1": 0.1,

         "verbosity": -1,

         "random_state": 42}





def lgb_trimmed_model():

    maes = []

    rmses = []

    tr_maes = []

    tr_rmses = []



    pcol = []

    pcor = []

    pval = []

    

    train_y = pd.read_csv(os.path.join(TRAIN_DIR, 'train_y.csv'))

    y = train_y['time_to_failure'].values



    for col in scaled_train_X.columns:

        pcol.append(col)

        pcor.append(abs(stats.pearsonr(scaled_train_X[col], y)[0]))

        pval.append(abs(stats.pearsonr(scaled_train_X[col], y)[1]))



    df = pd.DataFrame(data={'col': pcol, 'cor': pcor, 'pval': pval}, index=range(len(pcol)))

    df.sort_values(by=['cor', 'pval'], inplace=True)

    df.dropna(inplace=True)

    df = df.loc[df['pval'] <= 0.05]



    drop_cols = []



    for col in scaled_train_X.columns:

        if col not in df['col'].tolist():

            drop_cols.append(col)



    scaled_train_X.drop(labels=drop_cols, axis=1, inplace=True)

    scaled_test_X.drop(labels=drop_cols, axis=1, inplace=True)



    

    predictions = np.zeros(len(scaled_test_X))

    preds_train = np.zeros(len(scaled_train_X))



    print('shapes of train and test:', scaled_train_X.shape, scaled_test_X.shape)



    n_fold = 6

    folds = KFold(n_splits=n_fold, shuffle=False, random_state=42)



    for fold_, (trn_idx, val_idx) in enumerate(folds.split(scaled_train_X, train_y.values)):

        print('working fold %d' % fold_)

        strLog = "fold {}".format(fold_)

        print(strLog)



        X_tr, X_val = scaled_train_X.iloc[trn_idx], scaled_train_X.iloc[val_idx]

        y_tr, y_val = train_y.iloc[trn_idx], train_y.iloc[val_idx]



        model = lgb.LGBMRegressor(**params, n_estimators=60000, n_jobs=-1)

        model.fit(X_tr, y_tr,

                  eval_set=[(X_tr, y_tr), (X_val, y_val)], eval_metric='mae',

                  verbose=1000, early_stopping_rounds=200)



        # model = xgb.XGBRegressor(n_estimators=1000,

        #                                learning_rate=0.1,

        #                                max_depth=6,

        #                                subsample=0.9,

        #                                colsample_bytree=0.67,

        #                                reg_lambda=1.0, # seems best within 0.5 of 2.0

        #                                # gamma=1,

        #                                random_state=777+fold_,

        #                                n_jobs=12,

        #                                verbosity=2)

        # model.fit(X_tr, y_tr)



        # predictions

        preds = model.predict(scaled_test_X)  #, num_iteration=model.best_iteration_)

        predictions += preds / folds.n_splits

        preds = model.predict(scaled_train_X)  #, num_iteration=model.best_iteration_)

        preds_train += preds / folds.n_splits



        preds = model.predict(X_val)  #, num_iteration=model.best_iteration_)



        # mean absolute error

        mae = mean_absolute_error(y_val, preds)

        print('MAE: %.6f' % mae)

        maes.append(mae)



        # root mean squared error

        rmse = mean_squared_error(y_val, preds)

        print('RMSE: %.6f' % rmse)

        rmses.append(rmse)



        # training for over fit

        preds = model.predict(X_tr)  #, num_iteration=model.best_iteration_)



        mae = mean_absolute_error(y_tr, preds)

        print('Tr MAE: %.6f' % mae)

        tr_maes.append(mae)



        rmse = mean_squared_error(y_tr, preds)

        print('Tr RMSE: %.6f' % rmse)

        tr_rmses.append(rmse)



    print('MAEs', maes)

    print('MAE mean: %.6f' % np.mean(maes))

    print('RMSEs', rmses)

    print('RMSE mean: %.6f' % np.mean(rmses))



    print('Tr MAEs', tr_maes)

    print('Tr MAE mean: %.6f' % np.mean(tr_maes))

    print('Tr RMSEs', rmses)

    print('Tr RMSE mean: %.6f' % np.mean(tr_rmses))



    submission.time_to_failure = predictions

    submission.to_csv('submission_2.csv')  # index needed, it is seg id



    pr_tr = pd.DataFrame(data=preds_train, columns=['time_to_failure'], index=range(0, preds_train.shape[0]))

    pr_tr.to_csv(r'preds_tr_xgb_slope_pearson_6fold.csv', index=False)

    print('Train shape: {}, Test shape: {}, Y shape: {}'.format(scaled_train_X.shape, scaled_test_X.shape, train_y.shape))

 
lgb_trimmed_model()