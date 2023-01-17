import pandas as pd
import matplotlib.pyplot as plt

train=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
sumple=pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')
train.describe()
X_train=train[['MSSubClass','LotFrontage','LotArea','PoolArea','MiscVal','MoSold','YrSold']]
y_train=train['SalePrice']
X_test=test[['MSSubClass','LotFrontage','LotArea','PoolArea','MiscVal','MoSold','YrSold']]
from sklearn.model_selection import train_test_split

X_train,X_valid,y_train,y_valid=train_test_split(X_train,y_train,test_size=0.2,random_state=0)
import lightgbm as lgb
import numpy as np
import sklearn.datasets
import sklearn.metrics
from sklearn.model_selection import train_test_split

import optuna


# FYI: Objective functions can take additional arguments
# (https://optuna.readthedocs.io/en/stable/faq.html#objective-func-additional-args).
def objective(trial):

    dtrain = lgb.Dataset(X_train, label=y_train)

    param = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }

    gbm = lgb.train(param, dtrain)
    preds = gbm.predict(X_valid)

    accuracy = sklearn.metrics.accuracy_score(y_valid, preds)
    return accuracy




    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)
X_train=X_train.fillna(X_train.mean())
y_train=y_train.fillna(y_train.mean())
X_valid=X_valid.fillna(X_valid.mean())
y_valid=y_valid.fillna(y_valid.mean())


X_train=X_train.astype(float)
y_train=y_train.astype(float)
X_valid=X_valid.astype(float)
y_valid=y_valid.astype(float)
from optuna.integration import lightgbm as lgb
dtrain = lgb.Dataset(X_train, label=y_train)
eval_data = lgb.Dataset(X_valid, label=y_valid)

param = {
        'objective': 'regression',
        'metric': 'rmse',
        'lambda_l1': 8.72896788870908e-06,
        'lambda_l2': 5.433642964479813e-07,
        'num_leaves': 2,
        'feature_fraction': 0.5402652675292832,
        'bagging_fraction': 0.5999425893986495,
        'bagging_freq': 4,
        'min_child_samples': 14,
    }

best = lgb.train(param, 
                 dtrain,
                 valid_sets=eval_data,
                 early_stopping_rounds=10)
predict = best.predict(X_test, num_iteration=best.best_iteration)
sumple['SalePrice']=predict
sumple.to_csv('sions.csv', index=False)