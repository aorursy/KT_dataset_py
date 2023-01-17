import pandas as pd
import xgboost as xgb  

train=pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv')
test=pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')

shops=pd.read_csv('../input/competitive-data-science-predict-future-sales/shops.csv')
item=pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')
item_category=pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv')

train['revenue'] = train['item_price'] *  train['item_cnt_day']
train=train.drop('date',axis=1)

X_train=train.drop('item_cnt_day',axis=1)
X_train=X_train.drop('date_block_num',axis=1)
X_train=X_train.drop('item_price',axis=1)

y_train=train['item_cnt_day']

test=test.drop('ID',axis=1)
from sklearn.model_selection import train_test_split  

X_train,X_valid,y_train,y_valid = train_test_split(X_train,y_train, test_size=0.2, shuffle=True)  
dtrain = xgb.DMatrix(X_train, label=y_train)  
dvalid = xgb.DMatrix(X_valid, label=y_valid) 
from sklearn.model_selection import GridSearchCV

print("Parameter optimization")
xgb_model = xgb.XGBRegressor()
reg_xgb = GridSearchCV(xgb_model,
                   {'max_depth': [2,4,6],
                    'n_estimators': [50,100,200]}, verbose=1)

reg_xgb.fit(X_train, y_train)
import xgboost as xgb
# モデルのインスタンス作成
mod = xgb.XGBRegressor()
mod.fit(X_train, y_train)
y_train_pred = reg_xgb.predict(X_train)
import pandas as pd

sumple=pd.read_csv('../input/competitive-data-science-predict-future-sales/sample_submission.csv')
data=pd.DataFrame(y_train_pred)
data=data[:214200]
sumple['item_cnt_month']=data
sumple.to_csv('last.csv',index=False)
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


import lightgbm as lgb
import numpy as np
import sklearn.datasets
import sklearn.metrics
from sklearn.model_selection import train_test_split

import optuna


# FYI: Objective functions can take additional arguments
# (https://optuna.readthedocs.io/en/stable/faq.html#objective-func-additional-args).
def objective(trial):

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
    pred_labels = np.rint(preds)
    accuracy = sklearn.metrics.accuracy_score(y_valid, pred_labels)
    return accuracy


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
import lightgbm as lgb
import numpy as np
import sklearn.datasets
import sklearn.metrics
from sklearn.model_selection import train_test_split

import optuna

lgbm_params = {
    'objective': 'regression'
    
    }

auc_list = []
precision_list = []
recall_list = []

dtrain = lgb.Dataset(X_train, label=y_train)
eval_data = lgb.Dataset(X_valid, label=y_valid)


def objective(trial):

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
    pred_labels = np.rint(preds)
    accuracy = sklearn.metrics.accuracy_score(y_valid, pred_labels)
    return accuracy



study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print('Number of finished trials: {}'.format(len(study.trials)))

print('Best trial:')
trial = study.best_trial

print('  Value: {}'.format(trial.value))

print('  Params: ')
for key, value in trial.params.items():
    print('    {}: {}'.format(key, value))    

    # optunaでサーチしたパラメータ
trial.params['objective'] = 'regression'
lgbm_params = trial.params


    # データセットを生成する
lgb_train = lgb.Dataset(X_train, y_train)

    # モデル評価用
lgb_valid = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

model = lgb.train(lgbm_params, 
                    lgb_train,
                    valid_sets=lgb_valid,
                    num_boost_round=100000,
                    early_stopping_rounds=10)

predict = model.predict(test, num_iteration=model.best_iteration)

data=pd.DataFrame(predict)
data=data[:214200]
sumple['item_cnt_month']=data
sumple.to_csv('last.csv',index=False)