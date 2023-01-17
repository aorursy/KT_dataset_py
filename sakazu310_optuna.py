import numpy as np
import math
import scipy as sp
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
%matplotlib inline

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
##
import optuna
##
import lightgbm as lgb
df_train = pd.read_csv('../input/homework-for-students4plus/train.csv', index_col=0, parse_dates=['issue_d'])
df_test  = pd.read_csv('../input/homework-for-students4plus/test.csv' , index_col=0, parse_dates=['issue_d'])
df_train.head()
df_train.reset_index(drop=True,inplace=True)
X_train = df_train
X_test  = df_test.copy()
y_train = X_train["loan_condition"]
X_train = X_train.drop("loan_condition",axis=1)
## 時刻
date_col = ['issue_d']
## 言語
lang_col = ['emp_title','title']
## カテゴリ
cats_col = []
for col in X_train.columns:
    if X_train[col].dtype == 'object':
        cats_col.append(col)
cats_col = list(set(cats_col) - set(lang_col))
## 数値
num_col = list(X_train.columns)
num_col = list(set(num_col) - set(date_col))
num_col = list(set(num_col) - set(lang_col))
num_col = list(set(num_col) - set(cats_col))
## 確認
print("date_col : " + str(len(date_col)))
print("lang_col : " + str(len(lang_col)))
print("cats_col : " + str(len(cats_col)))
print("num_col  : " + str(len(num_col)))
print("total    : " + str(len(X_train.columns)))
## 時刻
X_train_date = X_train[date_col].copy()
X_test_date  = X_test[date_col].copy()
## 言語
X_train_lang = X_train[lang_col].copy()
X_test_lang  = X_test[lang_col].copy()
## カテゴリ
X_train_cats = X_train[cats_col].copy()
X_test_cats  = X_test[cats_col].copy()
## 数値
X_train_num = X_train[num_col].copy()
X_test_num  = X_test[num_col].copy()
X_train_num.fillna(X_train_num.median(),axis=0, inplace=True)
X_test_num.fillna(X_train_num.median() ,axis=0, inplace=True)
X_train_cats.head()
cats_col
target = 'loan_condition'
X_temp = pd.concat([X_train_cats, y_train], axis=1)

for col in cats_col:
    print(col)
    # X_testはX_trainでエンコーディングする
    summary = X_temp.groupby([col])[target].mean()
    X_test_cats[col] = X_test_cats[col].map(summary) 


    # X_trainのカテゴリ変数をoofでエンコーディングする
    skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)
    enc_train = pd.Series(np.zeros(len(X_train)), index=X_train.index)

    for i, (train_ix, val_ix) in enumerate((skf.split(X_train, y_train))):
        X_train_, _ = X_temp.iloc[train_ix], y_train.iloc[train_ix]
        X_val, _ = X_temp.iloc[val_ix], y_train.iloc[val_ix]

        summary = X_train_.groupby([col])[target].mean()
        enc_train.iloc[val_ix] = X_val[col].map(summary)
        
    X_train_cats[col]  = enc_train
# 確認
X_train_cats.head()
# 確認
X_test_cats.head()
# 欠損処理
X_train_cats.fillna(X_train_cats.median(),axis=0, inplace=True)
X_test_cats.fillna(X_train_cats.median() ,axis=0, inplace=True)
X_train = pd.concat([X_train_date,X_train_cats, X_train_num], axis=1)
X_test  = pd.concat([X_test_cats , X_test_num], axis=1)
df_concat = pd.concat([X_train,y_train],axis=1)
df_concat = df_concat.set_index("issue_d")
inner_train = df_concat[:"2015-06-01"].reset_index()
inner_val   = df_concat["2015-06-01":].reset_index()
inner_train["issue_d"].max()
X_inner_train = inner_train.drop(["issue_d","loan_condition"],axis=1)
y_inner_train = inner_train["loan_condition"]
X_inner_val   = inner_val.drop(["issue_d","loan_condition"],axis=1)
y_inner_val   = inner_val["loan_condition"]
def objectives(trial):
    ## ハイパーパラメータの空間を適当に設定
    params = {
        'num_leaves'       : trial.suggest_int('num_leaves', 2, 256),
        'min_child_samples': int(trial.suggest_loguniform('min_child_samples', 100, 10000)),
        'min_child_weight' : trial.suggest_loguniform('min_child_weight', 0.1, 2000),
        'subsample'        : trial.suggest_uniform('subsample', 0.8, 1),
        'colsample_bytree' : trial.suggest_uniform('colsample_bytree', 0.5, 1),
        'learning_rate'    : trial.suggest_loguniform('learning_rate', 0.025, 0.5),
        'min_data_in_leaf' : int(trial.suggest_loguniform('min_data_in_leaf', 1, 1000)),
    }

    # LightGBMで学習
    model = lgb.LGBMRegressor(**params)
    model.fit(X_inner_train, y_inner_train,
              eval_set=(X_inner_val,y_inner_val),
              early_stopping_rounds=20,
              eval_metric='auc',
              verbose=False)

    # 検証用データのスコア
    y_inner_pred = model.predict(X_inner_val)
    score = roc_auc_score(y_inner_val,y_inner_pred)
    
    return score
%%time
# optunaによる最適化呼び出し
opt = optuna.create_study(direction='maximize',sampler=optuna.samplers.RandomSampler(seed=71))
opt.optimize(objectives, n_trials=20)
optuna.visualization.plot_optimization_history(opt)
# 最適パラメータ取得
trial = opt.best_trial
params_best = dict(trial.params.items())
params_best['random_seed'] = 71
params_best["num_leaves"] = int(params_best["num_leaves"])
params_best["min_data_in_leaf"] = int(params_best["min_data_in_leaf"])
# 最適パラメータで学習
model_best = lgb.LGBMRegressor(**params_best)
model_best.fit(X_inner_train, y_inner_train,
               eval_set=(X_inner_val,y_inner_val),
               early_stopping_rounds=20,
               verbose=False)    
y_pred = model_best.predict(X_test)
submission = pd.read_csv('../input/homework-for-students4plus/sample_submission.csv', index_col=0)

submission.loan_condition = y_pred
submission.to_csv('submission.csv')