# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv('/kaggle/input/1056lab-football-players-value-prediction/train.csv',index_col=0)

test_df = pd.read_csv('/kaggle/input/1056lab-football-players-value-prediction/test.csv',index_col=0)
train_df
test_df
import seaborn as sns

from matplotlib import pyplot

import matplotlib.pyplot as plt



sns.set_style("darkgrid")

pyplot.figure(figsize=(34, 34))  # 図の大きさを大き目に設定

sns.heatmap(train_df.corr(), square=True, annot=True)  # 相関係数でヒートマップを作成
train_df.dtypes
train_df.isnull().sum()
test_df.isnull().sum()
train_df2=train_df[['overall','potential','international_reputation','skill_moves','attacking_short_passing','skill_long_passing','skill_ball_control','movement_reactions','power_shot_power','mentality_vision','mentality_composure','value_eur']]

test_df2=test_df[['overall','potential','international_reputation','skill_moves','attacking_short_passing','skill_long_passing','skill_ball_control','movement_reactions','power_shot_power','mentality_vision','mentality_composure']]
train_df2
test_df2
train_df2
import xgboost as xgb

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_log_error

train_dfx = train_df2.drop('value_eur', axis=1).values

train_dfy = train_df2['value_eur'].values

X_train, X_valid, y_train, y_valid = train_test_split(train_dfx, train_dfy, test_size=0.2, random_state=0)
import optuna

# Objective Functionの作成

def opt(trial):

    learning_rate = trial.suggest_uniform('learning_rate',0.01,0.3)

    n_estimators = trial.suggest_int('n_estimators', 0, 1000)

    max_depth = trial.suggest_int('max_depth', 1, 30)

    min_child_weight = trial.suggest_int('min_child_weight', 1, 20)

    subsample = trial.suggest_discrete_uniform('subsample', 0.5, 0.9, 0.1)

    colsample_bytree = trial.suggest_discrete_uniform('colsample_bytree', 0.5, 0.9, 0.1)

    reg_alpha = trial.suggest_int('reg_alpha',0,10)

    xgboost_tuna = XGBRegressor(

        random_state=42,

        learning_rate = learning_rate,

        n_estimators = n_estimators,

        max_depth = max_depth,

        min_child_weight = min_child_weight,

        subsample = subsample,

        colsample_bytree = colsample_bytree,

        reg_alpha = reg_alpha,

    )

    xgboost_tuna.fit(X_train,y_train)

    tuna_pred_test = xgboost_tuna.predict(X_valid)

    for i in range(len(tuna_pred_test)):

        if tuna_pred_test[i] < 0:

            tuna_pred_test[i]=0

    return mean_squared_log_error(y_valid, tuna_pred_test)
# 最適化

study = optuna.create_study()

study.optimize(opt, n_trials=50)
print(study.best_params)

print(study.best_value)

print(study.best_trial)
l_r = study.best_params['learning_rate'] #learning_rateの最適値

n_e = study.best_params['n_estimators']  # n_estimatorsの最適値

m_d = study.best_params['max_depth'] # max_depthの最適値

m_c_w = study.best_params['min_child_weight'] # min_child_weightの最適値

subsam = study.best_params['subsample'] #subsampleの最適値

col_by = study.best_params['colsample_bytree']  #colsample_bytreeの最適値

r_a = study.best_params['reg_alpha'] #leg_alphaの最適値

# 最適なハイパーパラメータを設定

fin_xgboost = XGBRegressor(

        random_state=42,

        learning_rate = l_r,

        n_estimators = n_e,

        max_depth = m_d,

        min_child_weight = m_c_w,

        subsample = subsam,

        colsample_bytree = col_by,

        reg_alpha = r_a,

    )

# モデル訓練

fin_xgboost.fit(train_dfx, train_dfy)
test_dfx = test_df2.values

predict = fin_xgboost.predict(test_dfx)
submit_df = pd.read_csv('/kaggle/input/1056lab-football-players-value-prediction/sampleSubmission.csv', index_col=0)

submit_df['value_eur'] = predict

submit_df.to_csv('Submission4.csv')