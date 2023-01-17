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
df_train = pd.read_csv('/kaggle/input/1056lab-football-players-value-prediction/train.csv', index_col=0)

df_test = pd.read_csv('/kaggle/input/1056lab-football-players-value-prediction/test.csv', index_col=0)
df_train
df_test
all_df = pd.concat([df_train.drop('value_eur', axis=1), df_test])

all_df
columns = all_df.columns

for c in columns:

    all_df[c] = pd.to_numeric(all_df[c], errors='ignore')

all_df
all_df.dtypes
columns = all_df.columns

for c in columns:

    if all_df[c].isna().any():

        if all_df[c].dtypes != np.object:

            median = all_df[c].median()

            all_df[c] = all_df[c].replace(np.NaN, median)

        else:

            mfv = all_df[c].mode()[0]

            all_df[c] = all_df[c].replace(np.NaN, mfv)
columns = all_df.columns

for c in columns:

    if all_df[c].dtypes == np.object:

        all_df = pd.concat([all_df, pd.get_dummies(all_df[[c]])], axis=1)

        all_df = all_df.drop(c, axis=1)

all_df
X_train_all = all_df[:len(df_train)].to_numpy()

y_train_all = df_train['value_eur'].to_numpy()



X_test_all = all_df[len(df_train):].to_numpy()
import xgboost as xgb

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_log_error

X_train, X_valid, y_train, y_valid = train_test_split(X_train_all, y_train_all, test_size=0.2, random_state=0)
def root_mean_squared_log_error(real, predicted):

    sum=0.0

    for x in range(len(predicted)):

        if predicted[x]<0 or real[x]<0: # check for negative values

            continue

        p = np.log(predicted[x]+1)

        r = np.log(real[x]+1)

        sum = sum + (p - r)**2

    return (sum/len(predicted))**0.5
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

    reg_lamda = trial.suggest_int('reg_lamda',1,10)

    xgboost_tuna = XGBRegressor(

        random_state=3,

        learning_rate = learning_rate,

        n_estimators = n_estimators,

        max_depth = max_depth,

        min_child_weight = min_child_weight,

        subsample = subsample,

        colsample_bytree = colsample_bytree,

        reg_alpha = reg_alpha,

        reg_lamda = reg_lamda,

    )

    xgboost_tuna.fit(X_train,y_train)

    tuna_pred_test = xgboost_tuna.predict(X_valid)

    return root_mean_squared_log_error(y_valid, tuna_pred_test)
study = optuna.create_study()

study.optimize(opt, n_trials=3)
l_r = study.best_params['learning_rate'] #learning_rateの最適値

n_e = study.best_params['n_estimators']  # n_estimatorsの最適値

m_d = study.best_params['max_depth'] # max_depthの最適値

m_c_w = study.best_params['min_child_weight'] # min_child_weightの最適値

subsam = study.best_params['subsample'] #subsampleの最適値

col_by = study.best_params['colsample_bytree']  #colsample_bytreeの最適値

r_a = study.best_params['reg_alpha'] #leg_alphaの最適値

r_l = study.best_params['reg_lamda'] #leg_lamdaの最適値

# 最適なハイパーパラメータを設定

fin_xgboost = XGBRegressor(

        random_state=3,

        learning_rate = l_r,

        n_estimators = n_e,

        max_depth = m_d,

        min_child_weight = m_c_w,

        subsample = subsam,

        colsample_bytree = col_by,

        reg_alpha = r_a,

        reg_lamda = r_l,

    )

# モデル訓練

fin_xgboost.fit(X_train_all, y_train_all)
p_test = fin_xgboost.predict(X_test_all)
submit_df = pd.read_csv('/kaggle/input/1056lab-football-players-value-prediction/sampleSubmission.csv', index_col=0)

submit_df['value_eur'] = p_test

submit_df
submit_df.to_csv('submission.csv')