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
train_df = pd.read_csv('/kaggle/input/1056lab-airbnb-price-predction/train.csv', index_col=0)

test_df = pd.read_csv('/kaggle/input/1056lab-airbnb-price-predction/test.csv', index_col=0)
train_df
test_df
df = pd.concat([train_df.drop('price', axis=1), test_df])

df
df = pd.concat([df, pd.get_dummies(df['neighbourhood_group'])], axis=1)

df = df.drop('neighbourhood_group', axis=1)

df
df = pd.concat([df, pd.get_dummies(df['neighbourhood'])], axis=1)

df = df.drop('neighbourhood', axis=1)

df
df = pd.concat([df, pd.get_dummies(df['room_type'])], axis=1)

df = df.drop('room_type', axis=1)

df
df = df.drop(['name', 'host_id', 'host_name', 'last_review', 'reviews_per_month'], axis=1)

df
nrow, ncol = train_df.shape

price_df = train_df[['price']]

train_df = df[:nrow]

train_df = pd.concat([train_df, price_df], axis=1)

train_df
nrow, ncol = train_df.shape

test_df = df[nrow:]

test_df
X_train = train_df.drop(['price'], axis=1).to_numpy()

y_train = train_df['price'].to_numpy()
import xgboost as xgb

from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split



X_train_t, X_valid, y_train_t, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
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

    xgboost_tuna.fit(X_train_t,y_train_t)

    tuna_pred_test = xgboost_tuna.predict(X_valid)

    for i in range(len(tuna_pred_test)):

        if tuna_pred_test[i] < 0:

            tuna_pred_test[i]=0

    return np.sqrt(mean_squared_error(y_valid, tuna_pred_test))
# 最適化

study = optuna.create_study()

study.optimize(opt, n_trials=10)
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

fin_xgboost.fit(X_train, y_train)
X = test_df.to_numpy()



p = fin_xgboost.predict(X)
submit_df = pd.read_csv('/kaggle/input/1056lab-airbnb-price-predction/sampleSubmission.csv')

submit_df['price'] = p

submit_df.to_csv('submission.csv', index=False)