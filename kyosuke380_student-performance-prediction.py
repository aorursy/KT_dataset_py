# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/1056lab-student-performance-prediction/train.csv',index_col=0)

df_test = pd.read_csv('../input/1056lab-student-performance-prediction/test.csv',index_col=0)

df_train
import seaborn as sns

from matplotlib import pyplot

import matplotlib.pyplot as plt



sns.set_style("darkgrid")

pyplot.figure(figsize=(17, 17))  # 図の大きさを大き目に設定

sns.heatmap(df_train.corr(), square=True, annot=True)  # 相関係数でヒートマップを作成
df_train = df_train[['age', 'Medu', 'Fedu', 'studytime', 'failures','higher','G3']]  # 列を選択

df_test = df_test[['age', 'Medu', 'Fedu', 'studytime', 'failures','higher']]  # 列を選択
df_train
df_test
import xgboost as xgb

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

X_train = df_train.drop('G3', axis=1).values

y_train = df_train['G3'].values

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
xgboost_tuna = XGBRegressor(random_state=42)
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

        random_state=42,

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

    return mean_squared_error(y_valid, tuna_pred_test)
# 最適化

study = optuna.create_study()

study.optimize(opt, n_trials=100)
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

r_l = study.best_params['reg_lamda'] #leg_lamdaの最適値

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

        reg_lamda = r_l,

    )

# モデル訓練

fin_xgboost.fit(X_train, y_train)

 

#テストデータで推測値を算出

fin_test_pred = fin_xgboost.predict(X_valid)

mean_squared_error(fin_test_pred, y_valid)
model = XGBRegressor(

    random_state=42,

    learning_rate = l_r,

    n_estimators=n_e,

    max_depth=m_d,

    min_child_weight=m_c_w,

    subsample=subsam,

    colsample_bytree=col_by,   

    reg_alpha = r_a,

    reg_lamda = r_l,

) # 最適パラメーターのXGBRegressor

model.fit(X_train, y_train)  # 学習
X_test = df_test.values

predict = model.predict(X_test)
df_submit=pd.read_csv('../input/1056lab-student-performance-prediction/sampleSubmission.csv',index_col=0)

df_submit['G3']=predict

df_submit.to_csv('Submission3.csv')