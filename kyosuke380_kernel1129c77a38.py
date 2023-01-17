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
train_df = pd.read_csv('/kaggle/input/1056lab-import-car-insurance-prediction/train.csv', na_values="?",index_col=0)

test_df = pd.read_csv('/kaggle/input/1056lab-import-car-insurance-prediction/test.csv', na_values="?",index_col=0)
train_df
test_df
train_df.dtypes
test_df.dtypes
train_df.isnull().sum()
test_df.isnull().sum()
from collections import Counter
Counter(train_df['normalized-losses'])
Counter(train_df['make'])
Counter(train_df['fuel-type'])
Counter(train_df['aspiration'])
Counter(train_df['num-of-doors'])
Counter(train_df['body-style'])
Counter(train_df['drive-wheels'])
Counter(train_df['engine-location'])
Counter(train_df['engine-type'])
Counter(train_df['num-of-cylinders'])
Counter(train_df['fuel-system'])
Counter(train_df['bore'])
Counter(train_df['stroke'])
Counter(train_df['horsepower'])
Counter(train_df['peak-rpm'])
Counter(train_df['price'])
train_df2=train_df.drop(columns='normalized-losses')
train_df2 = train_df2.dropna(how='any')
train_df2
train_df2['make'] = train_df2['make'].map({'alfa-romero':1, 'audi':2, 'bmw':3, 'chevrolet':4, 'dodge':5, 'honda':6, 'isuzu':7, 'jaguar':8, 'mazda':9, 'mercedes-benz':10, 'mercury':11, 'mitsubishi':12, 'nissan':13, 'peugot':14, 'plymouth':15, 'porsche':16, 'renault':17, 'saab':18, 'subaru':19, 'toyota':20, 'volkswagen':21, 'volvo':22})
train_df2['fuel-type'] = train_df2['fuel-type'].map({'gas':1, 'diesel':2})
train_df2['aspiration'] = train_df2['aspiration'].map({'std':1, 'turbo':2})
train_df2['num-of-doors'] = train_df2['num-of-doors'].map({'two':1, 'four':2})
train_df2['body-style'] = train_df2['body-style'].map({'convertible':1, 'sedan':2, 'wagon':3, 'hatchback':4, 'hardtop':5})
train_df2['drive-wheels'] = train_df2['drive-wheels'].map({'rwd':1, 'fwd':2, '4wd':3})
train_df2['engine-location'] = train_df2['engine-location'].map({'front':1, 'rear':2})
train_df2['engine-type'] = train_df2['engine-type'].map({'dohc':1, 'ohc':2, 'l':3, 'ohcv':4, 'rotor':5, 'ohcf':6, 'dohcv':7})
train_df2['num-of-cylinders'] = train_df2['num-of-cylinders'].map({'four':1, 'five':2, 'six':3, 'three':4, 'twelve':5, 'two':6, 'eight':7})
train_df2['fuel-system'] = train_df2['fuel-system'].map({'mpfi':1, '2bbl':2, 'mfi':3, '1bbl':4, 'spfi':5, '4bbl':6, 'idi':7, 'spdi':8})
train_df2
train_df2.dtypes
test_df2=test_df.drop(columns='normalized-losses')
test_df2['make'] = test_df2['make'].map({'alfa-romero':1, 'audi':2, 'bmw':3, 'dodge':5, 'honda':6, 'mazda':9, 'mercedes-benz':10, 'mitsubishi':12, 'nissan':13, 'peugot':14, 'plymouth':15, 'porsche':16, 'saab':18, 'subaru':19, 'toyota':20, 'volkswagen':21, 'volvo':22})
test_df2['fuel-type'] = test_df2['fuel-type'].map({'gas':1, 'diesel':2})
test_df2['aspiration'] = test_df2['aspiration'].map({'std':1, 'turbo':2})
test_df2['num-of-doors'] = test_df2['num-of-doors'].map({'two':1, 'four':2})
test_df2['body-style'] = test_df2['body-style'].map({'convertible':1, 'sedan':2, 'wagon':3, 'hatchback':4, 'hardtop':5})
test_df2['drive-wheels'] = test_df2['drive-wheels'].map({'rwd':1, 'fwd':2, '4wd':3})
test_df2['engine-location'] = test_df2['engine-location'].map({'front':1, 'rear':2})
test_df2['engine-type'] = test_df2['engine-type'].map({'dohc':1, 'ohc':2, 'l':3, 'ohcv':4, 'ohcf':6})
test_df2['num-of-cylinders'] = test_df2['num-of-cylinders'].map({'four':1, 'five':2, 'six':3})
test_df2['fuel-system'] = test_df2['fuel-system'].map({'mpfi':1, '2bbl':2, '1bbl':4, 'idi':7, 'spdi':8})
test_df2.dtypes
test_df2
import seaborn as sns

from matplotlib import pyplot

import matplotlib.pyplot as plt



sns.set_style("darkgrid")

pyplot.figure(figsize=(17, 17))  # 図の大きさを大き目に設定

sns.heatmap(train_df2.corr(), square=True, annot=True)  # 相関係数でヒートマップを作成
train_df2=train_df2[['num-of-doors','body-style','engine-location','wheel-base','length','width','height','curb-weight','peak-rpm','symboling']]

test_df2=test_df2[['num-of-doors','body-style','engine-location','wheel-base','length','width','height','curb-weight','peak-rpm']]
train_df2
test_df2
import xgboost as xgb

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

train_dfx = train_df2.drop('symboling', axis=1).values

train_dfy = train_df2['symboling'].values

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

fin_xgboost.fit(train_dfx, train_dfy)
test_dfx = test_df2.values

predict = fin_xgboost.predict(test_dfx)
submit_df = pd.read_csv('/kaggle/input/1056lab-import-car-insurance-prediction/sampleSubmission.csv', index_col=0)

submit_df['symboling'] = predict

submit_df.to_csv('Submission1.csv')