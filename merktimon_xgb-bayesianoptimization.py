# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import pandas_profiling as pp
# data preprocessing
from sklearn.preprocessing import StandardScaler
from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer  
from skopt.utils import use_named_args
import xgboost as xgb
from xgboost import XGBClassifier
# data splitting
from sklearn.model_selection import train_test_split
from skopt.plots import plot_convergence

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/health-care-data-set-on-heart-attack-possibility/heart.csv')
#print first 10 rows
df.head(10)
plt.figure(figsize=(12,10))
sns.heatmap(df.corr(), robust=True, fmt="f", cmap='RdBu_r', vmin=-1, vmax=1)
pp.ProfileReport(df)
y = df["target"]
X = df.drop('target',axis=1)
dtrain = xgb.DMatrix(X, label = y)
space_XGB  = [Integer(2, 20, name='max_depth'),
              Real(0.001, 10**1, "log-uniform", name="gamma"), 
          Real(10**-5, 10**0, "log-uniform", name='learning_rate'),
          Real(0,30, "uniform", name="min_child_weight"), 
          Real(0,10, "uniform", name="max_delta_step"), 
          Real(0.1, 1, "uniform", name="subsample"),
          Real(0.1,1, "uniform", name="colsample_bytree")]
@use_named_args(space_XGB)
def objective_XGB(**params):
    print(params)
    
    params_ = {'max_depth': int(params["max_depth"]),
         'gamma': params['gamma'],
         'learning_rate': params["learning_rate"],
          'min_child_weight':params["min_child_weight"],
          'max_delta_step':params["max_delta_step"],
          'subsample':params["subsample"],
          'colsample_bytree':params["colsample_bytree"], 
         'eta': 0.1} 
         #'tree_method' : 'gpu_hist', 
         #'gpu_id' : 1}
    
    cv_result = xgb.cv(params_, dtrain=dtrain, nfold=3, metrics='error', early_stopping_rounds=10, num_boost_round=500,
                      shuffle=True)
    print(1-cv_result.iloc[-1]['test-error-mean'])
    
    return -(1-cv_result.iloc[-1]['test-error-mean']) # error is wrong cases / all cases
res_gp = gp_minimize(objective_XGB, space_XGB, n_calls=150, random_state=0)
plot_convergence(res_gp)

# best 3 fold CV accuracy
-res_gp.fun
# best parameters
res_gp.x
