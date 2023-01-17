# Import necessary libraries

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns 

import os

import warnings



warnings.filterwarnings("ignore")
# For plot sizes

plt.rcParams["figure.figsize"] = (18,8)

sns.set(rc={'figure.figsize':(18,8)})
os.listdir('../input')
# Load Part 1 data

data_ind = pd.read_csv('../input/pubg-walkthrough/Training_Data_New.csv')

print("Done loading data from part 1")
data_ind.drop(['Unnamed: 0', 'Id', 'groupId', 'matchId'], axis=1, inplace=True)
data_matchT = pd.get_dummies(data_ind['matchType'])

data_ind = pd.concat([data_ind, data_matchT], axis=1)

data_ind.drop('matchType', axis=1, inplace=True)

data_ind.head()
data_ind.shape
from sklearn.metrics import mean_absolute_error as mae

from sklearn.model_selection import train_test_split

import tqdm
y = data_ind['winPlacePerc']
X = data_ind

X.drop('winPlacePerc', axis=1, inplace=True)
y.shape
X.shape
X_train_ind, X_test_ind, y_train_ind, y_test_ind = train_test_split(X, y, test_size=0.15, random_state=12)
from lightgbm import LGBMRegressor

import datetime
time_0 = datetime.datetime.now()



lgbm = LGBMRegressor(objective='mae', n_jobs=-1, random_state=12)



lgbm.fit(X_train_ind, y_train_ind)



time_1  = datetime.datetime.now()



print('Training took {} seconds.'.format((time_1 - time_0).seconds))

print('Mean Absolute Error is {:.5f}'.format(mae(y_test_ind, lgbm.predict(X_test_ind))))
import shap
shap.initjs()



SAMPLE_SIZE = 10000

SAMPLE_INDEX = np.random.randint(0, X_test_ind.shape[0], SAMPLE_SIZE)



X = X_test_ind.iloc[SAMPLE_INDEX]



explainer = shap.TreeExplainer(lgbm)

shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X)
shap.summary_plot(shap_values, X, plot_type='bar', color='darkred')
# Let's also try xgboost 

import xgboost as xgb
regressor = xgb.XGBRegressor(objective = 'reg:squarederror')

regressor
regressor.fit(X_train_ind, y_train_ind)

y_pred = regressor.predict(X_test_ind)
# check the MAE

Mae = mae(y_test_ind, y_pred)

print('MAE %f' % (Mae))
xgb.plot_importance(regressor)

plt.title("xgboost.plot_importance(regressor)")

plt.show()
from catboost import CatBoostRegressor
cat = CatBoostRegressor(iterations = 300, eval_metric='MAE', metric_period=10)
cat.fit(X_train_ind, 

       y_train_ind,

      eval_set=(X_test_ind, y_test_ind),

     use_best_model=True)
explainer = shap.TreeExplainer(cat)

shap_values = explainer.shap_values(X)
# summary of features for cat model 

shap.summary_plot(shap_values, X)