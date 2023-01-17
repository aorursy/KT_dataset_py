from numpy import loadtxt

from xgboost import XGBRegressor

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

import pandas as pd 

import pandas as pd

from sklearn import preprocessing

import numpy as np

from sklearn.model_selection import KFold,TimeSeriesSplit,StratifiedKFold

from sklearn.metrics import roc_auc_score,mean_squared_log_error,mean_squared_error,f1_score,r2_score

from xgboost import plot_importance

from sklearn.metrics import make_scorer

from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK, STATUS_RUNNING

import xgboost as xgb

import gc

from sklearn.datasets import load_boston

import pandas as pd

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

import statsmodels.api as sm

%matplotlib inline

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.feature_selection import RFE

from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
train = pd.read_csv('../input/covid19-global-forecasting-week-3/train.csv')

test = pd.read_csv('../input/covid19-global-forecasting-week-3/test.csv')

train
le = preprocessing.LabelEncoder()



#nomalize country region names 

train["Country_Region"]=le.fit_transform(train["Country_Region"])

test["Country_Region"]=le.transform(test["Country_Region"])



#deleting Id column in train

train.drop(["Id"],inplace=True,axis=1)



#changing dates to int 

test["Date"] = test["Date"].apply(lambda x: x.replace("-",""))

test["Date"]  = test["Date"].astype(int)

train["Date"] = train["Date"].apply(lambda x: x.replace("-",""))

train["Date"]  = train["Date"].astype(int)



#clearing NaN 

train["Province_State"].fillna("a",inplace=True)

test["Province_State"].fillna("a",inplace=True)



#nomalize states names 

train["Province_State"]=le.fit_transform(train["Province_State"])

test["Province_State"]=le.transform(test["Province_State"]) 



#Using Pearson Correlation

plt.figure(figsize=(12,10))

cor = train.corr()

sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)

plt.show()
y_conf = train["ConfirmedCases"]

y_fat = train["Fatalities"]

X_conf_train = train.drop(["ConfirmedCases","Fatalities"],axis=1)

X_fat_train = train.drop(["Fatalities"],axis=1)

X_fat_test = test.drop(["ForecastId"],axis=1)

X_conf_test = test.drop(["ForecastId"],axis=1)
#defining  conf models

model_conf = xgb.XGBRegressor(base_score=0.5, booster=None, colsample_bylevel=1,

             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=0,

             importance_type='gain', interaction_constraints=None,

             learning_rate=0.3, max_delta_step=0, max_depth=25,

             min_child_weight=1, monotone_constraints=None,

             n_estimators=1800, n_jobs=0, num_parallel_tree=1,

             objective='reg:squarederror', random_state=0, reg_alpha=0,

             reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method='gpu_hist',

             validate_parameters=False, verbosity=None)
#fat model



model_fat = xgb.XGBRegressor(base_score=0.5, booster=None, colsample_bylevel=1,

             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=0,

             importance_type='gain', interaction_constraints=None,

             learning_rate=0.1, max_delta_step=0, max_depth=25,

             min_child_weight=1, monotone_constraints=None,

             n_estimators=1800, n_jobs=0, num_parallel_tree=1,

             objective='reg:squarederror', random_state=0, reg_alpha=0,

             reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method='gpu_hist',

             validate_parameters=False, verbosity=None)
def train_conf():

    model_conf.fit(X_conf_train,y_conf)
def conf_predict():

    # final predictions

    pr = model_conf.predict(X_conf_test)

    tmp_pr = []

    for i in pr:

        if i < 0:

            tmp_pr.append(0)

            continue

        tmp_pr.append(int(i))

    for i in range(1,len(tmp_pr)):

        if tmp_pr[i] < tmp_pr[i-1] and tmp_pr[i] != 0:

            tmp_pr[i] = tmp_pr[i-1]

    pr_conf = tmp_pr

    return pr_conf
def train_fat():

    model_fat.fit(X_fat_train,y_fat)
def fat_predict(list_conf):

    tmp_df = X_conf_test

    tmp_df['ConfirmedCases'] = list_conf

    pr = model_fat.predict(tmp_df)

    tmp_pr = []

    for i in pr:

        if i < 0:

            tmp_pr.append(0)

            continue

        tmp_pr.append(int(i))

    for i in range(1,len(tmp_pr)):

        if tmp_pr[i] < tmp_pr[i-1] and tmp_pr[i] != 0:

            tmp_pr[i] = tmp_pr[i-1]

    pr_fat = tmp_pr

    return pr_fat
for i in range(1):

    train_conf()

ans_conf = conf_predict()

for i in range(1):

    train_fat()

ans_fat = fat_predict(ans_conf)
submission = pd.read_csv("../input/covid19-global-forecasting-week-3/submission.csv")
submission['ConfirmedCases'] = ans_conf

submission['Fatalities'] = ans_fat

submission
submission.to_csv('submission.csv', index = False)