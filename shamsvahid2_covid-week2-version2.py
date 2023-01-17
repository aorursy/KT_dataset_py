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
train=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/train.csv')

test=pd.read_csv('/kaggle/input//covid19-global-forecasting-week-2/test.csv')
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
#defining fat and conf models

model_conf = xgb.XGBRegressor(base_score=0.5, booster=None, colsample_bylevel=1,

             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,

             importance_type='gain', interaction_constraints=None,

             learning_rate=0.3, max_delta_step=0, max_depth=19,

             min_child_weight=1, monotone_constraints=None,

             n_estimators=1000, n_jobs=0, num_parallel_tree=1,

             objective='reg:squarederror', random_state=0, reg_alpha=0,

             reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method=None,

             validate_parameters=False, verbosity=None)



model_fat = xgb.XGBRegressor(base_score=0.5, booster=None, colsample_bylevel=1,

             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,

             importance_type='gain', interaction_constraints=None,

             learning_rate=0.3, max_delta_step=0, max_depth=18,

             min_child_weight=1, monotone_constraints=None,

             n_estimators=1000, n_jobs=0, num_parallel_tree=1,

             objective='reg:squarederror', random_state=0, reg_alpha=0,

             reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method=None,

             validate_parameters=False, verbosity=None)
train
def train_function():

    train_tmp = pd.DataFrame(train)

    y_conf = train_tmp["ConfirmedCases"]

    y_fat = train_tmp["Fatalities"]

    X_fat = train_tmp.drop(["Fatalities"],axis=1)

    X_conf = train_tmp.drop(["ConfirmedCases","Fatalities"],axis=1)

    model_fat.fit(X_fat,y_fat)

    model_conf.fit(X_conf,y_conf)
train_function()
def test_fat(conf_list):

    test_tmp = test.drop(["ForecastId"],axis=1)

    test_tmp["ConfirmedCases"] = np.array(conf_list)

    pr = model_fat.predict(test_tmp)

    tmp_pr = []

    for i in pr:

        if i < 0:

            tmp_pr.append(0)

            continue

        tmp_pr.append(int(i))

    pr_fat = tmp_pr

    for i in range(1,len(tmp_pr)):

        if tmp_pr[i] < tmp_pr[i-1]:

            tmp_pr[i] = tmp_pr[i-1]

    return tmp_pr
def test_conf():

    test_tmp = test.drop(["ForecastId"],axis=1)

    pr = model_conf.predict(test_tmp)

    tmp_pr = []

    for i in pr:

        if i < 0:

            tmp_pr.append(0)

            continue

        tmp_pr.append(int(i))

    pr_conf = tmp_pr

    for i in range(1,len(tmp_pr)):

        if tmp_pr[i] < tmp_pr[i-1]:

            tmp_pr[i] = tmp_pr[i-1]

    return tmp_pr
conf_list = test_conf()

fat_list = test_fat(conf_list)
submission = pd.DataFrame()

submission["ForecastId"] = np.array(test["ForecastId"])

submission["ConfirmedCases"] = np.array(conf_list)

submission["Fatalities"] = np.array(fat_list)
submission
# submission.to_csv("submission.csv",index = False)
#double learning defining fat and conf models

model_conf2 = xgb.XGBRegressor(base_score=0.5, booster=None, colsample_bylevel=1,

             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,

             importance_type='gain', interaction_constraints=None,

             learning_rate=0.3, max_delta_step=0, max_depth=19,

             min_child_weight=1, monotone_constraints=None,

             n_estimators=1000, n_jobs=0, num_parallel_tree=1,

             objective='reg:squarederror', random_state=0, reg_alpha=0,

             reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method=None,

             validate_parameters=False, verbosity=None)



model_fat2 = xgb.XGBRegressor(base_score=0.5, booster=None, colsample_bylevel=1,

             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,

             importance_type='gain', interaction_constraints=None,

             learning_rate=0.3, max_delta_step=0, max_depth=18,

             min_child_weight=1, monotone_constraints=None,

             n_estimators=1000, n_jobs=0, num_parallel_tree=1,

             objective='reg:squarederror', random_state=0, reg_alpha=0,

             reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method=None,

             validate_parameters=False, verbosity=None)
def train_function2():

    train_tmp = pd.DataFrame(train)

    y_conf = train_tmp["ConfirmedCases"]

    y_fat = train_tmp["Fatalities"]

    X_fat = train_tmp.drop(["Fatalities"],axis=1)

    X_conf = train_tmp.drop(["ConfirmedCases"],axis=1)

    model_fat2.fit(X_fat,y_fat)

    model_conf2.fit(X_conf,y_conf)
train_function2()
def test_fat2(conf_list):

    test_tmp = test.drop(["ForecastId"],axis=1)

    test_tmp["ConfirmedCases"] = np.array(conf_list)

    pr = model_fat2.predict(test_tmp)

    tmp_pr = []

    for i in pr:

        if i < 0:

            tmp_pr.append(0)

            continue

        tmp_pr.append(int(i))

    pr_fat = tmp_pr

    for i in range(1,len(tmp_pr)):

        if tmp_pr[i] < tmp_pr[i-1]:

            tmp_pr[i] = tmp_pr[i-1]

    return tmp_pr
def test_conf2(fat_list):

    test_tmp = test.drop(["ForecastId"],axis=1)

    test_tmp["Fatalities"] = np.array(fat_list)

    pr = model_conf2.predict(test_tmp)

    tmp_pr = []

    for i in pr:

        if i < 0:

            tmp_pr.append(0)

            continue

        tmp_pr.append(int(i))

    pr_conf = tmp_pr

    for i in range(1,len(tmp_pr)):

        if tmp_pr[i] < tmp_pr[i-1]:

            tmp_pr[i] = tmp_pr[i-1]

    return tmp_pr
conf_list2 = test_conf2(fat_list)

fat_list2 = test_fat(conf_list)
submission2 = pd.DataFrame()

submission2["ForecastId"] = np.array(test["ForecastId"])

submission2["ConfirmedCases"] = np.array(conf_list2)

submission2["Fatalities"] = np.array(fat_list2)
submission2
submission2.to_csv("submission.csv",index = False)