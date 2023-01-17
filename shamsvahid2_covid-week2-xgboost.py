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
df_train=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/train.csv')

test=pd.read_csv('/kaggle/input//covid19-global-forecasting-week-2/test.csv')

list_id = test['ForecastId'].values
df_train.head()

le = preprocessing.LabelEncoder()
df_train["Country_Region"]=le.fit_transform(df_train["Country_Region"])

test["Country_Region"]=le.transform(test["Country_Region"])

df_train.drop(["Id"],inplace=True,axis=1)

test["Date"] = test["Date"].apply(lambda x: x.replace("-",""))

test["Date"]  = test["Date"].astype(int)

df_train["Date"] = df_train["Date"].apply(lambda x: x.replace("-",""))

df_train["Date"]  = df_train["Date"].astype(int)
df_train["Province_State"].fillna("a",inplace=True)

test["Province_State"].fillna("a",inplace=True)
df_train["Province_State"]=le.fit_transform(df_train["Province_State"])

test["Province_State"]=le.transform(test["Province_State"])
y_Confim_cases = df_train["ConfirmedCases"]

y_Fatality = df_train["Fatalities"]

X=df_train.drop(["ConfirmedCases","Fatalities"],axis=1)
# split data into train and test sets

seed = 1 

test_size = 0.3

X_train, X_test, y_train, y_test = train_test_split(X,y_Confim_cases, test_size=test_size,random_state=seed)

model_conf = xgb.XGBRegressor(base_score=0.5, booster=None, colsample_bylevel=1,

             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,

             importance_type='gain', interaction_constraints=None,

             learning_rate=0.3, max_delta_step=0, max_depth=19,

             min_child_weight=1, monotone_constraints=None,

             n_estimators=1000, n_jobs=0, num_parallel_tree=1,

             objective='reg:squarederror', random_state=0, reg_alpha=0,

             reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method=None,

             validate_parameters=False, verbosity=None)

model_conf.fit(X_train, y_train)
# final predictions

model_conf.fit(X,y_Confim_cases)

pr = model_conf.predict(X_test)

tmp_pr = []

for i in pr:

    if i < 0:

        tmp_pr.append(0)

        continue

    tmp_pr.append(int(i))

pr_conf = tmp_pr
# split data into train and test sets

seed = 1 

test_size = 0.3

X_train, X_test, y_train, y_test = train_test_split(X,y_Fatality, test_size=test_size,random_state=seed)
model_fat = xgb.XGBRegressor(base_score=0.5, booster=None, colsample_bylevel=1,

             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,

             importance_type='gain', interaction_constraints=None,

             learning_rate=0.3, max_delta_step=0, max_depth=18,

             min_child_weight=1, monotone_constraints=None,

             n_estimators=1000, n_jobs=0, num_parallel_tree=1,

             objective='reg:squarederror', random_state=0, reg_alpha=0,

             reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method=None,

             validate_parameters=False, verbosity=None)

model_fat.fit(X_train, y_train)
pr = model_fat.predict(X_test)

tmp_pr = []

for i in pr:

    if i < 0:

        tmp_pr.append(0)

        continue

    tmp_pr.append(int(i))

pr_fat = tmp_pr
print("Train Error",mean_squared_log_error(pr_fat, y_test))
data_test = test 

data_test.drop(["ForecastId"],inplace=True,axis=1)

data_test
#training model (whole data)

model_conf.fit(X,y_Confim_cases)

model_fat.fit(X,y_Fatality)





#predict conf 

pr = model_conf.predict(data_test)

tmp_pr = []

for i in pr:

    if i < 0:

        tmp_pr.append(0)

        continue

    tmp_pr.append(int(i))

pr_conf = tmp_pr





#predict fat

pr = model_fat.predict(data_test)

tmp_pr = []

for i in pr:

    if i < 0:

        tmp_pr.append(0)

        continue

    tmp_pr.append(int(i))

pr_fat = tmp_pr

pr_conf
test


ans = {'ForecastId' : list_id,

       'ConfirmedCases' :pr_conf,

       'Fatalities' : pr_fat

}
df = pd.read_csv('../input/covid19-global-forecasting-week-2/submission.csv')

df['ForecastId'] = list_id

df['ConfirmedCases'] = pr_conf

df['Fatalities'] = pr_fat
df.to_csv("submission.csv", index = False)