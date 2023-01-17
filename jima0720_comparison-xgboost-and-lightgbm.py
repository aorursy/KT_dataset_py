import warnings
warnings.filterwarnings('ignore')
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
%%time
import pandas as pd, numpy as np, time
from sklearn.model_selection import train_test_split

data = pd.read_csv("../input/flights.csv")
data = data.sample(frac = 0.1, random_state=10)

data = data[["MONTH","DAY","DAY_OF_WEEK","AIRLINE","FLIGHT_NUMBER","DESTINATION_AIRPORT",
                 "ORIGIN_AIRPORT","AIR_TIME", "DEPARTURE_TIME","DISTANCE","ARRIVAL_DELAY"]]
data.dropna(inplace=True)

data["ARRIVAL_DELAY"] = (data["ARRIVAL_DELAY"]>10)*1

cols = ["AIRLINE","FLIGHT_NUMBER","DESTINATION_AIRPORT","ORIGIN_AIRPORT"]
for item in cols:
    data[item] = data[item].astype("category").cat.codes +1
 
train, test, y_train, y_test = train_test_split(data.drop(["ARRIVAL_DELAY"], axis=1), data["ARRIVAL_DELAY"],
                                                random_state=10, test_size=0.25)

train.shape
train.head(3)
test.shape
test.head(3)
# ちょっとした興味
train2 = train[['AIR_TIME', 'DISTANCE']]
train2['VELOCITY'] = train2['DISTANCE']/ (train2['AIR_TIME'] / 60.0)
train2.head(3)
import matplotlib.pyplot as plt
import seaborn as sns

plt.scatter(train2['DISTANCE'], train2['VELOCITY'])
plt.xlabel('distance')
plt.ylabel('velocity[km/h]')
%%time
import xgboost as xgb
from sklearn import metrics, model_selection

def auc(m, train, test): 
    return (metrics.roc_auc_score(y_train,m.predict_proba(train)[:,1]),
                            metrics.roc_auc_score(y_test,m.predict_proba(test)[:,1]))


'''
class xgboost.XGBRegressor(max_depth=3, learning_rate=0.1, n_estimators=100, silent=True, objective='reg:linear', 
booster='gbtree', n_jobs=1, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1, 
colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, random_state=0, seed=None, missing=None, importance_type='gain', **kwargs)
'''
# Parameter Tuning
model = xgb.XGBClassifier()
param_dist = {"max_depth": [3, 6, 9, 12], # default 6
              "min_child_weight" : [1], # default 1
              "n_estimators": [100, 150], # default 100
              "learning_rate": [0.05, 0.1, 0.15], # default 0.1
             } 

grid_search = model_selection.GridSearchCV(model, param_grid=param_dist, cv = 3, 
                                   verbose=10, n_jobs=-1, scoring='roc_auc')
grid_search.fit(train, y_train)

print("{}".format(grid_search.best_estimator_))
print("{}".format(grid_search.best_score_))
print("{}".format(grid_search.best_params_))
# print("{}".format(grid_search.scorer_ ))
# print("{}".format(grid_search.cv_results_))


# model = xgb.XGBClassifier(max_depth=50, min_child_weight=1,  n_estimators=200,\
#                           n_jobs=-1 , verbose=1,learning_rate=0.16)
# model.fit(train,y_train)

# auc(model, train, test)

# おおよそ 40sec / fit

for i, param in enumerate(grid_search.cv_results_['params']):
    print("{} / ROC_AUC SCORE: {:.2f}".format(param, grid_search.cv_results_['mean_test_score'][i]))

%%time
import lightgbm as lgb
from sklearn import metrics, model_selection
def auc2(m, train, test): 
    return (metrics.roc_auc_score(y_train,m.predict(train)),
                            metrics.roc_auc_score(y_test,m.predict(test)))

lg = lgb.LGBMClassifier(silent=False)

'''
class lightgbm.LGBMModel(boosting_type='gbdt', num_leaves=31, max_depth=-1, learning_rate=0.1, n_estimators=100,
    subsample_for_bin=200000, objective=None, class_weight=None, min_split_gain=0.0, min_child_weight=0.001, min_child_samples=20,
    subsample=1.0, subsample_freq=0, colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=0.0, random_state=None, n_jobs=-1, silent=True,
    importance_type='split', **kwargs)[source]
'''
param_dist = {"max_depth": [-1], # default -1
              "learning_rate" : [0.05,0.1, 0.15], # default 0.1
              "num_leaves": [31, 63], # default 31
              "n_estimators": [100, 150] # default 100
             }
grid_search = model_selection.GridSearchCV(lg, n_jobs=-1, param_grid=param_dist, cv = 3, scoring="roc_auc", verbose=10)
grid_search.fit(train,y_train)

print("{}".format(grid_search.best_estimator_))
print("{}".format(grid_search.best_score_))
print("{}".format(grid_search.best_params_))




# model2 = lgb.train(grid_search.best_params_, lgb.Dataset(train, label=y_train))
# train_auc, test_auc = auc2(model2, train, test)

# print("train_auc: {:.3f}".format(train_auc))
# print("test_auc: {:.3f}".format(test_auc))


for i, param in enumerate(grid_search.cv_results_['params']):
    print("{} / ROC_AUC SCORE: {:.2f}".format(param, grid_search.cv_results_['mean_test_score'][i]))

%%time
from sklearn import metrics, model_selection
from sklearn import ensemble
# def auc2(m, train, test): 
#     return (metrics.roc_auc_score(y_train,m.predict(train)),
#                             metrics.roc_auc_score(y_test,m.predict(test)))

rf = ensemble.RandomForestClassifier()

'''
class sklearn.ensemble.RandomForestClassifier(n_estimators=’warn’, criterion=’gini’, max_depth=None, min_samples_split=2, 
        min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=’auto’, max_leaf_nodes=None, min_impurity_decrease=0.0, 
        min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None)[source]
'''
param_dist = {
#               "max_depth": [-1], # default -1
#               "learning_rate" : [0.05,0.1, 0.15], # default 0.1
#               "num_leaves": [31, 63], # default 31
              "n_estimators": [50, 100, 150] # default 100
             }
grid_search = model_selection.GridSearchCV(rf, n_jobs=-1, param_grid=param_dist, cv = 3, scoring="roc_auc", verbose=10)
grid_search.fit(train,y_train)

print("{}".format(grid_search.best_estimator_))
print("{}".format(grid_search.best_score_))
print("{}".format(grid_search.best_params_))

# 1min / fit

for i, param in enumerate(grid_search.cv_results_['params']):
    print("{} / ROC_AUC SCORE: {:.2f}".format(param, grid_search.cv_results_['mean_test_score'][i]))
