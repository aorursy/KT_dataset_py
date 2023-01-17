import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn import preprocessing

import time



pd.set_option('display.max_columns', 500)

pd.set_option('display.max_colwidth', 500)

pd.set_option('display.max_rows', 1000)



import warnings

warnings.filterwarnings('ignore')

%matplotlib inline 
train = pd.read_csv("../input/combination.csv")
#There is no common tube_assembly_id between train and test data. So we drop this variable.

train.drop("tube_assembly_id", axis=1, inplace=True)
train.head().transpose()
train.quote_date = pd.to_datetime(train.quote_date)
#add new numeric time features



train["year"] = train.quote_date.dt.year

train["month"] = train.quote_date.dt.month

train["day"] = train.quote_date.dt.day

train["day_of_week"] = train.quote_date.dt.dayofweek
#only use numeric data

data = train.select_dtypes(include=['int', 'float'])
#fill null by 0

data.replace(np.nan, 0, inplace=True)
#define a evaluation function



def rmsle_score(preds, true):

    rmsle_score = (np.sum((np.log1p(preds)-np.log1p(true))**2)/len(true))**0.5

    return rmsle_score
#Define a evaluation matrix 

from sklearn.metrics.scorer import make_scorer



RMSLE = make_scorer(rmsle_score)
# define a function for comparing predictions and true data.



def compare_result(preds, true):

    compare = pd.DataFrame({"test_id": true.index,

                           "real_cost": true,

                           "pred_cost": preds})

    compare = compare[["test_id", "real_cost", "pred_cost"]].reset_index(drop=True)

    

    compare["error_percent_(%)"] = np.abs(compare.real_cost - compare.pred_cost) / compare.real_cost * 100

    

    return compare
import xgboost as xgb

from xgboost import XGBRegressor
# split for machine learning model



train_data, valid_data = train_test_split(data, test_size = 0.2)



label = "cost"



data_labels = train_data.columns.tolist()

data_labels.remove(label)



train_df = train_data[data_labels]

valid_df = valid_data[data_labels]

train_label = train_data[label]

valid_label = valid_data[label]
#XGB regression



start = time.time()

xgb_regressor=XGBRegressor(max_depth=3, n_estimators=300, learning_rate=0.1)





label_log=np.log1p(train_label)



model = xgb_regressor.fit(train_df, label_log)

xgb_preds1 = model.predict(valid_df)



xgb_preds = np.expm1(xgb_preds1)

        

rmsle_xgb = rmsle_score(xgb_preds, valid_label)

print ("XGB RMSLE is : {}".format(rmsle_xgb))



compare_xgb = compare_result(preds=xgb_preds, true=valid_label)



end = time.time()

duration = end - start

print ("It takes {} seconds".format(duration))
compare_xgb.head(10)
from sklearn.model_selection import GridSearchCV

#set parameters

parameters = {

    'max_depth': [3, 5, 7],

    "n_estimators": [100, 300, 500],

}



#define XGB Grid Search model

xgb_gridsearch = GridSearchCV(xgb_regressor, parameters, scoring=RMSLE, cv=5)
#grid search experiment

start = time.time()



#label_log=np.log1p(train_label)



xgb_gridsearch.fit(train_df, train_label)



end = time.time()

duration = end-start

print ("It takes {} seconds".format(duration))
#get/show the best parameters

best_parameters, score, _ = min(xgb_gridsearch.grid_scores_, key=lambda x: x[1])

print('score:', score)



for param_name in sorted(best_parameters.keys()):

    print("%s: %r" % (param_name, best_parameters[param_name]))

    



#use best model to predict

start = time.time()

xgb_regressor = XGBRegressor(max_depth=best_parameters["max_depth"], n_estimators=best_parameters["n_estimators"], learning_rate=0.1)



label_log = np.log1p(train_label)



model = xgb_regressor.fit(train_df, label_log)

xgb_preds1 = model.predict(valid_df)



xgb_preds = np.expm1(xgb_preds1)

        

rmsle_xgb = rmsle_score(xgb_preds, valid_label)

print ("XGB RMSLE is : {}".format(rmsle_xgb))



end = time.time()

duration = end - start

print ("It takes {} seconds".format(duration))



compare_xgb = compare_result(preds=xgb_preds, true=valid_label)
compare_xgb.head(10)
from sklearn.model_selection import RandomizedSearchCV

#define XGB Random Grid Search model

xgb_randomsearch = RandomizedSearchCV(xgb_regressor, parameters, scoring=RMSLE, cv=5, n_iter=3) #n_iter works for what?



#set parameters

parameters = {

    'max_depth': [3, 5, 7],

    "n_estimators": [100, 300, 500],

}
#Random Grid Search experiment

start = time.time()



#label_log=np.log1p(train_label)



xgb_randomsearch.fit(train_df, train_label)



end = time.time()

duration = end-start

print ("It takes {} seconds".format(duration))
#get/show the best parameters

best_parameters, score, _ = min(xgb_randomsearch.grid_scores_, key=lambda x: x[1])

print('score:', score)



for param_name in sorted(best_parameters.keys()):

    print("%s: %r" % (param_name, best_parameters[param_name]))

    

#use best model to predict

start = time.time()

xgb_regressor = XGBRegressor(max_depth=best_parameters["max_depth"], n_estimators=best_parameters["n_estimators"], learning_rate=0.1)



label_log = np.log1p(train_label)



model = xgb_regressor.fit(train_df, label_log)

xgb_preds1 = model.predict(valid_df)



xgb_preds = np.expm1(xgb_preds1)

        

rmsle_xgb = rmsle_score(xgb_preds, valid_label)

print ("XGB RMSLE is : {}".format(rmsle_xgb))



end = time.time()

duration = end - start

print ("It takes {} seconds".format(duration))



compare_xgb = compare_result(preds=xgb_preds, true=valid_label)
compare_xgb.head(10)