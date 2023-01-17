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
data.head()
#define a evaluation function



def rmsle_score(preds, true):

    rmsle_score = (np.sum((np.log1p(preds)-np.log1p(true))**2)/len(true))**0.5

    return rmsle_score
#Define a evaluation matrix 

from sklearn.metrics.scorer import make_scorer



RMSLE = make_scorer(rmsle_score)
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
train_df.head()
#XGB regression

start = time.time()



#set up a XGB model with default parameters

xgb_regressor = XGBRegressor(max_depth=7, 

                           n_estimators=500, 

                           #objective="reg:linear", 

                           min_child_weight = 6,

                           subsample = 0.87,

                           colsample_bytree = 0.50,

                           scale_pos_weight = 1.0,                       

                           learning_rate=0.1)





label_log = np.log1p(train_label)



model = xgb_regressor.fit(train_df, label_log)

xgb_preds1 = model.predict(valid_df)



xgb_preds = np.expm1(xgb_preds1)

        

rmsle_xgb = rmsle_score(xgb_preds, valid_label)

print ("XGB RMSLE is : {}".format(rmsle_xgb))



end = time.time()

duration = end - start

print ("It takes {} seconds".format(duration))
# split for cross_val_score machine learning model



label = "cost"



data_labels = data.columns.tolist()

data_labels.remove(label)



X = data[data_labels]

y = data[label]
#XGB Regression and cross_val_score

from sklearn.cross_validation import cross_val_score



start = time.time()

xgb_regressor=XGBRegressor(max_depth=3, 

                           n_estimators=300, 

                           objective="reg:linear", 

                           min_child_weight = 6,

                           subsample = 0.87,

                           colsample_bytree = 0.50,

                           scale_pos_weight = 1.0,                       

                           learning_rate=0.1)



#y_log=np.log1p(y)



rmsle_scores = cross_val_score(xgb_regressor, X, y, scoring=RMSLE, cv=5)

print("RMSLE are:{}".format(rmsle_scores))

print("Mean RMSLE is : {}".format(np.mean(rmsle_scores)))



end = time.time()

duration = end-start

print ("It takes {} seconds".format(duration))
#XGB Regression and KFold

from sklearn.model_selection import KFold



start = time.time()

xgb_regressor=XGBRegressor(max_depth=3, n_estimators=300, learning_rate=0.05)

scores = []



kf = KFold(n_splits=5)



for i, (train_index, test_index) in enumerate(kf.split(X)):

    #print("TRAIN:", train_index, "TEST:", test_index)

    X_train, X_test = X.loc[train_index, :], X.loc[test_index, :]

    y_train, y_test = y[train_index], y[test_index]

    

    y_log = np.log1p(y_train)



    model = xgb_regressor.fit(X_train, y_log)

    xgb_preds1 = model.predict(X_test)



    xgb_preds = np.expm1(xgb_preds1)

        

    rmsle_xgb = rmsle_score(xgb_preds, y_test)

    print ("Folder cv {}, XGB RMSLE is : {}".format(i+1, rmsle_xgb))

    scores.append(rmsle_xgb)

    

print("Mean RMSLE is : {}".format(np.mean(scores)))



end = time.time()

duration = end - start

print ("It takes {} seconds".format(duration))