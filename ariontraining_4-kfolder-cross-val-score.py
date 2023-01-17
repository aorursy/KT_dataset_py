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
# split for machine learning model



train_data, valid_data = train_test_split(data, test_size = 0.3)



label = "cost"



data_labels = train_data.columns.tolist()

data_labels.remove(label)



train_df = train_data[data_labels]

valid_df = valid_data[data_labels]

train_label = train_data[label]

valid_label = valid_data[label]
train_df.head()
valid_df.head()
#Linear regression



from sklearn.linear_model import LinearRegression



start = time.time()

linear=LinearRegression()





label_log=np.log1p(train_label)



model=linear.fit(train_df, label_log)

linear_preds1=model.predict(valid_df)



linear_preds=np.expm1(linear_preds1)

        

rmsle_linear = rmsle_score(linear_preds, valid_label)

print ("Linear RMSLE is : {}".format(rmsle_linear))



compare_linear_log = compare_result(preds=linear_preds, true=valid_label)



end = time.time()

duration = end - start

print ("It takes {} seconds".format(duration))
# split for cross_val_score machine learning model



label = "cost"



data_labels = data.columns.tolist()

data_labels.remove(label)



X = data[data_labels]

y = data[label]
#Linear regression and cross_val_score

from sklearn.cross_validation import cross_val_score



from sklearn.linear_model import LinearRegression



start = time.time()

linear=LinearRegression()



rmsle_scores = cross_val_score(linear, X, y, scoring=RMSLE, cv=5)

print("RMSLE are:{}".format(rmsle_scores))

print("Mean RMSLE score:{}".format(np.mean(rmsle_scores)))



end = time.time()

duration = end - start

print ("It takes {} seconds".format(duration))
# RandomForest Regression 

from sklearn.ensemble import RandomForestRegressor



start = time.time()

rf=RandomForestRegressor(random_state=0)



label_log=np.log1p(train_label)



model=rf.fit(train_df, label_log)

rf_preds1=model.predict(valid_df)



rf_preds=np.expm1(rf_preds1)

        

rmsle_rf = rmsle_score(rf_preds, valid_label)

print ("Random Forest RMSLE is : {}".format(rmsle_rf))



compare_rf_log = compare_result(preds=rf_preds, true=valid_label)



end = time.time()

duration = end - start

print ("It takes {} seconds".format(duration))
#RandomForest Regression and cross_val_score

from sklearn.cross_validation import cross_val_score

from sklearn.ensemble import RandomForestRegressor



start = time.time()

rf=RandomForestRegressor(random_state=0)



rmsle_scores = cross_val_score(rf, X, y, scoring=RMSLE, cv=5)

print("RMSLE are:{}".format(rmsle_scores))

print("Mean RMSLE score:{}".format(np.mean(rmsle_scores)))



end = time.time()

duration = end - start

print ("It takes {} seconds".format(duration))