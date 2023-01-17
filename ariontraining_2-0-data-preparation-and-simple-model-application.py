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
train.head()
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
train_data, valid_data = train_test_split(data, test_size = 0.3)
label = "cost"
data_labels = train_data.columns.tolist()

data_labels.remove(label)
train_df = train_data[data_labels]

valid_df = valid_data[data_labels]

train_label = train_data[label]

valid_label = valid_data[label]
#define a evaluation function



def rmse_score(preds, true):

    rmse_score = (np.sum((np.log1p(preds)-np.log1p(true))**2)/len(true))**0.5

    return rmse_score
# define a function for comparing predictions and true data.



def compare_result(preds, true):

    compare = pd.DataFrame({"test_id": true.index,

                           "real_cost": true,

                           "pred_cost": preds})

    compare = compare[["test_id", "real_cost", "pred_cost"]].reset_index(drop=True)

    

    compare["error_percent_(%)"] = np.abs(compare.real_cost - compare.pred_cost) / compare.real_cost * 100

    

    return compare



from sklearn.linear_model import LinearRegression



start = time.time()

linear=LinearRegression()





model=linear.fit(train_df, train_label)

linear_preds=model.predict(valid_df)



        

rmse_linear = rmse_score(linear_preds, valid_label)

print ("Linear RMSLE is : {}".format(rmse_linear))



compare_linear = compare_result(preds=linear_preds, true=valid_label)



end = time.time()

duration = end - start

print ("It takes {} seconds".format(duration))
compare_linear.sample(5)
from sklearn.linear_model import LinearRegression



start = time.time()

linear=LinearRegression()





label_log=np.log1p(train_label)



model=linear.fit(train_df, label_log)

linear_preds=model.predict(valid_df)



linear_preds=np.expm1(linear_preds)

        

rmse_linear = rmse_score(linear_preds, valid_label)

print ("Linear RMSLE is : {}".format(rmse_linear))



compare_linear_log = compare_result(preds=linear_preds, true=valid_label)



end = time.time()

duration = end - start

print ("It takes {} seconds".format(duration))
compare_linear_log.sample(5)
# sklearn LinearRegression

# Preprocess: do feature scaling or not 

from sklearn.ensemble import RandomForestRegressor



start = time.time()

rf=RandomForestRegressor(random_state=0)



label_log=np.log1p(train_label)



model=rf.fit(train_df, label_log)

linear_preds=model.predict(valid_df)



linear_preds=np.expm1(linear_preds)

        

rmse_linear = rmse_score(linear_preds, valid_label)

print ("Random Forest RMSLE is : {}".format(rmse_linear))



compare_rf_log = compare_result(preds=linear_preds, true=valid_label)



end = time.time()

duration = end - start

print ("It takes {} seconds".format(duration))
compare_rf_log.sample(5)
# sklearn LinearRegression

# Preprocess: do feature scaling or not 

from sklearn.ensemble import RandomForestRegressor



start = time.time()

rf=RandomForestRegressor(n_estimators=50, random_state=0)



label_log=np.log1p(train_label)



model=rf.fit(train_df, label_log)

linear_preds=model.predict(valid_df)



linear_preds=np.expm1(linear_preds)

        

rmse_linear = rmse_score(linear_preds, valid_label)

print ("Random Forest RMSLE is : {}".format(rmse_linear))



compare_rf50_log = compare_result(preds=linear_preds, true=valid_label)



end = time.time()

duration = end - start

print ("It takes {} seconds".format(duration))
compare_rf50_log.sample(5)