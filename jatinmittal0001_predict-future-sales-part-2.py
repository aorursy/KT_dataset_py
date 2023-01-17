import warnings

warnings.filterwarnings('ignore')

import gc

import pickle

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import math

from itertools import product

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.preprocessing import StandardScaler, RobustScaler

from xgboost import XGBRegressor

from sklearn.linear_model import SGDRegressor

from sklearn.svm import SVR

import matplotlib.pyplot as plt

import seaborn as sns

from xgboost import plot_importance

from matplotlib.pyplot import figure

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression

def plot_features(booster, figsize):    

    fig, ax = plt.subplots(1,1,figsize=figsize)

    return plot_importance(booster=booster, ax=ax)

import os

print(os.listdir("../input/predict-future-sales-part-1"))



# Any results you write to the current directory are saved as output.
#Importing datasets

sales_train = pd.read_csv("../input/competitive-data-science-predict-future-sales/sales_train.csv")

test = pd.read_csv("../input/competitive-data-science-predict-future-sales/test.csv")

test.drop('ID',axis=1,inplace=True)
test_shop_item_pair = test[(test.shop_id==10)]

test_shop_item_pair.loc[test_shop_item_pair.shop_id == 10, 'shop_id']= 11

test.loc[test.shop_id == 10, 'shop_id']= 11
def reduce_size(df):

    float_cols = [c for c in df if df[c].dtype == "float64"]

    int_cols = [c for c in df if df[c].dtype in ["int64", "int32"]]

    df[float_cols] = df[float_cols].astype(np.float32)

    df[int_cols] = df[int_cols].astype(np.int16)

    return df
# to open the dataset from previous kernel

all_data1 = pickle.load(open("../input/predict-future-sales-part-1/all_data1","rb"))
feat_to_keep = ['date_block_num', 'shop_id', 'item_id', 'item_cnt_month',

       'item_category_id', 'subtype', 'db_avg_items_sold_lag_1',

       'db_shop_cat_avg_items_sold_lag_1', 'db_item_id_items_sold_lag_1',

       'db_item_id_items_sold_lag_2', 'db_item_id_items_sold_lag_3',

       'db_item_id_items_sold_lag_6','city_target_enc',

       'item_id_target_enc', 'month_target_enc','db_shop_city_avg_items_sold_lag_1',

       'db_shop_city_avg_items_sold_lag_2', 'db_city_avg_items_sold_lag_1',

       'month', 'item_months_since_first_sale','item_shop_last_sale',

       'db_item_avg_price_lag_1','delta_price_lag', 'delta_price_lag_1',

        'delta_price_lag_3','max_cnt_lag_1', 'max_cnt_lag_3','max_cnt_lag_6',  'revenue_shop_lag_2']
all_data1 = all_data1.loc[:,feat_to_keep]
X_train = all_data1[all_data1.date_block_num < 33].drop(['item_cnt_month'], axis=1)

Y_train = all_data1[all_data1.date_block_num < 33]['item_cnt_month']

X_valid = all_data1[all_data1.date_block_num == 33].drop(['item_cnt_month'], axis=1)

Y_valid = all_data1[all_data1.date_block_num == 33]['item_cnt_month']

X_test = all_data1[all_data1.date_block_num == 34].drop(['item_cnt_month'], axis=1)



del all_data1

gc.collect();
#Defining function to calculate RMSE, for manual evaluation

from sklearn.metrics import mean_squared_error

from math import sqrt



def rmse(y_pred, y_test):

    rmse = sqrt(mean_squared_error(y_test,y_pred))

    return print(rmse)
def predict(model,name):

    Y_test = model.predict(X_test).clip(0., 20.)

    results = X_test.loc[:,['shop_id', 'item_id']]

    results['prediction'] = Y_test

    if len(test.columns)==3:

        test.drop('ID',axis=1,inplace=True)

    sub = pd.merge(test, results, on = ['shop_id', 'item_id'], how='left')

    submission = pd.DataFrame({

        "ID": test.index, 

        "item_cnt_month": sub['prediction']

    })

    file_name = str(name) + '_submission.csv'

    submission.to_csv(file_name, index=False)
#base model-1



xgb = XGBRegressor(

    max_depth=8,

    n_estimators=45,

    min_child_weight=300, 

    colsample_bytree=0.9, 

    subsample=0.8, 

    eta=0.3,    

    seed=4)



xgb.fit(X_train, Y_train, eval_set=[(X_train, Y_train), (X_valid, Y_valid)],

        eval_metric = 'rmse', early_stopping_rounds =10,verbose=True)



predict(xgb,'xgb')



plot_features(xgb, (10,30))



'''







model1_train = xgb.predict(X_train)

model1_valid = xgb.predict(X_valid)

model1_test = xgb.predict(X_test)





del xgb

gc.collect();



#base model-2

svr_model= SVR(kernel='rbf', degree=10, verbose=True, max_iter = 50)



svr_model.fit(X_train, Y_train)

model2_train = svr_model.predict(X_train)

model2_valid = svr_model.predict(X_valid)

model2_test = svr_model.predict(X_test)







del svr_model

gc.collect();



#base model-3

from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.05, max_iter=100)

lasso.fit(X_train, Y_train)

model3_train = lasso.predict(X_train)

model3_valid = lasso.predict(X_valid)

model3_test = lasso.predict(X_test)





del lasso

gc.collect();



#stacking

base_predictions_train = pd.DataFrame( {'XGBoost': model1_train.ravel(),

     'SVR model': model2_train.ravel(),

     'lasso': model3_train.ravel()

     #'SVM': model4_train.ravel()

    })

base_predictions_test = pd.DataFrame( {'XGBoost': model1_test.ravel(),

     'SVR model': model2_test.ravel(),

     'lasso': model3_test.ravel()

     #'SVM': model4_test.ravel()

    })



base_predictions_valid = pd.DataFrame( {'XGBoost': model1_valid.ravel(),

     'SVR model': model2_valid.ravel(),

     'lasso': model3_valid.ravel()

     #'SVM': model4_valid.ravel()

    })



X_new_train = base_predictions_train.as_matrix()

X_new_valid = base_predictions_valid.as_matrix()

X_new_test = base_predictions_test.as_matrix()

base_predictions_train.head()



#heatmap to see correlation between different predictions

sns.heatmap(base_predictions_train.astype(float).corr(),

            linewidths=0.1,vmax=1.0, square=True, linecolor='white', annot=True)

'''
'''

# META Model

lm = LinearRegression()

lm.fit(X_new_train,Y_train)

y_valid_pred = lm.predict(X_new_valid)

plt.plot(y_valid_pred, '.', Y_valid, 'x')

#y_test_pred = lm.predict(X_new_test)



Y_test = lm.predict(X_new_test).clip(0., 20.)

results = X_test.loc[:,['shop_id', 'item_id']]

results['prediction'] = Y_test

if len(test.columns)==3:

    test.drop('ID',axis=1,inplace=True)

sub = pd.merge(test, results, on = ['shop_id', 'item_id'], how='left')

submission = pd.DataFrame({

    "ID": test.index, 

    "item_cnt_month": sub['prediction']

})

submission.to_csv('stack_submission.csv', index=False)

'''
'''

from catboost import CatBoostRegressor



model = CatBoostRegressor(

    random_seed=63,

    iterations=200,

    learning_rate=0.12,

    depth=4,

    loss_function='RMSE',

    rsm = 0.85,

    od_type='Iter',

    od_wait=20,

)

model.fit(

    X_train, Y_train,

    logging_level='Silent',

    eval_set=(X_valid, Y_valid),

    plot=True

)



importances = model.get_feature_importance(prettified=True)

feature_labels = []

feature_value = []

for i in range(0,len(importances)):

    feature_labels.append(importances[i][0])

    feature_value.append(importances[i][1])

    

fig, ax = plt.subplots(1,1,figsize=(10,20))

sns.barplot(y = feature_labels,  x=feature_value, ax=ax)



predict(model,'catboost')

'''