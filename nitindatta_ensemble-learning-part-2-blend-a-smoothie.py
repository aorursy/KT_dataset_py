import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import pandas as pd

import numpy as np

import time

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import StandardScaler, MinMaxScaler

import eli5

from eli5.sklearn import PermutationImportance

import gc
train_set = pd.read_csv('/kaggle/input/preprocessed-sales-data/train_set.csv')

validation_set = pd.read_csv('/kaggle/input/preprocessed-sales-data/validation_set.csv')

test_set = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')
train_set.head().T
# Creating training and validation sets

x_train = train_set.drop(['item_cnt_month','date_block_num'],axis=1)

y_train = train_set['item_cnt_month'].astype(int)



x_val = validation_set.drop(['item_cnt_month', 'date_block_num'], axis=1)

y_val = validation_set['item_cnt_month'].astype(int)
latest_records = pd.concat([train_set, validation_set]).drop_duplicates(subset=['shop_id', 'item_id'], keep='last')

x_test = pd.merge(test_set, latest_records, on=['shop_id', 'item_id'], how='left', suffixes=['', '_'])

x_test['year'] = 2015

x_test['month'] = 9

x_test.drop('item_cnt_month', axis=1, inplace=True)

x_test = x_test[x_train.columns]
ts = time.time()

sets = [x_train, x_val,x_test]

for dataset in sets:

    for shop_id in dataset['shop_id'].unique():

        for column in dataset.columns:

            shop_median = dataset[(dataset['shop_id'] == shop_id)][column].median()

            dataset.loc[(dataset[column].isnull()) & (dataset['shop_id'] == shop_id), column] = shop_median



x_test.fillna(x_test.mean(), inplace=True)

print('Time taken : ',time.time()-ts)
x_test.head().T
all_f = ['shop_id', 'item_id', 'item_cnt', 'mean_item_cnt', 'transactions', 'year',

       'month', 'item_cnt_mean', 'item_cnt_std',

       'item_cnt_shifted1', 'item_cnt_shifted2', 'item_cnt_shifted3',

       'item_trend', 'shop_mean', 'item_mean', 'shop_item_mean', 'year_mean',

       'month_mean']
x_tr = x_train[all_f]

x_va = x_val[all_f]

x_te = x_test[all_f]
from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.neighbors import KNeighborsRegressor
def models(model, x_tr, y_train, x_va, x_te):

    '''

    This model is used to train the model and make predictions.

    - model : the model in usage (LR, KN, RF)

    - x_tr : x_train dataframe

    - y_train : y_train dataframe

    - x_va : x_validation dataframe

    - x_te : x_test dataframe

    '''

    

    #training the model

    model.fit(x_tr,y_train)

    

    #storing the predictions for `train` and `validation` sets

    train_pred = model.predict(x_tr)

    val_pred = model.predict(x_va)

    test_pred = model.predict(x_te)

    

    return train_pred, val_pred, test_pred
# Training the first model: Linear Regression

LR = LinearRegression(n_jobs=-1)

M1_train, M1_val, M1_test = models(LR,x_tr,y_train,x_va,x_te)
print('Train rmse for LINEAR REGRESSION:', np.sqrt(mean_squared_error(y_train, M1_train)))

val_lr = np.sqrt(mean_squared_error(y_val, M1_val))

print('Validation rmse for LINEAR REGRESSION:', np.sqrt(mean_squared_error(y_val, M1_val)))
perm = PermutationImportance(LR, random_state=42).fit(x_va, y_val)

eli5.show_weights(perm, feature_names = x_va.columns.tolist())
x_t_knn = x_tr[:100000]

y_t_knn = y_train[:100000]



scaler = MinMaxScaler()

scaler.fit(x_t_knn)

scaled_x_t_knn = scaler.transform(x_t_knn)

scaled_x_v = scaler.transform(x_va)



KN = KNeighborsRegressor(n_neighbors=20, leaf_size=15,n_jobs=-1)

M2_train, M2_val, M2_test = models(KN,scaled_x_t_knn,y_t_knn,scaled_x_v,x_te)
print('Train rmse for KNN:', np.sqrt(mean_squared_error(y_t_knn, M2_train)))

val_knn = np.sqrt(mean_squared_error(y_val, M2_val))

print('Validation rmse for KNN:', np.sqrt(mean_squared_error(y_val, M2_val)))
x_va_knn = x_va[:10000]

y_val_knn = y_val[:10000]

perm = PermutationImportance(KN, random_state=42).fit(x_va_knn, y_val_knn)

eli5.show_weights(perm, feature_names = x_va.columns.tolist())
# Training the third model: Random Forest

ts = time.time()

RF = RandomForestRegressor(n_jobs=-1,n_estimators=40, max_depth=8, random_state=42)

M3_train, M3_val,M3_test = models(RF,x_tr,y_train,x_va,x_te)

print('Total time taken : ',time.time()-ts)
print('Train rmse for RANDOM FOREST:', np.sqrt(mean_squared_error(y_train, M3_train)))

val_rf = np.sqrt(mean_squared_error(y_val, M3_val))

print('Validation rmse for RANDOM FOREST:', np.sqrt(mean_squared_error(y_val, M3_val)))
perm = PermutationImportance(RF, random_state=42).fit(x_va, y_val)

eli5.show_weights(perm, feature_names = x_va.columns.tolist())
val_predictions = {'LR': M1_val,

                     'KN': M2_val,

                     'RF': M3_val}



val_predictions = pd.DataFrame(val_predictions)
val_predictions.head(10).T
test_predictions = {'LR': M1_test,

                   'KN': M2_test,

                   'RF': M3_test}



test_predictions = pd.DataFrame(test_predictions)
test_predictions.head(10).T
# Stacking Model 

stack_model = LinearRegression()

stack_model.fit(val_predictions, y_val)



stack_val_preds = stack_model.predict(val_predictions)

stack_test_preds = stack_model.predict(test_predictions)
val_stack = np.sqrt(mean_squared_error(stack_val_preds,y_val))

print('Validation rmse for STACKING:', np.sqrt(mean_squared_error(stack_val_preds,y_val)))
perm = PermutationImportance(stack_model, random_state=42).fit(val_predictions, y_val)

eli5.show_weights(perm, feature_names = val_predictions.columns.tolist())
blend_df_valid = pd.concat([x_va,val_predictions],axis=1)

blend_df_test=pd.concat([x_te,test_predictions],axis=1)
blend_df_valid.head().T
blend_model = LinearRegression()

blend_model.fit(blend_df_valid,y_val)

blend_val_preds = blend_model.predict(blend_df_valid)

blend_test_preds = blend_model.predict(blend_df_test)
val_blend = np.sqrt(mean_squared_error(blend_val_preds,y_val))

print('Validation rmse for BLENDING :', np.sqrt(mean_squared_error(blend_val_preds,y_val)))
perm = PermutationImportance(blend_model, random_state=42).fit(blend_df_valid, y_val)

eli5.show_weights(perm, feature_names = blend_df_valid.columns.tolist())
del(LR,M1_train, M1_val, M1_test,perm)

del(x_t_knn,y_t_knn)

del(scaled_x_t_knn,scaled_x_v)

del(x_va_knn,y_val_knn)

del(KN,M2_train, M2_val, M2_test)

del(RF,M3_train,M3_val,M3_test)

gc.collect()
from sklearn.ensemble import BaggingRegressor

from sklearn import tree

from sklearn.svm import SVR



bagging_model = BaggingRegressor(base_estimator=LinearRegression(),n_estimators=20, random_state=0) 

bagging_model.fit(x_tr, y_train)



bagging_val_preds = bagging_model.predict(x_va)

bagging_test_preds = bagging_model.predict(x_te)
val_bagging = np.sqrt(mean_squared_error(bagging_val_preds,y_val))

print('Validation rmse for BAGGING :', np.sqrt(mean_squared_error(bagging_val_preds,y_val)))
from xgboost import XGBRegressor

xgb_model = XGBRegressor(random_state=42, colsample_bylevel=1,

                         colsample_bytree=0.5, learning_rate=0.1, max_depth=5,

                         n_estimators=20, n_jobs=-1, objective='reg:linear')

xgb_model.fit(x_tr,y_train,eval_metric="rmse", 

              eval_set=[(x_tr, y_train), (x_va, y_val)], 

              verbose=10, 

              early_stopping_rounds=15)
val_xgboost = 0.79400
from sklearn.ensemble import AdaBoostRegressor

ada_model = AdaBoostRegressor(random_state=42,n_estimators=20,learning_rate=0.1)



ada_model.fit(x_tr,y_train)
ada_train_preds = ada_model.predict(x_tr)

ada_val_preds = ada_model.predict(x_va)



print('Train rmse for AdaBoost :', np.sqrt(mean_squared_error(ada_train_preds,y_train)))

val_adaboost =  np.sqrt(mean_squared_error(ada_val_preds,y_val))

print('Validation rmse for AdaBoost :', np.sqrt(mean_squared_error(ada_val_preds,y_val)))
import lightgbm as lgb



train_data=lgb.Dataset(x_tr,label=y_train)

params = {'num_iterations':50,'max_depth':10,'learning_rate':0.001}

lgb_model= lgb.train(params, train_data, 100)



lgb_train_preds = lgb_model.predict(x_tr)

lgb_val_preds = lgb_model.predict(x_va)

val_lgb =  np.sqrt(mean_squared_error(lgb_val_preds,y_val))



print('Train rmse for LightGBM :', np.sqrt(mean_squared_error(lgb_train_preds,y_train)))

print('Validation rmse for LightGBM  :', val_lgb)
del(stack_model,stack_val_preds,stack_test_preds)

del(blend_model,blend_val_preds,blend_test_preds)

del(bagging_model,bagging_val_preds,bagging_test_preds)

del(xgb_model)

gc.collect()
x_tr['shop_id'] = x_tr['shop_id'].astype(int)

x_tr['item_id'] = x_tr['item_id'].astype(int)

x_tr['year'] = x_tr['year'].astype(int)

x_tr['month'] = x_tr['month'].astype(int)



x_va['shop_id'] = x_va['shop_id'].astype(int)

x_va['item_id'] = x_va['item_id'].astype(int)

x_va['year'] = x_va['year'].astype(int)

x_va['month'] = x_va['month'].astype(int)

from catboost import CatBoostRegressor

features =  [0, 1, 5, 6]

cat_model = CatBoostRegressor(iterations=100, verbose=50, depth=5)

cat_model.fit(x_tr, y_train,cat_features=features,eval_set=(x_va, y_val))
cat_train_preds = cat_model.predict(x_tr)

cat_val_preds = cat_model.predict(x_va)



val_cat = np.sqrt(mean_squared_error(cat_val_preds,y_val))

print('Train rmse for CatBoost :', np.sqrt(mean_squared_error(cat_train_preds,y_train)))

print('Validation rmse for CatBoost  :', val_cat)
RMSE = [val_lr, val_knn, val_rf, val_stack, val_blend, val_bagging, val_xgboost, val_adaboost, val_lgb, val_cat]

import seaborn as sns 

import matplotlib.pyplot as plt

y_ax = ['Linear Regression' ,'KNN', 'Random Forest Regression','Stacking', 'Blending','Bagging' ,'XGBoost', 'AdaBoost', 'LightGBM','CatBoost']

x_ax = RMSE

out = pd.DataFrame()

out['RMSE'] = RMSE

out['Algorithms'] = y_ax
# sns.barplot(x=x_ax,y=y_ax,linewidth=1.5,edgecolor="0.1")

# plt.xlabel('RMSE')

import plotly.express as px



fig = px.bar(out,y=out['RMSE'], x=out['Algorithms'], 

             color=out['Algorithms'])

fig.update_layout(title='Algorithms with RMSE', title_x=0.5)

fig.show()