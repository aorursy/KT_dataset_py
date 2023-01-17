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
#All the imports are added here

import numpy as np

import pandas as pd

import pickle

import time

import sys

import gc

import os

from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor
INPUT_DIR='../input/analysed-finaldata/'
final_tr_data=pd.read_pickle(INPUT_DIR+'final_tr_data_.pkl')
dates = final_tr_data['date_block_num']



last_block = dates.max()

print('Test `date_block_num` is {0}'.format(last_block))



X_train = final_tr_data.loc[dates <  last_block]

X_test =  final_tr_data.loc[dates == last_block]



columns_to_delete = ['date_block_num', 'value']

X_train = X_train.drop(columns_to_delete, axis=1)

X_test = X_test.drop(columns_to_delete, axis=1)



X_train = X_train.values

X_test = X_test.values



y_train = final_tr_data.loc[dates <  last_block, 'value'].values

y_test =  final_tr_data.loc[dates == last_block, 'value'].values



#use block 33 as validation month block

X_valid_train = final_tr_data.loc[dates <  last_block-1]

X_valid_test =  final_tr_data.loc[dates == last_block-1]



columns_to_delete = ['date_block_num', 'value']

X_valid_train = X_valid_train.drop(columns_to_delete, axis=1)

X_valid_test = X_valid_test.drop(columns_to_delete, axis=1)





X_valid_train = X_valid_train.values

X_valid_test  = X_valid_test.values



y_valid_train = final_tr_data.loc[dates <  last_block-1, 'value'].values

y_valid_test =  final_tr_data.loc[dates == last_block-1, 'value'].values



del dates

del final_tr_data

gc.collect()
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(X_train, y_train)

pred_lr = lr.predict(X_test).clip(0,20)
#LB Score 1.05

submit = pd.DataFrame({'ID':range(len(pred_lr)), 'item_cnt_month': pred_lr})

submit.to_csv('submit_lr.csv', index=False)
from itertools import product

def validate(estimator, X_train_, y_train_, X_val_, y_val_, grid_params):

    keys = grid_params.keys()

    vals = grid_params.values()

    parameters = []

    rmses = []

    rmses_train = []

    return_obj ={}

    prods = product(*vals)

   

    for idx, instance in enumerate(prods):

        print('-'*50)

        print('model {0}:'.format(idx))

        model_params = dict(zip(keys, instance))

        parameters.append(model_params)

        

        print(model_params)

        model = estimator(**model_params)

        model.fit(X_train_, y_train_)

            

        pred_test = model.predict(X_val_)       

        mse = mean_squared_error(y_val_, pred_test)

        rmse = np.sqrt(mse)

        print('RMSE: {0}'.format(rmse))

        rmses = rmses + [rmse]

        

        best_rmse_so_far = np.min(rmses)

        print('Best rmse so far: {0}'.format(best_rmse_so_far))

        best_model_params_so_far = parameters[np.argmin(rmses)]

        print('Best model params so far: {0}'.format(best_model_params_so_far))

        

        del best_rmse_so_far

        del best_model_params_so_far

        del pred_test

        del model

        gc.collect()

    

    rmses = np.array(rmses)

    best_rmse = np.min(rmses)

    print('Best rmse: {0}'.format(best_rmse))

    best_model_params = parameters[np.argmin(rmses)]

    print('Best model params: {0}'.format(best_model_params))



    return_obj['rmses'] = rmses

    return_obj['best_rmse'] = best_rmse

    return_obj['best_model_params'] = best_model_params

      

    return return_obj
from sklearn.linear_model import Ridge

#lets validate with Ridge regression

alphas = [10, 100, 1000,2000, 3000, 4000, 5000]

grid_params = {'alpha':alphas}

val_res = validate(Ridge, X_valid_train, y_valid_train, 

                   X_valid_test, y_valid_test, grid_params)
best_alpha=3000

ridge_model = Ridge(best_alpha)

ridge_model.fit(X_train, y_train)

predictions = ridge_model.predict(X_test)
submit = pd.DataFrame({'ID':range(len(pred_lr)), 'item_cnt_month': pred_lr})

submit.to_csv('submit_ridge.csv', index=False)
best_params = {'learning_rate': 0.16, 'n_estimators': 500, 

               'max_depth': 6, 'min_child_weight': 7,

               'subsample': 0.9, 'colsample_bytree': 0.7, 'nthread': -1, 

               'scale_pos_weight': 1, 'random_state': 42, 

               

               #next parameters are used to enable gpu for fasting fitting

               'tree_method': 'gpu_hist', 'predictor': 'gpu_predictor', 'gpu_id': 0}
model = XGBRegressor(**best_params)
model = XGBRegressor(**best_params)

model.fit(X_valid_train, 

                y_valid_train,

                eval_metric="rmse", 

                eval_set=[(X_valid_test, y_valid_test)], 

                verbose=True, 

                early_stopping_rounds = 50)
#model.fit(X_train, 

 #               y_train,

  #              eval_metric="rmse",

   #             verbose=True)
pred = model.predict(X_test).clip(0,20)
submit = pd.DataFrame({'ID':range(len(pred)), 'item_cnt_month': pred})

submit.to_csv('submit.csv', index=False)
##save model and X_test

import pickle

import joblib

filename = 'xgboost_model.sav'

joblib.dump(model, filename)
##save model and X_test

import pickle

import joblib

filename = 'test_data.sav'

joblib.dump(X_test, filename)