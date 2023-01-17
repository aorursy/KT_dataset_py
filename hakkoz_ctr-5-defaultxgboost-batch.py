import pickle

import pandas as pd

import numpy as np

import string

import gzip

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import StratifiedShuffleSplit

import xgboost 

from sklearn.model_selection import GridSearchCV

from tqdm import tqdm



#import utility functions

from utils3 import display_test_scores_v2
chunksize = 10 ** 5

params = {#"tree_method": "gpu_hist",

            "objective": "binary:logistic",

            "random_state" : 0, 

            #"n_estimators" : 100,

            "max_depth" : 5,

            "subsample" : 0.5, 

            "colsample_bytree" : 0.5, 

            "colsample_bylevel" : 0.5, 

            "colsample_bynode" : 0.5,

         }



model = None

for chunk in tqdm(pd.read_csv('../input/ctr-targetencoding-smoothing-1/first_df.csv', chunksize=chunksize,index_col=0)):

    dtrain = xgboost.DMatrix(chunk, label=chunk["label"])

    model = xgboost.train(params, dtrain, xgb_model=model)

    

for chunk in tqdm(pd.read_csv('../input/ctr-targetencoding-smoothing-2/second_df.csv', chunksize=chunksize,index_col=0)):

    dtrain = xgboost.DMatrix(chunk, label=chunk["label"])

    model = xgboost.train(params, dtrain, xgb_model=model)

    

for chunk in tqdm(pd.read_csv('../input/ctr-targetencoding-smoothing-3/third_df.csv', chunksize=chunksize,index_col=0)):

    dtrain = xgboost.DMatrix(chunk, label=chunk["label"])

    model = xgboost.train(params, dtrain, xgb_model=model)
pred_list=[]

y_test_list=[]

for chunk in tqdm(pd.read_csv('../input/ctr-targetencoding-smoothing-4/fourth_df.csv', chunksize=chunksize,index_col=0)):

    dtrain = xgboost.DMatrix(chunk, label=chunk["label"])

    y_pred = model.predict(dtrain)

    pred_list.append(y_pred)

    y_test_list.append(chunk["label"])

    

pred_array = np.concatenate(pred_list)

y_test = pd.concat(y_test_list)

results, false = display_test_scores_v2(np.where(y_test > 0.5, 1, 0), np.where(pred_array > 0.5, 1, 0))

print(results)