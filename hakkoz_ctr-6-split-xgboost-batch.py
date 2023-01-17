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
dtype_mapping = pd.read_csv("../input/ctr-train-test-split-0/dtype_mapping.csv", index_col=0)

dtype_mapping = dtype_mapping.iloc[:,0].to_dict()
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

for chunk in tqdm(pd.read_csv('../input/ctr-train-test-split-1/train_df1.csv', chunksize=chunksize,index_col=0, dtype=dtype_mapping), total=51):

    #chunk = chunk.drop(columns = ["label"])

    #chunk['communication_onlinerate'] = chunk['communication_onlinerate'].cat.codes

    chunk = chunk.iloc[:, 35:]

    dtrain = xgboost.DMatrix(chunk.drop(columns = ["label_tenc"]), label=chunk["label_tenc"])

    model = xgboost.train(params, dtrain, xgb_model=model)

    

for chunk in tqdm(pd.read_csv('../input/ctr-train-test-split-1-5/train_df1_5.csv', chunksize=chunksize,index_col=0, dtype=dtype_mapping), total=50):

    #chunk = chunk.drop(columns = ["label"])

    #chunk['communication_onlinerate'] = chunk['communication_onlinerate'].cat.codes

    chunk = chunk.iloc[:, 35:]

    dtrain = xgboost.DMatrix(chunk.drop(columns = ["label_tenc"]), label=chunk["label_tenc"])

    model = xgboost.train(params, dtrain, xgb_model=model)

    

for chunk in tqdm(pd.read_csv('../input/ctr-train-test-split-2/train_df3.csv', chunksize=chunksize,index_col=0, dtype=dtype_mapping), total=50):

    #chunk = chunk.drop(columns = ["label"])

    #chunk['communication_onlinerate'] = chunk['communication_onlinerate'].cat.codes

    chunk = chunk.iloc[:, 35:]

    dtrain = xgboost.DMatrix(chunk.drop(columns = ["label_tenc"]), label=chunk["label_tenc"])

    model = xgboost.train(params, dtrain, xgb_model=model)



for chunk in tqdm(pd.read_csv('../input/ctr-train-test-split-2-5/train_df2_5.csv', chunksize=chunksize,index_col=0, dtype=dtype_mapping), total=50):

    #chunk = chunk.drop(columns = ["label"])

    #chunk['communication_onlinerate'] = chunk['communication_onlinerate'].cat.codes

    chunk = chunk.iloc[:, 35:]

    dtrain = xgboost.DMatrix(chunk.drop(columns = ["label_tenc"]), label=chunk["label_tenc"])

    model = xgboost.train(params, dtrain, xgb_model=model)



for chunk in tqdm(pd.read_csv('../input/ctr-train-test-split-3/train_df3.csv', chunksize=chunksize,index_col=0, dtype=dtype_mapping), total=50):

    #chunk = chunk.drop(columns = ["label"])

    #chunk['communication_onlinerate'] = chunk['communication_onlinerate'].cat.codes

    chunk = chunk.iloc[:, 35:]

    dtrain = xgboost.DMatrix(chunk.drop(columns = ["label_tenc"]), label=chunk["label_tenc"])

    model = xgboost.train(params, dtrain, xgb_model=model)

    

for chunk in tqdm(pd.read_csv('../input/ctr-train-test-split-3-5/train_df3_5.csv', chunksize=chunksize,index_col=0, dtype=dtype_mapping), total=50):

    #chunk = chunk.drop(columns = ["label"])

    #chunk['communication_onlinerate'] = chunk['communication_onlinerate'].cat.codes

    chunk = chunk.iloc[:, 35:]

    dtrain = xgboost.DMatrix(chunk.drop(columns = ["label_tenc"]), label=chunk["label_tenc"])

    model = xgboost.train(params, dtrain, xgb_model=model)

    

for chunk in tqdm(pd.read_csv('../input/ctr-train-test-split-4v2/train_df4', chunksize=chunksize,index_col=0, dtype=dtype_mapping), total=50):

    #chunk = chunk.drop(columns = ["label"])

    #chunk['communication_onlinerate'] = chunk['communication_onlinerate'].cat.codes

    chunk = chunk.iloc[:, 35:]

    dtrain = xgboost.DMatrix(chunk.drop(columns = ["label_tenc"]), label=chunk["label_tenc"])

    model = xgboost.train(params, dtrain, xgb_model=model)

    

for chunk in tqdm(pd.read_csv('../input/ctr-train-test-split-4-5/train_df4_5', chunksize=chunksize,index_col=0, dtype=dtype_mapping), total=50):

    #chunk = chunk.drop(columns = ["label"])

    #chunk['communication_onlinerate'] = chunk['communication_onlinerate'].cat.codes

    chunk = chunk.iloc[:, 35:]

    dtrain = xgboost.DMatrix(chunk.drop(columns = ["label_tenc"]), label=chunk["label_tenc"])

    model = xgboost.train(params, dtrain, xgb_model=model)

    

    
pred_list=[]

y_test_list=[]

for chunk in tqdm(pd.read_csv('../input/ctr-train-test-split-0/test_df.csv', chunksize=chunksize,index_col=0, dtype=dtype_mapping), total=20):

    y_test_list.append(chunk["label"])

    #chunk = chunk.drop(columns = ["label"])

    #chunk['communication_onlinerate'] = chunk['communication_onlinerate'].cat.codes

    chunk = chunk.iloc[:, 35:]

    dtrain = xgboost.DMatrix(chunk.drop(columns = ["label_tenc"]), label=chunk["label_tenc"])

    y_pred = model.predict(dtrain)

    pred_list.append(y_pred)

    

    

pred_array = np.concatenate(pred_list)

y_test = pd.concat(y_test_list)

results, false = display_test_scores_v2(y_test, np.where(pred_array > 0.7, 1, 0))

print(results)
# save model to file

pickle.dump(model, open("xgboost.pkl", "wb"))