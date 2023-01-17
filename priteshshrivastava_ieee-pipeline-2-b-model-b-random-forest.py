import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import math

from sklearn.metrics import roc_auc_score

import pickle

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import fastai_structured as fs



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_X = pd.read_pickle("/kaggle/input/ieee-pipeline-1-create-validation-set/train_X.pkl")

train_y = pd.read_csv("/kaggle/input/ieee-pipeline-1-create-validation-set/train_y.csv")
train_y.head()
val_X = pd.read_pickle("/kaggle/input/ieee-pipeline-1-create-validation-set/val_X.pkl")

val_y = pd.read_csv("/kaggle/input/ieee-pipeline-1-create-validation-set/val_y.csv")
test_df = pd.read_pickle("/kaggle/input/ieee-pipeline-1-create-validation-set/test_df.pkl")

test_df.head()
fs.train_cats(train_X)

fs.apply_cats(val_X, train_X)

fs.apply_cats(test_df, train_X)
nas = {}

df_trn, _, nas = fs.proc_df(train_X, na_dict=nas)   ## Avoid creating NA columns as total cols may not match later

df_test, _, _ = fs.proc_df(test_df, na_dict=nas)

df_val, _, _ = fs.proc_df(val_X, na_dict = nas)

df_trn.head()
def auc(x,y): 

    return roc_auc_score(x,y)

def print_score(m):

    res = [auc(m.predict(df_trn), train_y), auc(m.predict(df_val), val_y)]

    print(res)
modelB = RandomForestClassifier(n_estimators=30, min_samples_leaf=20, max_features=0.7, 

                                n_jobs=-1, oob_score=True) ## Use all CPUs available
modelB.fit(df_trn, train_y)
print_score(modelB)
predsB = pd.Series(modelB.predict(df_val))
test_predsB = pd.Series(modelB.predict(df_test))
predsB.to_csv("predsB.csv", index = False, header = True)

test_predsB.to_csv("test_predsB.csv", index = False, header = True)
sample_submission = pd.read_csv("/kaggle/input/ieee-fraud-detection/sample_submission.csv")

sample_submission['isFraud'] = modelB.predict_proba(df_test)[:,1]   

#sample_submission['isFraud'] = modelB.predict(df_test)

sample_submission.to_csv('simple_RF.csv', index=False)