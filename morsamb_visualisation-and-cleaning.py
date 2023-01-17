import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import os
import pandas_profiling
train = pd.read_csv("../input/train.csv")
train.head()
test = pd.read_csv("../input/test.csv")
test.head()
pandas_profiling.ProfileReport(train)
pandas_profiling.ProfileReport(test)
feat_with_missing = ['meaneduc', 'rez_esc', 'v18q1', 'v2a1', 'SQBmeaned']
train[feat_with_missing].isnull().sum()/train.shape[0]
test[feat_with_missing].isnull().sum()/test.shape[0]
columns = train.columns[1:-1]
train_test = pd.concat([train[columns], test[columns]], axis=0)
train .fillna({'meaneduc': train_test.meaneduc.mean()}, inplace=True)
test.fillna({'meaneduc': train_test.meaneduc.mean()}, inplace=True)
train.fillna({'SQBmeaned': train_test.SQBmeaned.mean()}, inplace=True)
test.fillna({'SQBmeaned': train_test.SQBmeaned.mean()}, inplace=True)
train.drop(columns=['rez_esc', 'v18q1', 'v2a1'], inplace=True)
test.drop(columns=['rez_esc', 'v18q1', 'v2a1'], inplace=True)