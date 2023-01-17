# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#imports
import pandas as pd
import numpy as np
import statistics as st
import matplotlib.pyplot as plt
import random
from skimage import io
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.manifold import TSNE

import gc
import lightgbm as lgb

import time

plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }
print('about to read')
train = pd.read_csv("../input/train.csv", skiprows=range(1,179903890), nrows=5000000, dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'])
test = pd.read_csv("../input/test.csv", dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])
print('done reading')
len_train = len(train)
train=train.append(test)
del test
gc.collect()
train['hour'] = pd.to_datetime(train['click_time']).dt.hour.astype('uint8')
train['day'] = pd.to_datetime(train['click_time']).dt.day.astype('uint8')
gc.collect()
def groupby_comb(comb, att, newname, opt):
    print('about to add feature')
    global train
    combatt = np.concatenate((comb, att), axis=0)
    if(opt == 1):
        newdf = train[combatt].groupby(by=comb)[att].count().reset_index().rename(index=str, columns={att[0]: newname})
    elif(opt == 2):
        newdf = train[combatt].groupby(by=comb)[att].mean().reset_index().rename(index=str, columns={att[0]: newname})
    else:
        newdf = train[combatt].groupby(by=comb)[att].var().reset_index().rename(index=str, columns={att[0]: newname})
    train = train.merge(newdf, on=comb, how='left')
    del newdf
    gc.collect
    print('added feature')
groupby_comb(['ip'], ['channel'], 'a', 1)
groupby_comb(['ip','day','hour'], ['channel'], 'b', 1)
groupby_comb(['ip', 'app'], ['channel'], 'c', 1)
groupby_comb(['ip', 'app', 'os'], ['channel'], 'd', 1)
groupby_comb(['ip','day','channel'], ['hour'], 'e', 3)
groupby_comb(['ip', 'app', 'os'], ['hour'], 'f', 3)
groupby_comb(['ip', 'app', 'channel'], ['day'], 'g', 3)
groupby_comb(['ip', 'app', 'channel'], ['hour'], 'h', 2)
test = train[len_train:]
train = train[:len_train]
target = 'is_attributed'
predictors = ['app','device','os', 'channel', 'hour', 'day', 
              'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
categorical = ['app', 'device', 'os', 'channel', 'hour', 'day']
lgbtrain = lgb.Dataset(train[predictors].values, label=train[target].values,
                      feature_name=predictors,
                      categorical_feature=categorical)
                      
lgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'subsample_for_bin': 200000,  # Number of samples for constructing bin
    'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
    'reg_alpha': 0,  # L1 regularization term on weights
    'reg_lambda': 0,  # L2 regularization term on weights
    'nthread': 4,
    'verbose': 0,
    'metric':'auc',     
 
    'learning_rate': 0.15,
    'num_leaves': 7,  # 2^max_depth - 1
    'max_depth': 3,  # -1 means no limit
    'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
    'max_bin': 100,  # Number of bucketed bin for feature values
    'subsample': 0.7,  # Subsample ratio of the training instance.
    'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
    'colsample_bytree': 0.9,  # Subsample ratio of columns when constructing each tree.
    'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
    'scale_pos_weight':99
}

evals_results = {}
num_boost_round = 250
early_stopping_rounds = 30

print('about to train')
booster = lgb.train(
     lgb_params, 
     lgbtrain, 
     valid_sets=[lgbtrain], 
     valid_names=['train'], 
     evals_result=evals_results, 
     num_boost_round=num_boost_round,
     early_stopping_rounds=early_stopping_rounds,
     verbose_eval=1)
print('trained')

del train
gc.collect()
submission = pd.DataFrame()
submission['click_id'] = test.click_id.astype('uint32')
print('about to predict')
submission['is_attributed'] = booster.predict(test[predictors])
print('predicted')
print(submission.head())
print('about to write')
submission.to_csv('sub.csv', index=False, float_format='%.10f')
print('written successfully')