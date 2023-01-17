# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import ExtraTreesClassifier

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df_test = pd.read_csv("../input/application_test.csv")
df_train = pd.read_csv("../input/application_train.csv")
###Example down sampling, returns indexes
def down_sampling (df, target_variable, random_state, class_keep=1, class_down=0):
    pos = df[df[target_variable]==class_keep]
    pos_ratio = float(len(pos))/len(df)
    neg = df[df[target_variable]==class_down].sample(frac=pos_ratio/(1-pos_ratio), replace=False, random_state=random_state)
    data_index = pos.index.union(neg.index).sort_values()
    return data_index

###Example feature generator
def simple_feature_generator(df):
    feat_cols = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    for col in feat_cols:
        df.loc[df[col].isnull(), col] = df[col].mean()
    return df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']]
    
###Example cross validation split function
def skf_cross_val(X, y, nfolds):
    skf = StratifiedKFold(n_splits=nfolds)
    return skf.split(X, y)

###Example predictor for linear regression
def lr_fit_predictor(X_train, y_train, X_test):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    return lr.predict(X_test)

###Example predictor for Extratrees
def extrees_fit_predictor(X_train, y_train, X_test):
    ext = ExtraTreesClassifier(n_estimators=100, min_samples_split = 1000)
    ext.fit(X_train, y_train)
    return ext.predict_proba(X_test)[:, 1]
    
###Combines helper functions to output prediction
def crossval_predict(df_train, df_test, feature_gen, crossval_split, fit_predictor):
    df = feature_gen(pd.concat([df_train, df_test]))
    
    df_Xtrain = df.iloc[:len(df_train)]
    df_ytrain = df_train['TARGET']
    
    df_Xtest = df.iloc[len(df_train):]
    
    skf = crossval_split(df_Xtrain.values, df_ytrain.values, 5)
    
    auc = []
    for idx, (train_index, test_index) in enumerate(skf):
        pred = fit_predictor(df_Xtrain.iloc[train_index].values, 
                         df_ytrain.iloc[train_index].values,
                         df_Xtrain.iloc[test_index].values)
        score = roc_auc_score(df_ytrain[test_index].values, pred)
        auc.append(score)
        print('Fold {} AUC: {}'.format(idx, score))
    print('\nMean AUC: {}'.format(np.mean(auc)))
    
    pred = fit_predictor(df_Xtrain.values, 
                         df_ytrain.values, 
                         df_Xtest.values)
    
    df_test['TARGET'] = pred
    return df_test[['SK_ID_CURR', 'TARGET']]

#for each model, you get a different random seed
bagging_size = 5
random_states = list(range(bagging_size))
data_dict = {}
for state in random_states:
    data_dict[state] = down_sampling(df=df_train, target_variable='TARGET', random_state=state)

#Creating a sample data for a model
data_train = df_train.iloc[data_dict[1]]
data_train.sample(10,replace=False)
output = crossval_predict(df_train=df_train, 
                          df_test=df_test, 
                          feature_gen=simple_feature_generator,
                          crossval_split=skf_cross_val,
                          fit_predictor=extrees_fit_predictor)
output.to_csv('extratrees_simple.csv', index=False)
