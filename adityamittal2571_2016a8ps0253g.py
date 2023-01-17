#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""

Created on Mon Mar 23 14:08:22 2020



"""

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split as tts

from sklearn.linear_model import Lasso as la

from sklearn.linear_model import LinearRegression as lr

from sklearn.preprocessing import StandardScaler as ss

from sklearn.metrics import mean_squared_error as mse

from sklearn.ensemble import RandomForestRegressor as rfr

from sklearn.feature_selection import SelectFromModel as sfm

import seaborn as sns



df = pd.read_csv('/kaggle/input/bits-f464-l1/train.csv')

X_train, X_test, y_train, y_test = tts(

    df.drop(labels=['label', 'id'], axis=1),

    df['label'],

    test_size=0.4,

    random_state=0)



test_df = pd.read_csv('/kaggle/input/bits-f464-l1/test.csv')



#scalar = ss()

#X_train_scaled = scalar.fit_transform(X_train)

#X_test_scaled = scalar.fit_transform(X_test)



def evaluate(model, X_train, y_train, X_test, y_test):

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    rmse = np.sqrt(((y_pred-y_test)**2).mean())

    return rmse, y_pred



#model_lr = lr()

#rmse_baseline, y_pred_baseline = evaluate(model_lr, X_train, y_train, X_test, y_test)

#

model_rf = rfr(bootstrap=True, criterion='mse',

                      max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0,

                      min_impurity_split=None, min_samples_leaf=1,

                      min_samples_split=2, min_weight_fraction_leaf=0.0,

                      n_estimators=100, n_jobs=None, oob_score=False,

                      random_state=None, verbose=0, warm_start=False)

rmse_rf_base, y_pred_rmse_base = evaluate(model_rf, X_train, y_train, X_test, y_test)
rmse_rf_base
df_train_X = df.drop(columns = ['label', 'id'])

df_train_y = df['label']

model_rf.fit(df_train_X, df_train_y)

test_pred = model_rf.predict(test_df.drop(columns = ['id']))
answer = {'id' : test_df['id'], 'label' : test_pred}

ans = pd.DataFrame(answer, columns = ['id', 'label'])

ans.to_csv("answer.csv", index = False)