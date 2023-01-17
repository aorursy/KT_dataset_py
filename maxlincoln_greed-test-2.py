%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn import linear_model, datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from collections import OrderedDict
import numpy as np
import pandas as pd
import scipy as sp
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt
from matplotlib import pyplot
import seaborn as sns
from itertools import product
from xgboost import XGBRegressor
from xgboost import plot_importance
from xgboost import XGBClassifier
from numpy import loadtxt
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
dataset = pd.read_csv('../input/matrix2/data.csv')
teste = pd.read_csv('../input/conjuntotest/test.csv')
dataset
X_train = dataset[dataset.date_block_num < 33].drop(['item_cnt_month'], axis=1)
Y_train = dataset[dataset.date_block_num < 33]['item_cnt_month']
X_valid = dataset[dataset.date_block_num == 33].drop(['item_cnt_month'], axis=1)
Y_valid = dataset[dataset.date_block_num == 33]['item_cnt_month']
X_test = dataset[dataset.date_block_num == 34].drop(['item_cnt_month'], axis=1)

'''model = XGBClassifier()'''
model = XGBRegressor(
    max_depth=7,
    n_estimators=100,
    min_child_weight=300, 
    colsample_bytree=0.8, 
    subsample=0.8, 
    eta=0.3,    
    seed=42)

model.fit(
    X_train, 
    Y_train, 
    eval_metric="rmse", 
    eval_set=[(X_train, Y_train), (X_valid, Y_valid)], 
    verbose=True, 
    early_stopping_rounds = 10)
'''
label_encoded_y = LabelEncoder().fit_transform(Y_train)

n_estimators = [10, 60, 110, 160]
max_depth = [2, 4, 6, 8]
print(max_depth)
param_grid = dict(max_depth=max_depth, n_estimators=n_estimators)
param_grid = dict(max_depth=max_depth, n_estimators=n_estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold, verbose=1)
grid_result = grid_search.fit(X_train, label_encoded_y)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
	print("%f (%f) with: %r" % (mean, stdev, param))

'''
Y_pred = model.predict(X_valid).clip(0, 20)
Y_test = model.predict(X_test).clip(0, 20)
submission = pd.DataFrame({
    "ID": teste.index, 
    "item_cnt_month": Y_test
})
submission.to_csv('xgb_submission7-100.csv', index=False)