import gc

import os

import time

import sys

import logging

import datetime

import warnings

import numpy as np

import pandas as pd

import seaborn as sns

import xgboost as xgb

import lightgbm as lgb

from scipy import stats

from scipy.signal import hann

from tqdm import tqdm_notebook

import matplotlib.pyplot as plt

from scipy.signal import hilbert

from scipy.signal import convolve

from sklearn.svm import NuSVR, SVR

from catboost import CatBoostRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import mean_absolute_error

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

from sklearn.model_selection import KFold,StratifiedKFold, RepeatedKFold

from sklearn.model_selection import cross_val_score, train_test_split, cross_val_predict

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone



warnings.filterwarnings("ignore")

import os

print(os.listdir("../input"))
train_X = pd.read_csv('../input/earthquake-data-overlapping4/train_X.csv')

train_y = pd.read_csv('../input/earthquake-data-overlapping4/train_y.csv')

test_X = pd.read_csv('../input/earthquake-data-overlapping4/test_X.csv')

train_X.head()
scaler = StandardScaler()

scaler.fit(train_X)

scaled_train_X = pd.DataFrame(scaler.transform(train_X), columns=train_X.columns)

#scaled_train_X = train_X
scaled_test_X = pd.DataFrame(scaler.transform(test_X), columns=test_X.columns)

#scaled_test_X = test_X
scaled_train_X.head(), scaled_test_X.head()
scaled_train_X.shape, scaled_test_X.shape
train_X_mean = np.mean(scaled_train_X, axis=0)

train_X_mean.head()
test_X_mean = np.mean(scaled_test_X, axis=0)

test_X_mean.head()
c = 0

for _, i in (train_X_mean - test_X_mean).abs().sort_values().iteritems():

    print(i, _)

    if _ == 'std_roll_std_50':

        print(c - 9)

    c += 1
features = (train_X_mean - test_X_mean).abs().sort_values()

features = features[features > 0]

drop_features = features.index[167:]
train_X_less = scaled_train_X.drop(drop_features, axis=1)

test_X_less = scaled_test_X.drop(drop_features, axis=1)
n_fold = 5

def mae_cv (model):

    folds = KFold(n_splits=n_fold, shuffle=True, random_state=42).get_n_splits(train_X_less.values)

    mae = -cross_val_score (model, train_X_less.values, train_y, scoring="neg_mean_absolute_error",

                           verbose=0,

                           cv=folds)

    return mae
%%time

rf_model = RandomForestRegressor(n_estimators=120, n_jobs=-1, min_samples_leaf=1, 

                           max_features = "auto",max_depth=15, )

#score = mae_cv(rf_model)

#print("Random Forest score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))

rf_model
%%time

params = {'loss_function':'MAE',}

cat_model = CatBoostRegressor(iterations=1000,  eval_metric='MAE', verbose=False, **params)



#score = mae_cv(cat_model)

#print("Cat Boost score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))

cat_model
%%time

#ENet_model = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=0.9, random_state=3,max_iter=5000))

ENet_model = ElasticNet(alpha=0.0005, l1_ratio=0.9, random_state=3,max_iter=5000)

#score = mae_cv(ENet_model)

#print("Elastic Net score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))

ENet_model
%%time

lasso_model = Lasso(alpha =0.0005, random_state=1)

#score = mae_cv(lasso_model)

#print("Lasso score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))

lasso_model
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, models):

        self.models = models

        

    # we define clones of the original models to fit the data in

    def fit(self, X, y):

        self.models_ = [clone(x) for x in self.models]

        

        # Train cloned base models

        for model in self.models_:

            model.fit(X, y)



        return self

    

    #Now we do the predictions for cloned models and average them

    def predict(self, X):

        

        predictions = np.column_stack([

            model.predict(X) for model in self.models_

        ])

        return np.mean(predictions, axis=1)   
%%time

averaged_models = AveragingModels(models = (rf_model,cat_model))

#score_rf_cat = mae_cv(averaged_models_rf_cat)

#print(" Averaged base models score_rf_cat: {:.4f} ({:.4f})\n".format(score_rf_cat.mean(), score_rf_cat.std()))



#averaged_models_cat_enet = AveragingModels(models = (cat_model, ENet_model))

#score_cat_enet = mae_cv(averaged_models_cat_enet)

#print(" Averaged base models score_cat_enet: {:.4f} ({:.4f})\n".format(score_cat_enet.mean(), score_cat_enet.std()))



#averaged_models_cat_lasso = AveragingModels(models = (cat_model,lasso_model))

#score_cat_lasso = mae_cv(averaged_models_cat_lasso)

#print(" Averaged base models score_cat_lasso: {:.4f} ({:.4f})\n".format(score_cat_lasso.mean(), score_cat_lasso.std()))
averaged_models.fit (train_X_less.values, train_y)

averaged_train_predict = averaged_models.predict(train_X_less.values)

print(mean_absolute_error(train_y, averaged_train_predict))
averaged_prediction = np.zeros(len(test_X_less))

averaged_prediction += averaged_models.predict(test_X_less.values)

averaged_prediction
submission = pd.read_csv('../input/LANL-Earthquake-Prediction/sample_submission.csv', index_col='seg_id')

submission.head()
submission.time_to_failure = averaged_prediction

submission.to_csv('submission_average_167_features.csv',index=True)