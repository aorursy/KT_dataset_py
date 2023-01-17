# LOAD PACKAGE



import numpy as np

import pandas as pd





import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))





# SETTING MAXIMUM DISPLAY

pd.options.display.max_columns = 999

pd.options.display.max_rows = 999

pd.options.display.width = 1000





# SETTING TO IGNORE ERROR

pd.set_option('mode.chained_assignment', None)



import warnings

warnings.filterwarnings("ignore")





# BASIC PACKAGE FOR TIME, MATH, ETC

import gc

import json

import math

from time import time

from tqdm import tqdm_notebook as tqdm

from collections import Counter

from scipy import stats

from sklearn.utils import shuffle





# FEATURE IMPORTANT

import eli5

import shap





# VISUALIZATION

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

# from matplotlib import rcParams

# rcParams['figure.figsize'] = 5,6





# CORRELATION

from numpy import cov

from scipy.stats import pearsonr

from scipy.stats import spearmanr





# PIPELINE

from sklearn.pipeline import Pipeline

from sklearn.pipeline import FeatureUnion

from sklearn.preprocessing import FunctionTransformer





# PREPROCESSING - IMPUTATIONS

# from sklearn.preprocessing import Imputer

from sklearn.impute import SimpleImputer





# PREPROCESSING - ENCODER

from sklearn import preprocessing

from sklearn.preprocessing import LabelEncoder, OneHotEncoder





# PREPROCESSING - SCALING

from sklearn.preprocessing import scale

from sklearn.preprocessing import StandardScaler, MinMaxScaler





# CROSS VALIDATION

from sklearn.model_selection import train_test_split, GroupKFold, KFold, StratifiedKFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV





# MODEL ALGORITHM - BASIC FOR LINEAR AND CLASSFICATION

from sklearn import linear_model

from sklearn.linear_model import LogisticRegression, LinearRegression

from sklearn.multiclass import OneVsRestClassifier

from sklearn.naive_bayes import MultinomialNB





# MODEL ALGORITHM - ENSAMBLE

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingRegressor





# MODEL ALGORITHM BOOSTING

from catboost import Pool, CatBoost, CatBoostClassifier

from catboost import CatBoostRegressor

import xgboost as xgb

from xgboost import XGBClassifier, XGBRegressor

from xgboost import plot_importance

import lightgbm as lgb





# HYPERPARAMETER TUNING

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso





# METRICS

from sklearn.metrics import roc_auc_score

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.metrics import cohen_kappa_score, mean_squared_error

from sklearn.metrics import precision_recall_fscore_support as score

from sklearn.metrics import f1_score





# DEEP LEARNING

import tensorflow as tf



# NLP

import re

import string

import nltk

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer
# Load Data



train_raw = pd.read_csv('../input/covid19-global-forecasting-week-4/train.csv')



test_raw = pd.read_csv('../input/covid19-global-forecasting-week-4/test.csv')



sub = pd.read_csv('../input/covid19-global-forecasting-week-4/submission.csv')



train_raw.shape, test_raw.shape, sub.shape
train_raw.head()
train = train_raw.copy()

train.head()
test = test_raw.copy()

test.head()


train['Date'] = pd.to_datetime(train['Date'], infer_datetime_format=True)

test['Date'] = pd.to_datetime(test['Date'], infer_datetime_format=True)





train.info()


y1 = train['ConfirmedCases']

y1.head()

y2 = train['Fatalities']

y2.head()
# from https://www.kaggle.com/khoongweihao/covid-19-week-2-xgboost-lightgbm



EMPTY_VAL = "EMPTY_VAL"



def fillState(state, country):

    if state == EMPTY_VAL: return country

    return state



X_Train = train.copy()



X_Train['Province_State'].fillna(EMPTY_VAL, inplace=True)

X_Train['Province_State'] = X_Train.loc[:, ['Province_State', 'Country_Region']].apply(lambda x : fillState(x['Province_State'], x['Country_Region']), axis=1)



X_Train.loc[:, 'Date'] = X_Train.Date.dt.strftime("%m%d")

X_Train["Date"]  = X_Train["Date"].astype(int)



X_Train.head()
X_Test = test.copy()



X_Test['Province_State'].fillna(EMPTY_VAL, inplace=True)

X_Test['Province_State'] = X_Test.loc[:, ['Province_State', 'Country_Region']].apply(lambda x : fillState(x['Province_State'], x['Country_Region']), axis=1)



X_Test.loc[:, 'Date'] = X_Test.Date.dt.strftime("%m%d")

X_Test["Date"]  = X_Test["Date"].astype(int)



X_Test.head()
le = preprocessing.LabelEncoder()




X_Train.Country_Region = le.fit_transform(X_Train.Country_Region)

X_Train['Province_State'] = le.fit_transform(X_Train['Province_State'])



X_Train.head()



X_Test.Country_Region = le.fit_transform(X_Test.Country_Region)

X_Test['Province_State'] = le.fit_transform(X_Test['Province_State'])



X_Test.head()
sub.columns, train.columns, test.columns, X_Train.columns, X_Test.columns
features1 = ['Province_State', 'Country_Region', 'Date']

features2 = ['Province_State', 'Country_Region', 'Date', 'ConfirmedCases']





# features1 = ['Province_State', 'Date']

# features2 = ['Province_State', 'Date', 'ConfirmedCases']
# Training Data and Test Data



X1 = X_Train[features1]



X_test1 = X_Test[features1]
X1.shape, X_test1.shape
# Define Model - CatBoost





cat_model = CatBoostRegressor(

                               loss_function='RMSE',

                               custom_metric='RMSE', 

                               eval_metric='RMSE', 

                               iterations=3000, #100000,

#                                od_type='Iter',

                               verbose=500, #5000,

                               max_depth=3,

#                                one_hot_max_size=10,

#                                cat_features=cat_features,

#                                leaf_estimation_iterations=13,

#                                fold_permutation_block=5,

                               learning_rate=0.01,

                               early_stopping_rounds=1000,

                               task_type='GPU',

                               random_seed=42

                                )
# Country = X_Train.Country_Region v2

Country = X_Train.Province_State # v3



Country.head()
n_splits = 10



gkf = GroupKFold(n_splits=n_splits) 

kf = KFold(n_splits=n_splits)



oof1 = X_Train[['ConfirmedCases']]

oof1['predicts'] = 0

val_rmse1 = []

feature_importance_df1 = pd.DataFrame()
X1.shape, y1.shape, oof1.shape
fold = 0

# for in_index, oof_index in gkf.split(X1, y1, groups=Country):

for in_index, oof_index in kf.split(X1, y1):

    fold += 1

    print(f'fold {fold} of {n_splits}')

    

    X_train1, y_train1 = X1.iloc[in_index], y1.iloc[in_index]

    X_val1, y_val1 = X1.iloc[oof_index], y1.iloc[oof_index]

    

    train_dataset = Pool(data=X_train1, label=y_train1)

    eval_dataset = Pool(data=X_val1, label=y_val1)

    

    evals_result = {}

    

    cat_model.fit(train_dataset,use_best_model=True, verbose=500, eval_set=eval_dataset)

    

    y_pred1,y_test1 = 0,0

    y_pred1 += cat_model.predict(X_val1)

    y_test1 += cat_model.predict(X_test1)

    

    fold_importance_df1 = pd.DataFrame()

    fold_importance_df1["feature"] = features1

    fold_importance_df1["importance"] = cat_model.feature_importances_

    fold_importance_df1["fold"] = fold + 1

    feature_importance_df1 = pd.concat([feature_importance_df1, fold_importance_df1], axis=0)

    

    oof1['predicts'][oof_index] = y_pred1/n_splits

    sub['ConfirmedCases'] = y_test1/n_splits

    

    val_score1 = np.sqrt(mean_squared_error(y_val1, y_pred1))

    val_rmse1.append(val_score1)

    
sub.head()
X2 = X_Train[features2]



X_test2 = X_Test[features1]
X2.shape, X_test2.shape
X_test2['ConfirmedCases'] = sub['ConfirmedCases']



X2.shape, X_test2.shape
X_test2.head()
oof2 = X_Train[['Fatalities']]

oof2['predicts'] = 0

val_rmse2 = []

feature_importance_df2 = pd.DataFrame()
X2.shape, y2.shape, oof2.shape
fold = 0

# for in_index, oof_index in gkf.split(X2, y2, groups=Country):

for in_index, oof_index in kf.split(X2, y2):

    fold += 1

    print(f'fold {fold} of {n_splits}')

    

    X_train2, y_train2 = X2.iloc[in_index], y2.iloc[in_index]

    X_val2, y_val2 = X2.iloc[oof_index], y2.iloc[oof_index]

    

    train_dataset = Pool(data=X_train2, label=y_train2)

    eval_dataset = Pool(data=X_val2, label=y_val2)

    

    evals_result = {}

    

    cat_model.fit(train_dataset,use_best_model=True, verbose=500, eval_set=eval_dataset)

    

    y_pred2,y_test2 = 0,0

    y_pred2 += cat_model.predict(X_val2)

    y_test2 += cat_model.predict(X_test2)

    

    fold_importance_df2 = pd.DataFrame()

    fold_importance_df2["feature"] = features2

    fold_importance_df2["importance"] = cat_model.feature_importances_

    fold_importance_df2["fold"] = fold + 1

    feature_importance_df2 = pd.concat([feature_importance_df2, fold_importance_df2], axis=0)

    

    oof2['predicts'][oof_index] = y_pred2/n_splits

    sub['Fatalities'] = y_test2/n_splits

    

    val_score2 = np.sqrt(mean_squared_error(y_val2, y_pred2))

    val_rmse2.append(val_score2)
# sub.tail(500)
# oof2.head(100)
submission =  sub.copy()



submission['ConfirmedCases'] = submission['ConfirmedCases'].astype('int')

submission['Fatalities'] = submission['Fatalities'].astype('int')



submission.head()
submission.to_csv('submission.csv', index=False)

submission.head()