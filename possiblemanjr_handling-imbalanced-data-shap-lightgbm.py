import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline



from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve, roc_curve

from sklearn.preprocessing import StandardScaler , Binarizer

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

import lightgbm as lgb

from lightgbm import LGBMClassifier

import time

import os, sys, gc, warnings, random, datetime

import math

import shap

import joblib

warnings.filterwarnings('ignore')

import gc

# !pip install --upgrade git+https://github.com/stanfordmlgroup/ngboost.git

    

# from ngboost import NGBRegressor, NGBClassifier

# from ngboost.ngboost import NGBoost

# from ngboost.learners import default_tree_learner

# from ngboost.scores import CRPS, MLE , LogScore

# from ngboost.distns import LogNormal, Normal

# from ngboost.distns import k_categorical, Bernoulli

df = pd.read_pickle("../input/handling-imbalanced-data-eda-small-fe/df_for_use.pkl")

df_fe = pd.read_pickle("../input/handling-imbalanced-data-eda-small-fe/df_fe.pkl")
lgbm_clf = joblib.load('../input/handling-imbalanced-data-supervised-learning/lgbm_clf.pkl')

rf_clf = joblib.load('../input/handling-imbalanced-data-supervised-learning/rf_clf.pkl')

xgb_clf = joblib.load('../input/handling-imbalanced-data-supervised-learning/xgb_clf.pkl')

# ngb_clf = joblib.load('../input/handling-imbalanced-data-supervised-learning/ngb_clf.pkl')

X = df.drop('loan_condition_cat', axis=1)

y = df['loan_condition_cat']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2 , random_state = 2020, stratify = y)

y_trainin = y_train.to_frame()

for_sample_train_df = pd.concat([X_train, y_trainin], axis=1)

y_testet = y_test.to_frame()

for_sample_test_df = pd.concat([X_test, y_testet], axis=1)





X_ = for_sample_train_df.drop('loan_condition_cat', axis=1)

y_ = for_sample_train_df['loan_condition_cat']



sample_train_x, sample_test_x, sample_train_y, sample_test_y = train_test_split(X_, y_, test_size = 0.8 , random_state = 2020, stratify = y_)



del X_train, X_test, y_train, y_test , y_trainin, y_testet, 
gc.collect()
## Make sample for faster computation



X_ = for_sample_train_df.drop('loan_condition_cat', axis=1)

y_ = for_sample_train_df['loan_condition_cat']



sample_train_x, sample_test_x, sample_train_y, sample_test_y = train_test_split(X_, y_, test_size = 0.95 , random_state = 2020, stratify = y_)
## Make sample for faster computation



X = for_sample_train_df.drop('loan_condition_cat', axis=1)

y = for_sample_train_df['loan_condition_cat']



sample_train_x_smaller, sample_test_x_smaller, sample_train_y_smaller, sample_test_y_smaller = train_test_split(X, y, test_size = 0.98 , random_state = 2020, stratify = y)
X_sampled = sample_train_x.copy()
X_sampled_smaller = sample_train_x_smaller.copy()
#LightGBM

import shap

shap.initjs()



# (same syntax works for LightGBM, CatBoost, and scikit-learn models)



explainer = shap.TreeExplainer(lgbm_clf)

shap_values = explainer.shap_values(X_sampled)
shap.force_plot(explainer.expected_value[1], shap_values[1][2,:], X_sampled.iloc[2,:])
shap.force_plot(explainer.expected_value[1], shap_values[1][3,:], X_sampled.iloc[3,:])
shap.force_plot(explainer.expected_value[1], shap_values[1][4,:], X_sampled.iloc[4,:])
shap.force_plot(explainer.expected_value[1], shap_values[1][5,:], X_sampled.iloc[5,:])
shap.force_plot(explainer.expected_value[1], shap_values[1][1650,:], X_sampled.iloc[1650,:])
shap.decision_plot(explainer.expected_value[1], shap_values[1][0,:], X_sampled.iloc[0,:])
# summarize the effects of all the features

shap.summary_plot(shap_values, X_sampled, plot_type="bar")
shap.dependence_plot("total_rec_prncp",shap_values[1], X_sampled)
shap.dependence_plot("interest_rate",shap_values[1], X_sampled)
shap.dependence_plot("grade_cat",shap_values[1], X_sampled)
# (same syntax works for LightGBM, CatBoost, and scikit-learn models)

shap.initjs()

explainer = shap.TreeExplainer(lgbm_clf)

shap_values = explainer.shap_values(X_sampled_smaller)





shap.force_plot(base_value=explainer.expected_value[1], shap_values=shap_values[1], features=X_sampled_smaller.columns)