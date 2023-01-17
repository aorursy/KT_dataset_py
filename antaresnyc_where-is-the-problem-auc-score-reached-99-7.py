import pandas as pd, numpy as np
import os
import math
from math import ceil, floor, log
import random

from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, precision_recall_curve, auc, roc_curve, recall_score, classification_report 
from sklearn.model_selection import train_test_split
import sklearn
from sklearn import metrics
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import seaborn as sns

from yellowbrick.classifier import ClassificationReport
import scikitplot as skplt

from xgboost import XGBClassifier
import xgboost as xgb
from lightgbm import LGBMClassifier
import catboost
print(catboost.__version__)
from catboost import *
from catboost import datasets
from catboost import CatBoostClassifier

import scikitplot as skplt
SEED = 1970
random.seed(SEED)

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
path = '../input/health-insurance-cross-sell-prediction/'
df_train = pd.read_csv(path + "train.csv")
df_test = pd.read_csv(path + "test.csv")
print(df_train.isnull().sum())
print(df_test.isnull().sum())
col_list = df_train.columns.to_list()[1:]
df_train_corr = df_train.copy().set_index('id')
df_train_ones = df_train_corr.loc[df_train_corr.Response == 1].copy()

categorical_features = ['Gender', 'Driving_License', 'Region_Code', 'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage','Policy_Sales_Channel']
text_features = ['Gender', 'Vehicle_Age', 'Vehicle_Damage']

# code text categorical features
le = preprocessing.LabelEncoder()
for f in text_features :
    df_train_corr[f] = le.fit_transform(df_train_corr[f])
# change digital categorical datatype so CatBoost can deal with them
df_train_corr.Region_Code = df_train_corr.Region_Code.astype('int32')
df_train_corr.Policy_Sales_Channel = df_train_corr.Policy_Sales_Channel.astype('int32')
corr = df_train_corr.loc[:,:'Vintage'].corr()

mask = np.triu(np.ones_like(corr, dtype=bool))
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.figure(figsize = (8, 8))
sns.scatterplot(df_train_corr['Policy_Sales_Channel'],df_train_corr['age_bin_cat'])
plt.title('Binned Age vs Policy_Sales_Channel', fontsize = 15)
plt.show()
def plot_ROC(fpr, tpr, m_name):
    roc_auc = sklearn.metrics.auc(fpr, tpr)
    plt.figure(figsize=(6, 6))
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc, alpha=0.5)
    
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--', alpha=0.5)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True)
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('Receiver operating characteristic for %s'%m_name, fontsize=20)
    plt.legend(loc="lower right", fontsize=16)
    plt.show()
def upsample(df, u_feature, n_upsampling):
    ones = df.copy()
    for n in range(n_upsampling):
        if u_feature == 'Annual_Premium':
            df[u_feature] = ones[u_feature].apply(lambda x: x + random.randint(-1,1)* x *0.05) # change Annual_premiun in the range of 5%
        else:
            df[u_feature] = ones[u_feature].apply(lambda x: x + random.randint(-5,5)) # change Age in the range of 5 years
                
        if n == 0:
            df_new = df.copy()
        else:
            df_new = pd.concat([df_new, df])
    return df_new

try:
    df_train_corr.drop(columns = ['bin_age'], inplace = True)
except:
    print('already deleted')        

df_train_mod = df_train_corr.copy()
df_train_mod['old_damaged'] = df_train_mod.apply(lambda x: pow(2,x.Vehicle_Age)+pow(2,x.Vehicle_Damage), axis =1)

# we shall preserve validation set without augmentation/over-sampling
df_temp, X_valid, _, y_valid = train_test_split(df_train_mod, df_train_mod['Response'], train_size=0.8, random_state = SEED)
X_valid = X_valid.drop(columns = ['Response'])

# upsampling Positive Response class only
df_train_up_a = upsample(df_temp.loc[df_temp['Response'] == 1], 'Age', 1)
df_train_up_v = upsample(df_temp.loc[df_temp['Response'] == 1], 'Vintage', 1)
df_train_mod.head()
df_ext = pd.concat([df_train_mod,df_train_up_a])
df_ext = pd.concat([df_ext,df_train_up_v])
X_train = df_ext.drop(columns = ['Response'])
y_train = df_ext.Response
print('Train set target class count with over-sampling:')
print(y_train.value_counts())
print('Validation set target class count: ')
print(y_valid.value_counts())
X_train.head()
XGB_model_u = XGBClassifier(random_state = SEED, max_depth = 8, 
                            n_estimators = 3000, reg_lambda = 1.2, reg_alpha = 1.2, 
                            min_child_weight = 1, 
                            objective = 'binary:logistic',
                            learning_rate = 0.15, gamma = 0.3, colsample_bytree = 0.5, eval_metric = 'auc')

XGB_model_u.fit(X_train, y_train)
XGB_preds_u = XGB_model_u.predict_proba(X_valid)
XGB_score_u = roc_auc_score(y_valid, XGB_preds_u[:,1])
XGB_class_u = XGB_model_u.predict(X_valid)

(fpr, tpr, thresholds) = roc_curve(y_valid, XGB_preds_u[:,1])
plot_ROC(fpr, tpr,'XGBoost')
print('ROC AUC score for XGBoost model with over-sampling + 2 new features: %.4f'%XGB_score_u)
print('F1 score: %0.4f'%f1_score(y_valid, XGB_class_u))
skplt.metrics.plot_confusion_matrix(y_valid, XGB_class_u,
        figsize=(8,8))
xgb.plot_importance(XGB_model_u)
# X_train.drop(columns = ['Previously_Insured', 'Driving_License','Vehicle_Age','Vehicle_Damage'], inplace = True)
# X_valid.drop(columns = ['Previously_Insured', 'Driving_License','Vehicle_Age','Vehicle_Damage'], inplace = True)
# X_train.drop(columns = ['Previously_Insured'], inplace = True)
# X_valid.drop(columns = ['Previously_Insured'], inplace = True)
# XGB_model_ud = XGBClassifier(random_state = SEED, max_depth = 8, n_estimators = 3000, reg_lambda = 1.2, reg_alpha = 1.2, 
#                           min_child_weight = 1, 
#                           objective = 'binary:logistic',
#                           learning_rate = 0.15, gamma = 0.3, colsample_bytree = 0.5, eval_metric = 'auc')

# XGB_model_ud.fit(X_train, y_train)
# XGB_preds_ud = XGB_model_ud.predict_proba(X_valid)
# XGB_score_ud = roc_auc_score(y_valid, XGB_preds_ud[:,1])
# XGB_class_ud = XGB_model_ud.predict(X_valid)

# (fpr, tpr, thresholds) = roc_curve(y_valid, XGB_preds_ud[:,1])
# plot_ROC(fpr, tpr,'XGBoost')
# print('ROC AUC score for XGBoost model with over-sampling, and 4 features removed: %.4f'%XGB_score_ud)
# print('F1 score: %0.4f'%f1_score(y_valid, XGB_class_ud))
# skplt.metrics.plot_confusion_matrix(y_valid, XGB_class_ud,
#         figsize=(8,8))
# xgb.plot_importance(XGB_model_ud)
rf_params = {'max_depth': 20, 'n_estimators': 3000, 'min_samples_leaf': 1}
# rf_params = {'max_depth': 20, 'n_estimators': 300, 'min_samples_leaf': 1}
rf_params['random_state'] = SEED
rf = RandomForestClassifier(**rf_params)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_valid)
rf_preds_prob = rf.predict_proba(X_valid)[:,1]

reg_score_uc = roc_auc_score(y_valid, rf_preds_prob, average = 'weighted')
print('ROC AUC score for RandomForest model with over-sampling: %.4f'%reg_score_uc)
print('Optimized RF f1-score', f1_score(y_valid, rf_preds))
skplt.metrics.plot_confusion_matrix(y_valid, rf_preds,figsize=(8,8))

(fpr, tpr, thresholds) = roc_curve(y_valid, rf_preds_prob)
plot_ROC(fpr, tpr,'RandomForest')
title="Feature Importances Random Forest"
feat_imp = pd.DataFrame({'importance':rf.feature_importances_}) 
feat_imp['feature'] = X_train.columns
feat_imp.sort_values(by='importance', ascending=True, inplace=True)
feat_imp = feat_imp.set_index('feature', drop=True)
feat_imp.plot.barh(title=title, figsize=(8,8))
plt.xlabel('Feature Importance Score')
plt.show()
# rf.get_params()
categorical_features1 = ['Gender',
 'age_bin_cat',
 'Region_Code',
 'old_damaged',
 'Policy_Sales_Channel']

X_train.old_damaged = X_train.old_damaged.astype('int32')
X_valid.old_damaged = X_valid.old_damaged.astype('int32')
Cat_model1 = CatBoostClassifier( iterations = 30000, 
                                random_seed = SEED, 
#                                 task_type = 'GPU',
                                task_type = 'CPU',
                                learning_rate=0.15,
                                random_strength=0.1,
                                depth=8,
                                loss_function='Logloss',
                                eval_metric='Logloss',
                                leaf_estimation_method='Newton',
                                subsample = 0.9,
                                rsm = 0.8,
                                custom_loss = ['AUC'] )
Cat_model1.fit(X_train, y_train, cat_features = categorical_features1, eval_set = (X_valid, y_valid), plot = False,
              early_stopping_rounds=50,verbose = 1000)
Cat_preds1 = Cat_model1.predict_proba(X_valid)
Cat_class1 = Cat_model1.predict(X_valid)
Cat_score1 = roc_auc_score(y_valid, Cat_preds1[:,1])

(fpr, tpr, thresholds) = roc_curve(y_valid, Cat_preds1[:,1])
plot_ROC(fpr, tpr, 'CatBoost')
print('ROC AUC score for CatBoost model with over-sampling: %.4f'%Cat_score1)
print('CatBoost f1-score', f1_score(y_valid, Cat_class1))
skplt.metrics.plot_confusion_matrix(y_valid, Cat_class1,figsize=(8,8))
X_train.head()
XGB_model_l = XGBClassifier(random_state = SEED, max_depth = 8, 
                            n_estimators = 30000, 
                            reg_lambda = 1.2, reg_alpha = 1.2, 
                            min_child_weight = 1, 
                            objective = 'binary:logistic',
                            learning_rate = 0.15, gamma = 0.3, colsample_bytree = 0.5, eval_metric = 'auc')

XGB_model_l.fit(X_train, y_train,
                eval_set = [(X_valid, y_valid)],
                early_stopping_rounds=50,verbose = 1000)
XGB_preds_l = XGB_model_l.predict_proba(X_valid)
XGB_score_l = roc_auc_score(y_valid, XGB_preds_l[:,1])
XGB_class_l = XGB_model_l.predict(X_valid)
(fpr, tpr, thresholds) = roc_curve(y_valid, XGB_preds_l[:,1])
plot_ROC(fpr, tpr,'XGBoost')

print('ROC AUC score for XGBoost model with over-sampling + 2 new features: %.4f'%XGB_score_l)
print('F1 score: %0.4f'%f1_score(y_valid, XGB_class_l))
skplt.metrics.plot_confusion_matrix(y_valid, XGB_class_l,
        figsize=(8,8))

xgb.plot_importance(XGB_model_l)