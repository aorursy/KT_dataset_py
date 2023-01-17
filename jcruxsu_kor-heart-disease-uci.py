import numpy as np

import os

import pandas as pd
import missingno as msno

import seaborn as sns

import matplotlib.pyplot as plt
def regularization_mean_std_d(data_arr):

    mean = np.mean(data_arr)

    std = np.std(data_arr) 

    regularaz_arr = (data_arr - mean)/std

    return regularaz_arr
high_val_list = regularization_mean_std_d(np.arange(1000,10000,1000))

nomal_val_list = regularization_mean_std_d(np.arange(1,10))

f, ax = plt.subplots(1, 2, figsize=(18, 8))

sns.kdeplot(high_val_list, ax=ax[0])

sns.kdeplot(nomal_val_list, ax=ax[1])
def max_min_reg(data_arr):

    max_ = np.max(data_arr)

    min_ = np.min(data_arr) 

    regularaz_arr = (data_arr - min_)/(max_-min_)

    return regularaz_arr
def regularization_l2(data_arr):

    return (data_arr/ np.linalg.norm(data_arr))
DATA_PATH = '/kaggle/input/heart-disease-uci/heart.csv'
entire_df = pd.read_csv(DATA_PATH)
from sklearn.model_selection import train_test_split

df_train,df_test = train_test_split(entire_df,test_size=0.25, random_state=2020)
df_train.head()
df_train.describe()
df_train.isnull().sum()
f, ax = plt.subplots(1, 2, figsize=(18, 8))







df_train['target'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.f%%', ax=ax[0], shadow=True)

ax[0].set_title('Pie plot - target')

ax[0].set_ylabel('')

sns.countplot('target', data=df_train, ax=ax[1])

ax[1].set_title('Count plot - target')



plt.show()
df_train['age'].describe()
fig,ax = plt.subplots(1,1,figsize=(9,5))

sns.kdeplot(df_train[df_train['target']==0]['age'],ax = ax)

sns.kdeplot(df_train[df_train['target']==1]['age'],ax=ax)

plt.legend(['target==0','target==1'])

plt.title('Ogriginal age distributution')

plt.show()
fig,ax = plt.subplots(2,2,figsize=(9,5))



sns.kdeplot(np.log(df_train[df_train['target']==0]['age']),ax = ax[0][0])

sns.kdeplot(np.log(df_train[df_train['target']==1]['age']),ax=ax[0][0])

ax[0][0].legend(['target==0','target==1'])

ax[0][0].set_title('Logged chol plot')



sns.kdeplot(np.cos(df_train[df_train['target']==0]['age']),ax = ax[0][1])

sns.kdeplot(np.cos(df_train[df_train['target']==1]['age']),ax=ax[0][1])

ax[0][1].legend(['target==0','target==1'])

ax[0][1].set_title('Sin chol plot')



sns.kdeplot(regularization_mean_std_d(df_train[df_train['target']==0]['age']),ax = ax[1][0])

sns.kdeplot(regularization_mean_std_d(df_train[df_train['target']==1]['age']),ax=ax[1][0])

ax[1][0].legend(['target==0','target==1'])

ax[1][0].set_title('sub_mean_std_reg chol plot')



sns.kdeplot(regularization_l2(df_train[df_train['target']==0]['age']),ax = ax[1][1])

sns.kdeplot(regularization_l2(df_train[df_train['target']==1]['age']),ax=ax[1][1])

ax[1][1].legend(['target==0','target==1'])

ax[1][1].set_title('l2 reg chol plot')

plt.show()
sns.countplot('sex', hue='target', data=df_train)
df_train['cp'].describe()
sns.countplot('cp', hue='target', data=df_train)
df_train['trestbps'].describe()
fig,ax = plt.subplots(1,1,figsize=(9,5))

sns.kdeplot(df_train[df_train['target']==0]['trestbps'],ax = ax)

sns.kdeplot(df_train[df_train['target']==1]['trestbps'],ax=ax)

plt.legend(['target==0','target==1'])

plt.title('Ogriginal chol distributution')

plt.show()
fig,ax = plt.subplots(2,3,figsize=(9,5))



sns.kdeplot(np.log(df_train[df_train['target']==0]['trestbps']),ax = ax[0][0])

sns.kdeplot(np.log(df_train[df_train['target']==1]['trestbps']),ax=ax[0][0])

ax[0][0].legend(['target==0','target==1'])

ax[0][0].set_title('Logged trestbps plot')



sns.kdeplot(np.sin(df_train[df_train['target']==0]['trestbps']),ax = ax[0][1])

sns.kdeplot(np.sin(df_train[df_train['target']==1]['trestbps']),ax=ax[0][1])

ax[0][1].legend(['target==0','target==1'])

ax[0][1].set_title('Sin trestbps plot')



sns.kdeplot(regularization_mean_std_d(df_train[df_train['target']==0]['trestbps']),ax = ax[1][0])

sns.kdeplot(regularization_mean_std_d(df_train[df_train['target']==1]['trestbps']),ax=ax[1][0])

ax[1][0].legend(['target==0','target==1'])

ax[1][0].set_title('sub_mean_std_reg trestbps plot')



sns.kdeplot(regularization_l2(df_train[df_train['target']==0]['trestbps']),ax = ax[1][1])

sns.kdeplot(regularization_l2(df_train[df_train['target']==1]['trestbps']),ax=ax[1][1])

ax[1][1].legend(['target==0','target==1'])

ax[1][1].set_title('l2 reg trestbps plot')



sns.kdeplot(max_min_reg(df_train[df_train['target']==0]['trestbps']),ax = ax[1][2])

sns.kdeplot(max_min_reg(df_train[df_train['target']==1]['trestbps']),ax=ax[1][2])

ax[1][2].legend(['target==0','target==1'])

ax[1][2].set_title('min_max_reg trestbps plot')

plt.show()
df_train['chol'].describe()
fig,ax = plt.subplots(1,1,figsize=(9,5))

sns.kdeplot(df_train[df_train['target']==0]['chol'],ax = ax)

sns.kdeplot(df_train[df_train['target']==1]['chol'],ax=ax)

plt.legend(['target==0','target==1'])

plt.title('Ogriginal chol distributution')

plt.show()
fig,ax = plt.subplots(2,3,figsize=(9,5))



sns.kdeplot(np.log(df_train[df_train['target']==0]['chol']),ax = ax[0][0])

sns.kdeplot(np.log(df_train[df_train['target']==1]['chol']),ax=ax[0][0])

ax[0][0].legend(['target==0','target==1'])

ax[0][0].set_title('Logged chol plot')



sns.kdeplot(np.sin(df_train[df_train['target']==0]['chol']),ax = ax[0][1])

sns.kdeplot(np.sin(df_train[df_train['target']==1]['chol']),ax=ax[0][1])

ax[0][1].legend(['target==0','target==1'])

ax[0][1].set_title('Sin chol plot')



sns.kdeplot(regularization_mean_std_d(df_train[df_train['target']==0]['chol']),ax = ax[1][0])

sns.kdeplot(regularization_mean_std_d(df_train[df_train['target']==1]['chol']),ax=ax[1][0])

ax[1][0].legend(['target==0','target==1'])

ax[1][0].set_title('sub_mean_std_reg chol plot')



sns.kdeplot(regularization_l2(df_train[df_train['target']==0]['chol']),ax = ax[1][1])

sns.kdeplot(regularization_l2(df_train[df_train['target']==1]['chol']),ax=ax[1][1])

ax[1][1].legend(['target==0','target==1'])

ax[1][1].set_title('l2 reg chol plot')



sns.kdeplot(max_min_reg(df_train[df_train['target']==0]['chol']),ax = ax[1][2])

sns.kdeplot(max_min_reg(df_train[df_train['target']==1]['chol']),ax=ax[1][2])

ax[1][2].legend(['target==0','target==1'])

ax[1][2].set_title('min_max_reg chol plot')

plt.show()
df_train['fbs'].describe()
sns.countplot('fbs', hue='target', data=df_train)
sns.countplot('restecg', hue='target', data=df_train)
df_train['thalach'].describe()
fig,ax = plt.subplots(1,1,figsize=(9,5))

sns.kdeplot(df_train[df_train['target']==0]['thalach'],ax = ax)

sns.kdeplot(df_train[df_train['target']==1]['thalach'],ax=ax)

plt.legend(['target==0','target==1'])

plt.title('Ogriginal thalach distributution')

plt.show()
fig,ax = plt.subplots(2,3,figsize=(9,5))



sns.kdeplot(np.log(df_train[df_train['target']==0]['thalach']),ax = ax[0][0])

sns.kdeplot(np.log(df_train[df_train['target']==1]['thalach']),ax=ax[0][0])

ax[0][0].legend(['target==0','target==1'])

ax[0][0].set_title('Logged thalach plot')



sns.kdeplot(np.cos(df_train[df_train['target']==0]['thalach']),ax = ax[0][1])

sns.kdeplot(np.cos(df_train[df_train['target']==1]['thalach']),ax=ax[0][1])

ax[0][1].legend(['target==0','target==1'])

ax[0][1].set_title('Sin thalach plot')



sns.kdeplot(regularization_mean_std_d(df_train[df_train['target']==0]['thalach']),ax = ax[1][0])

sns.kdeplot(regularization_mean_std_d(df_train[df_train['target']==1]['thalach']),ax=ax[1][0])

ax[1][0].legend(['target==0','target==1'])

ax[1][0].set_title('sub_mean_std_reg thalach plot')



sns.kdeplot(regularization_l2(df_train[df_train['target']==0]['thalach']),ax = ax[1][1])

sns.kdeplot(regularization_l2(df_train[df_train['target']==1]['thalach']),ax=ax[1][1])

ax[1][1].legend(['target==0','target==1'])

ax[1][1].set_title('l2 reg thalach plot')





sns.kdeplot(max_min_reg(df_train[df_train['target']==0]['thalach']),ax = ax[1][2])

sns.kdeplot(max_min_reg(df_train[df_train['target']==1]['thalach']),ax=ax[1][2])

ax[1][2].legend(['target==0','target==1'])

ax[1][2].set_title('min_max_reg thalach plot')

plt.show()
df_train['exang'].describe()
sns.countplot('exang', hue='target', data=df_train)
df_train['oldpeak'].describe()
fig,ax = plt.subplots(1,1,figsize=(9,5))

sns.kdeplot(df_train[df_train['target']==0]['oldpeak'],ax = ax)

sns.kdeplot(df_train[df_train['target']==1]['oldpeak'],ax=ax)

plt.legend(['target==0','target==1'])

plt.title('Ogriginal oldpeak distributution')

plt.show()
df_train['slope'].describe()
sns.countplot('slope', hue='target', data=df_train)
sns.countplot('ca', hue='target', data=df_train)
df_train['thal'].describe()
sns.countplot('thal', hue='target', data=df_train)
df_train['age'] = regularization_mean_std_d(df_train['age'])

df_train['trestbps'] = max_min_reg(df_train['trestbps'])

df_train['chol'] = max_min_reg(df_train['chol'])

df_train['thalach'] = regularization_l2(df_train['thalach'])
df_test['age'] = regularization_mean_std_d(df_test['age'])

df_test['trestbps'] = max_min_reg(df_test['trestbps'])

df_test['chol'] = max_min_reg(df_test['chol'])

df_test['thalach'] = regularization_l2(df_test['thalach'])
#df_train = max_min_reg(df_train)

#df_test = max_min_reg(df_test)
df_train.head()
df_test.head()
y_train = np.asarray(df_train['target'])

x_train = np.asarray(df_train.drop(['target'],axis=1))
y_test = np.asarray(df_test['target'])

x_test = np.asarray(df_test.drop(['target'],axis=1))
df_train.head()
df_test.head()
models_result_label =['xgb','random forest']

models_result = []
from xgboost.sklearn import XGBClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import StratifiedKFold
xgb_c = XGBClassifier(    

    learning_rate =0.005,

    n_estimators=100,

    max_depth=10,

    min_child_weight=8,

    gamma=0.0008,

    reg_alpha=1e-05,

    subsample=0.99,

    colsample_bytree=0.8,

    objective= 'binary:logistic',

    nthread=-1,

    scale_pos_weight=1,

    seed=2020)
skfold = StratifiedKFold(n_splits=4)

sf_xgb_train_res_ensem = np.zeros(len(x_train))

sf_xgb_test_res_ensem =np.zeros(len(x_test))

for tr_index,test_index in skfold.split(x_train,y_train):

    xgb_c.fit(x_train[tr_index],y_train[tr_index])

    y_pred =  xgb_c.predict(x_train)

    sf_xgb_train_res_ensem+=y_pred

    sf_xgb_test_res_ensem+=xgb_c.predict(x_test)

sf_xgb_train_res_ensem =  np.round(sf_xgb_train_res_ensem/4)

sf_xgb_test_res_ensem = np.round(sf_xgb_test_res_ensem/4)
from sklearn.metrics import accuracy_score, f1_score, precision_score

print("acc_train score : ", accuracy_score(sf_xgb_train_res_ensem,y_train),",f1_score :",f1_score(sf_xgb_train_res_ensem,y_train))

print("acc_train score : ", accuracy_score(sf_xgb_test_res_ensem,y_test),",f1_score :",f1_score(sf_xgb_test_res_ensem,y_test))

models_result.append(f1_score(sf_xgb_test_res_ensem,y_test))
xgb_c.score(x_test,y_test)
sf_rf_train_res_ensem = np.zeros(len(x_train))

sf_rf_test_res_ensem = np.zeros(len(x_test))

skfold = StratifiedKFold(n_splits=5)

rf_c = RandomForestClassifier(    

            n_estimators= 100,

            max_depth=5,

            ccp_alpha=1e-07,

         random_state=2020)

for tr_index,test_index in skfold.split(x_train,y_train):

    rf_c.fit(x_train[tr_index],y_train[tr_index])

    y_pred =  rf_c.predict(x_train)

    sf_rf_train_res_ensem+=y_pred

    sf_rf_test_res_ensem+=rf_c.predict(x_test)

sf_rf_train_res_ensem =np.round(sf_rf_train_res_ensem/5)

sf_rf_test_res_ensem =np.round(sf_rf_test_res_ensem/5)

#rf_c.fit(x_train,y_train)

rf_c.score(x_test,y_test)
print("acc_train score : ", accuracy_score(sf_rf_train_res_ensem,y_train),",f1_score :",f1_score(sf_rf_train_res_ensem,y_train))

print("acc_train score : ", accuracy_score(sf_rf_test_res_ensem,y_test),",f1_score :",f1_score(sf_rf_test_res_ensem,y_test))

models_result.append(f1_score(sf_rf_test_res_ensem,y_test))
from pandas import Series
plt.figure(figsize=(8, 8))

sns.barplot(models_result_label,models_result)

plt.ylabel('performance')

plt.ylabel('pre-train model list')

plt.show()
feature_importance = xgb_c.feature_importances_

Series_feat_imp = Series(feature_importance, index=df_train.drop(['target'],axis=1).columns)

plt.figure(figsize=(8, 8))

Series_feat_imp.sort_values(ascending=True).plot.barh()

plt.xlabel('Feature importance')

plt.ylabel('xgboost Feature')

plt.show()
feature_importance = rf_c.feature_importances_

Series_feat_imp = Series(feature_importance, index=df_train.drop(['target'],axis=1).columns)

plt.figure(figsize=(8, 8))

Series_feat_imp.sort_values(ascending=True).plot.barh()

plt.xlabel('Feature importance')

plt.ylabel('Random forest Feature')

plt.show()