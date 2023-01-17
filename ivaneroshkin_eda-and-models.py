# !pip install -U vega_datasets notebook vega
import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
%matplotlib inline
from tqdm import tqdm_notebook
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVR, SVR
from sklearn.metrics import mean_absolute_error, roc_auc_score
# pd.options.display.precision = 15

import lightgbm as lgb
import xgboost as xgb
import time
import datetime
from catboost import CatBoostRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, GroupKFold, GridSearchCV, train_test_split, TimeSeriesSplit
from sklearn import metrics
from sklearn import linear_model
import gc
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# import eli5
# import shap
from IPython.display import HTML
import json
import altair as alt

import networkx as nx
import matplotlib.pyplot as plt
%matplotlib inline

# alt.renderers.enable('notebook')

# %env JOBLIB_TEMP_FOLDER=/tmp

import matplotlib.pyplot as plt
import numpy as np

# from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
def get_unique(df):
    uv = []
    cols = []
    for col in list(df.columns):
        uv.append(df[col].unique().shape[0])
        cols.append(col)
    df_uv = pd.DataFrame({'predictor': cols, 'uv': uv}  )
    return df_uv



## Function to reduce the DF size
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
# import pathlib
# pathlib.Path().absolute()
print(os.listdir("../input/"))
# загрузка датасетов
train_identity = pd.read_csv(r'../input/ieee-fraud-detection/train_identity.csv')
train_transaction = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv')
test_identity = pd.read_csv('../input/ieee-fraud-detection/test_identity.csv')
test_transaction = pd.read_csv(f'../input/ieee-fraud-detection/test_transaction.csv')
sub = pd.read_csv('../input/ieee-fraud-detection/sample_submission.csv') 
print('train_identity.shape: ', train_identity.shape)
print('train_transaction.shape: ', train_transaction.shape)
print('test_identity.shape: ', test_identity.shape)
print('test_transaction.shape: ', test_transaction.shape)
train_transaction.head(2)
train_identity.head(2)
train = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')
test = pd.merge(test_transaction, test_identity, on='TransactionID', how='left')
train = reduce_mem_usage(train)
test = reduce_mem_usage(test)
# Поскольку загруженные данные занимают много оперативной памяти, то удалим из памяти первично загруженные датасеты:
del train_identity, train_transaction, test_identity, test_transaction
# дополнительнопринудительно вызывем сборщик мусора, который почистит память (примерно 3 гб):
gc.collect()
y_dsb = train.groupby('isFraud')['isFraud'].count()
print('Доля хитов целевой переменной: {:.1%}'.format(y_dsb[1]/(y_dsb[1]+y_dsb[0])))

sns.distplot(train.isFraud,kde=False);
train_cols = pd.DataFrame({'type': train.dtypes}).reset_index()
train_cols['uv'] = get_unique(train).uv
train_cols.head()
null_sum = train.isnull().sum()
null_cnt = train.isnull().count()
train_cols.index = train_cols['index']
train_cols.drop(['index'], axis=1,inplace=True)
train_cols['num_vals'] = null_cnt
train_cols['num_nans'] = null_sum
train_cols['num_nans_perc'] = null_sum / null_cnt
train_cols.head()
train_cols.sort_values(by='num_nans_perc', ascending=False)
train_cols['cat'] = np.nan
train_cols.loc[train_cols.num_nans_perc>0.5, 'cat'] = 'del'
train_cols[(train_cols.cat!='del')].groupby('type')['type'].count()
train_cols.loc[(train_cols.cat!='del')&(train_cols.type!='object'), 'cat'] = 'num'
train_cols.loc[(train_cols.cat!='del')&(train_cols.type=='object'), 'cat'] = 'cat'

train_cols.groupby('cat')['cat'].count()
train_cols.sort_values(by='uv', ascending=True).head(20)
train_cols[train_cols.cat=='cat']
for col in list(train_cols[train_cols.cat=='cat'].index):
    print(col)
    plt.subplots(1,2,figsize=(15,5))
    plt.subplot(1,2,1)
    sns.countplot(train[col]);
    plt.subplot(1,2,2)
    sns.barplot(x=col, y='isFraud', data=train);
    plt.show()
# убираем из предикторов колонку 'M1'
train_cols.loc[train_cols.index=='M1', 'cat']='del'
train_cols.loc[train_cols.cat=='num'].sort_values('uv')
train_cols.loc[train_cols.cat=='num']
# i=0
# for col in train_cols.loc[train_cols.cat=='num'].index:
#     print(i)
#     print(col)
#     train.hist(col)
#     i = i + 1
#     plt.show()
# преобразуем TransactionDT в стандартную дату и создадим новую фичу Date
import datetime

genesis = datetime.datetime.strptime('2019-01-01', '%Y-%m-%d')
train['Date'] = train['TransactionDT'].apply(lambda x : genesis+datetime.timedelta(seconds=x))
# создадим новые фичи:
train['Weekdays'] = train['Date'].dt.dayofweek
train['Days'] = train['Date'].dt.day
train['Hours'] = train['Date'].dt.hour
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

g = sns.barplot(train.Weekdays, train.isFraud, ax=ax[0])
ax[0].set_title('Fraud Charges by Weekdays')
plt.setp(g.get_xticklabels(), visible=False)

g = sns.barplot(train.Days, train.isFraud, ax=ax[1])
ax[1].set_title('Fraud Charges by Days')
plt.setp(g.get_xticklabels(), visible=False)

g = sns.barplot(train.Hours, train.isFraud, ax=ax[2])
ax[2].set_title('Fraud Charges by Hours')
plt.setp(g.get_xticklabels(), visible=False)

plt.show()
test['Date'] = test['TransactionDT'].apply(lambda x : genesis+datetime.timedelta(seconds=x))
# создадим новые фичи:
test['Weekdays'] = test['Date'].dt.dayofweek
test['Days'] = test['Date'].dt.day
test['Hours'] = test['Date'].dt.hour
train.drop(['Date'], axis=1,inplace=True)
test.drop(['Date'], axis=1,inplace=True)
train_cols[train_cols.cat.isnull()].count()
train_cols.head(2)
train_cols.loc[train_cols.index=='TransactionDT', :]
train_cols.loc[train_cols.index=='isFraud', 'cat']='target'
train_cols.loc[train_cols.index=='TransactionDT', 'cat']='datetime'
categorical_columns = list(train_cols.loc[train_cols.cat=='cat',:].index)
numerical_columns = list(train_cols.loc[train_cols.cat=='num',:].index)
categorical_columns
numerical_columns
%%time
X = train[categorical_columns + numerical_columns]
y = train['isFraud']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y)

categorical_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
numerical_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='mean'))
])

preprocessing = ColumnTransformer(
    [('cat', categorical_pipe, categorical_columns),
     ('num', numerical_pipe, numerical_columns)])

rf = Pipeline([
    ('preprocess', preprocessing),
    ('classifier', RandomForestClassifier())
])

rf.fit(X_train, y_train)
print("RF train accuracy: %0.3f" % rf.score(X_train, y_train))
print("RF test accuracy: %0.3f" % rf.score(X_test, y_test))
ohe = (rf.named_steps['preprocess']
         .named_transformers_['cat']
         .named_steps['onehot'])
feature_names = ohe.get_feature_names(input_features=categorical_columns)
feature_names = np.r_[feature_names, numerical_columns]

tree_feature_importances = (
    rf.named_steps['classifier'].feature_importances_)
sorted_idx = tree_feature_importances.argsort()

y_ticks = np.arange(0, len(feature_names))
fig, ax = plt.subplots(figsize=(10,160))
ax.barh(y_ticks, tree_feature_importances[sorted_idx])
ax.set_yticklabels(feature_names[sorted_idx])
ax.set_yticks(y_ticks)
ax.set_title("Random Forest Feature Importances")
fig.tight_layout()
plt.show()
# %%time
# result = permutation_importance(rf, X_test, y_test, n_repeats=5)
# sorted_idx = result.importances_mean.argsort()
# df_tmp = pd.DataFrame()
# for i in range(result['importances'].shape[1]):
#     col_nm = 'col_'+str(i)
# #     print(col_nm)
#     df_tmp[col_nm]=result['importances'][:,i]
# df_tmp.head(3) 
# df_tmp.to_csv('permut_importances.csv', index=False)
df_tmp2 = pd.read_csv('../input/permut-importances02/permut_importances_02.csv')
tmp3 = np.array(df_tmp2)
result2={'importances': tmp3, 'importances_mean': tmp3.mean(axis=1), 'importances_std': tmp3.std(axis=1)}
result = result2
sorted_idx = result['importances_mean'].argsort()

fig, ax = plt.subplots(figsize=(10,160))
ax.boxplot(result['importances'][sorted_idx].T,
           vert=False, labels=X_test.columns[sorted_idx])
ax.set_title("Permutation Importances (test set)")
fig.tight_layout()
plt.show()
df_importances = pd.DataFrame({'predictor': X_test.columns[sorted_idx], 'importance': result['importances_mean'][sorted_idx]})
df_importances['importances_abs'] = df_importances[['importance']].apply(abs)
df_importances = df_importances.sort_values(by='importances_abs',ascending=True)
df_importances.iloc[-5:,:]
plt.figure(figsize=(10,160))
plt.barh(df_importances.predictor, df_importances.importances_abs);
df_importances.iloc[-5:,:]
train_cols2=train_cols.reset_index()
train_cols2=train_cols2.rename(columns={'index': 'predictor1'})
train_cols2 = train_cols2.merge(df_importances, how='left', left_on='predictor1', right_on='predictor')
train_cols2 = train_cols2.sort_values(by='importances_abs',ascending=False).reset_index(drop=True)
train_cols2['importances_abs'] = train_cols2[['importances_abs']].apply(lambda x: x/x.max()*100)
train_cols2[100:160]
col_cat_top = list(train_cols2[:30].loc[train_cols2.cat=='cat','predictor'])
col_num_top = list(train_cols2[:30].loc[train_cols2.cat=='num','predictor'])
for col in col_cat_top:
    print(col)
    plt.subplots(1,2,figsize=(15,5))
    plt.subplot(1,2,1)
    sns.countplot(train[col]);
    plt.subplot(1,2,2)
    sns.barplot(x=col, y='isFraud', data=train);
    plt.show()
for col in col_num_top:
    print(col)
#     plt.subplots(1,1,figsize=(15,5))
#     plt.subplot(1,2,1)
#     sns.countplot(train[col]);
#     plt.subplot(1,2,2)
#     sns.barplot(x=col, y='isFraud', data=train);
    sns.distplot(train[col],kde=False);
    plt.show()
train_cols2.groupby('cat')['cat'].count()
train_cols2.loc[train_cols2.predictor=='TransactionID','cat']='del'
train_cols2.loc[train_cols2.predictor=='TransactionID',:]

train_cols2.loc[train_cols2.importances_abs<=3,'cat']='del2'
train_cols2.groupby('cat')['cat'].count()
train_cols = train_cols2
train_cols.index = train_cols['predictor1']
train_cols.head(2)
train['TransactionAmt_to_mean_card1'] = train['TransactionAmt'] / train.groupby(['card1'])['TransactionAmt'].transform('mean')
train['TransactionAmt_to_mean_card4'] = train['TransactionAmt'] / train.groupby(['card4'])['TransactionAmt'].transform('mean')
train['TransactionAmt_to_std_card1'] = train['TransactionAmt'] / train.groupby(['card1'])['TransactionAmt'].transform('std')
train['TransactionAmt_to_std_card4'] = train['TransactionAmt'] / train.groupby(['card4'])['TransactionAmt'].transform('std')

test['TransactionAmt_to_mean_card1'] = test['TransactionAmt'] / test.groupby(['card1'])['TransactionAmt'].transform('mean')
test['TransactionAmt_to_mean_card4'] = test['TransactionAmt'] / test.groupby(['card4'])['TransactionAmt'].transform('mean')
test['TransactionAmt_to_std_card1'] = test['TransactionAmt'] / test.groupby(['card1'])['TransactionAmt'].transform('std')
test['TransactionAmt_to_std_card4'] = test['TransactionAmt'] / test.groupby(['card4'])['TransactionAmt'].transform('std')


train['D15_to_mean_card1'] = train['D15'] / train.groupby(['card1'])['D15'].transform('mean')
train['D15_to_mean_card4'] = train['D15'] / train.groupby(['card4'])['D15'].transform('mean')
train['D15_to_std_card1'] = train['D15'] / train.groupby(['card1'])['D15'].transform('std')
train['D15_to_std_card4'] = train['D15'] / train.groupby(['card4'])['D15'].transform('std')

test['D15_to_mean_card1'] = test['D15'] / test.groupby(['card1'])['D15'].transform('mean')
test['D15_to_mean_card4'] = test['D15'] / test.groupby(['card4'])['D15'].transform('mean')
test['D15_to_std_card1'] = test['D15'] / test.groupby(['card1'])['D15'].transform('std')
test['D15_to_std_card4'] = test['D15'] / test.groupby(['card4'])['D15'].transform('std')

train['D15_to_mean_addr1'] = train['D15'] / train.groupby(['addr1'])['D15'].transform('mean')
train['D15_to_mean_addr2'] = train['D15'] / train.groupby(['addr2'])['D15'].transform('mean')
train['D15_to_std_addr1'] = train['D15'] / train.groupby(['addr1'])['D15'].transform('std')
train['D15_to_std_addr2'] = train['D15'] / train.groupby(['addr2'])['D15'].transform('std')

test['D15_to_mean_addr1'] = test['D15'] / test.groupby(['addr1'])['D15'].transform('mean')
test['D15_to_mean_addr2'] = test['D15'] / test.groupby(['addr2'])['D15'].transform('mean')
test['D15_to_std_addr1'] = test['D15'] / test.groupby(['addr1'])['D15'].transform('std')
test['D15_to_std_addr2'] = test['D15'] / test.groupby(['addr2'])['D15'].transform('std')
numerical_columns_02 = ['TransactionAmt_to_mean_card1','TransactionAmt_to_mean_card4','TransactionAmt_to_std_card1','TransactionAmt_to_std_card4','D15_to_mean_card1','D15_to_mean_card4','D15_to_std_card1','D15_to_std_card4','D15_to_mean_addr1','D15_to_mean_addr2','D15_to_std_addr1','D15_to_std_addr2']
# by https://www.kaggle.com/dimartinot
def clean_inf_nan(df):
    return df.replace([np.inf, -np.inf], np.nan)   

# Cleaning infinite values to NaN
train = clean_inf_nan(train)
test = clean_inf_nan(test )
# df_tmp = pd.DataFrame({'col': categorical_columns + numerical_columns})
# df_tmp.groupby('col')['col'].count().sort_values( ascending=False)
categorical_columns = list(train_cols.loc[train_cols.cat=='cat',:].index)
numerical_columns = list(train_cols.loc[train_cols.cat=='num',:].index) + numerical_columns_02

X = train[categorical_columns + numerical_columns]
y = train['isFraud']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y)

categorical_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
numerical_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='mean'))
])

preprocessing = ColumnTransformer(
    [('cat', categorical_pipe, categorical_columns),
     ('num', numerical_pipe, numerical_columns)])

pipe = Pipeline([
    ('preprocess', preprocessing),
    ('classifier', RandomForestClassifier())
])
param_grid = {
    'classifier__max_depth': [2, 5, 20]
}

grid_search_01 = GridSearchCV(pipe, param_grid, scoring='roc_auc',cv=3)
%%time
grid_search_01.fit(X_train,y_train)
grid_search_01.best_score_ 
y_pred = grid_search_01.predict(X_test)
y_pred_proba = grid_search_01.predict_proba(X_test)[:,1]
auc_res_01 = roc_auc_score(y_test,y_pred_proba)
auc_res_01
grid_search_01.best_estimator_ 
categorical_columns = list(train_cols.loc[train_cols.cat=='cat',:].index)
numerical_columns = list(train_cols.loc[train_cols.cat=='num',:].index) + numerical_columns_02

X = train[categorical_columns + numerical_columns]
y = train['isFraud']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y)

categorical_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
numerical_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='mean'))
])

preprocessing = ColumnTransformer(
    [('cat', categorical_pipe, categorical_columns),
     ('num', numerical_pipe, numerical_columns)])

pipe = Pipeline([
    ('preprocess', preprocessing),
    ('classifier', xgb.XGBClassifier())
])
param_grid = {
    'classifier__max_depth': [20,30]
}

grid_search_02 = GridSearchCV(pipe, param_grid, scoring='roc_auc',cv=3)
%%time
grid_search_02.fit(X_train,y_train)
grid_search_02.best_score_
y_pred = grid_search_02.predict(X_test)
y_pred_proba = grid_search_02.predict_proba(X_test)[:,1]
auc_res_02 = roc_auc_score(y_test,y_pred_proba)
auc_res_02
categorical_columns = list(train_cols.loc[train_cols.cat=='cat',:].index)
numerical_columns = list(train_cols.loc[train_cols.cat=='num',:].index) + numerical_columns_02

sc = StandardScaler()

X = train[categorical_columns + numerical_columns]
y = train['isFraud']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y)

categorical_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
numerical_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='mean'))
])

preprocessing = ColumnTransformer(
    [('cat', categorical_pipe, categorical_columns),
     ('num', numerical_pipe, numerical_columns)])

pipe = Pipeline([
    ('preprocess', preprocessing),
    ('sc', sc),
    ('classifier', LogisticRegression(penalty='l2'))
])
param_grid = {
    'classifier__C': [0.01,0.5,0.99]
}

grid_search_03 = GridSearchCV(pipe, param_grid, scoring='roc_auc',cv=3)
%%time
grid_search_03.fit(X_train,y_train)
grid_search_03.best_score_
y_pred = grid_search_03.predict(X_test)
y_pred_proba = grid_search_03.predict_proba(X_test)[:,1]
auc_res_03 = roc_auc_score(y_test,y_pred_proba)
auc_res_03
mod_res = grid_search_01
y_pred_res = mod_res.predict(test[categorical_columns + numerical_columns])
sub['isFraud'] = y_pred_res
sub.to_csv('Result_v01.csv', index=False)
mod_res = grid_search_02
y_pred_res = mod_res.predict(test[categorical_columns + numerical_columns])
sub['isFraud'] = y_pred_res
sub.to_csv('Result_v02.csv', index=False)
mod_res = grid_search_03
y_pred_res = mod_res.predict(test[categorical_columns + numerical_columns])
sub['isFraud'] = y_pred_res
sub.to_csv('Result_v03.csv', index=False)