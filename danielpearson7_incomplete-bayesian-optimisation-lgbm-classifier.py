import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import RFECV
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from tqdm import tqdm_notebook

import lightgbm as lgb

import gc

import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
#Read files
train_identity = pd.read_csv('../input/ieee-fraud-detection/train_identity.csv')
train_transaction = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv')
test_identity = pd.read_csv('../input/ieee-fraud-detection/test_identity.csv')
test_transaction = pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv')
sub = pd.read_csv('../input/ieee-fraud-detection/sample_submission.csv')
#Merge the training set and also merge the test set based on transaction ID
train = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')
test = pd.merge(test_transaction, test_identity, on='TransactionID', how='left')

train.name = 'Train'
test.name = 'Test'

del test_identity, test_transaction, train_identity, train_transaction
gc.collect()
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    gc.collect()
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
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
    if verbose: 
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
print('Training set shape:',train.shape)
print('Test set shape:',test.shape)
print(train.columns[394:-2])
print(test.columns[393:-2])
test.columns = train.columns.drop('isFraud') #Giving both data sets the same columns, except for 'isFraud'
train.dtypes.value_counts()
#Count how many null values are in the training set

train.isnull().sum().sum()
train.isnull().sum().sort_values(ascending=False)[:25]
many_null_cols = []
many_null_cols_test = []

print('Train dataset columns with more than 90% of missing values: \n')
for col in train.columns:
    if train[col].isnull().sum() / train.shape[0] > 0.9:
        print(col,': ',train[col].isnull().sum() / train.shape[0])
        many_null_cols.append(col)
        
print('\n','-'*30,'\n')
      
print('Test dataset columns with more than 90% of missing values: \n')
for col in test.columns:
    if test[col].isnull().sum() / test.shape[0] > 0.9:
        print(col,': ',test[col].isnull().sum() / test.shape[0])
        many_null_cols_test.append(col)
one_value_cols = []
one_value_cols_test = []

print('Train columns with one unique value:')
for col in train.columns:
    if train[col].nunique() <= 1:
        print(col)
        one_value_cols.append(col)

print('\n','-'*30,'\n')

print('Test columns with one unique value: \n')
for col in test.columns:
    if test[col].nunique() <= 1:
        print(col)
        one_value_cols_test.append(col)
top_value_cols = []
top_value_cols_test = []

print('Train columns where the greatest value occurs over 90% of the time: \n')
for col in train.columns:
    if train[col].value_counts(dropna=False, normalize=True).values[0] > 0.9:
        print(col,': ',train[col].value_counts(dropna=False, normalize=True).values[0])
        top_value_cols.append(col)
        
print('\n','-'*30,'\n')

print('Test columns where the greatest value occurs over 90% of the time: \n')
for col in test.columns:
    if test[col].value_counts(dropna=False, normalize=True).values[0] > 0.9:
        print(col,': ',test[col].value_counts(dropna=False, normalize=True).values[0])
        top_value_cols_test.append(col)
#By using 'set', duplicates are not included more than once
cols_to_drop = list(set(many_null_cols + many_null_cols_test + one_value_cols + one_value_cols_test + top_value_cols + top_value_cols_test))

cols_to_drop.remove('isFraud') #Don't include the target value

print(sorted(cols_to_drop),'\n')
print(len(cols_to_drop), 'columns will be dropped from both the train and test set.')
train = train.drop(cols_to_drop, axis=1)
test = test.drop(cols_to_drop, axis=1)
train = reduce_mem_usage(train)
for col in tqdm_notebook(train.columns): 
    if train[col].dtype == 'object':
        le = LabelEncoder()
        le.fit(list(train[col].astype(str).values) + list(test[col].astype(str).values))
        train[col] = le.transform(list(train[col].astype(str).values))
from bayes_opt import BayesianOptimization
from sklearn.model_selection import train_test_split, StratifiedKFold,KFold
from sklearn.metrics import roc_auc_score

features = list(train)
features.remove('isFraud')
target = 'isFraud'
#cut tr and val
bayesian_tr_idx, bayesian_val_idx = train_test_split(train, test_size = 0.3, random_state = 42, stratify = train[target])
bayesian_tr_idx = bayesian_tr_idx.index
bayesian_val_idx = bayesian_val_idx.index
#black box LGBM 
def LGB_bayesian(
    #learning_rate,
    num_leaves, 
    bagging_fraction,
    feature_fraction,
    min_child_weight, 
    min_data_in_leaf,
    max_depth,
    reg_alpha,
    reg_lambda
     ):
    
    # LightGBM expects next three parameters need to be integer. 
    num_leaves = int(num_leaves)
    min_data_in_leaf = int(min_data_in_leaf)
    max_depth = int(max_depth)

    assert type(num_leaves) == int
    assert type(min_data_in_leaf) == int
    assert type(max_depth) == int
    

    param = {
              'num_leaves': num_leaves, 
              'min_data_in_leaf': min_data_in_leaf,
              'min_child_weight': min_child_weight,
              'bagging_fraction' : bagging_fraction,
              'feature_fraction' : feature_fraction,
              #'learning_rate' : learning_rate,
              'max_depth': max_depth,
              'reg_alpha': reg_alpha,
              'reg_lambda': reg_lambda,
              'objective': 'binary',
              'save_binary': True,
              'seed': 42,
              'feature_fraction_seed': 42,
              'bagging_seed': 42,
              'drop_seed': 42,
              'data_random_seed': 42,
              'boosting_type': 'gbdt',
              'verbose': 1,
              'is_unbalance': False,
              'boost_from_average': True,
              'metric':'auc'}    
    
    oof = np.zeros(len(train))
    trn_data= lgb.Dataset(train.iloc[bayesian_tr_idx][features].values, label=train.iloc[bayesian_tr_idx][target].values)
    val_data= lgb.Dataset(train.iloc[bayesian_val_idx][features].values, label=train.iloc[bayesian_val_idx][target].values)

    clf = lgb.train(param, trn_data,  num_boost_round=50, valid_sets = [trn_data, val_data], verbose_eval=0, early_stopping_rounds = 50)
    
    oof[bayesian_val_idx]  = clf.predict(train.iloc[bayesian_val_idx][features].values, num_iteration=clf.best_iteration)  
    
    score = roc_auc_score(train.iloc[bayesian_val_idx][target].values, oof[bayesian_val_idx])

    return score
bounds_LGB = {
    'num_leaves': (31, 500), 
    'min_data_in_leaf': (20, 200),
    'bagging_fraction' : (0.1, 0.9),
    'feature_fraction' : (0.1, 0.9),
    #'learning_rate': (0.01, 0.3),
    'min_child_weight': (0.00001, 0.01),   
    'reg_alpha': (1, 2), 
    'reg_lambda': (1, 2),
    'max_depth':(-1,50),
}
LGB_BO = BayesianOptimization(LGB_bayesian, bounds_LGB, random_state=42)
print(LGB_BO.space.keys)
init_points = 10
n_iter = 15
print('-' * 130)
LGB_BO.maximize(init_points=init_points, n_iter=n_iter, acq='ucb', xi=0.0, alpha=1e-6)
print(LGB_BO.max)
optimal_params = {
        'min_data_in_leaf': int(LGB_BO.max['params']['min_data_in_leaf']), 
        'num_leaves': int(LGB_BO.max['params']['num_leaves']), 
        #'learning_rate': LGB_BO.max['params']['learning_rate'],
        'min_child_weight': LGB_BO.max['params']['min_child_weight'],
        'bagging_fraction': LGB_BO.max['params']['bagging_fraction'], 
        'feature_fraction': LGB_BO.max['params']['feature_fraction'],
        'reg_lambda': LGB_BO.max['params']['reg_lambda'],
        'reg_alpha': LGB_BO.max['params']['reg_alpha'],
        'max_depth': int(LGB_BO.max['params']['max_depth']), 
        'objective': 'binary',
        'save_binary': True,
        'seed': 42,
        'feature_fraction_seed': 42,
        'bagging_seed': 42,
        'drop_seed': 42,
        'data_random_seed': 42,
        'boosting_type': 'gbdt',
        'verbose': 1,
        'is_unbalance': False,
        'boost_from_average': True,
        'metric':'auc'
    }
clf = lgb.LGBMClassifier(**optimal_params)
rfe = RFECV(estimator=clf, step=10, cv=KFold(n_splits=5, shuffle=False), scoring='roc_auc', verbose=1)
X = train.sort_values('TransactionDT').drop(['isFraud', 'TransactionDT', 'TransactionID'], axis=1)
y = train.sort_values('TransactionDT')['isFraud']
#rfe.fit(X, y)
%%timeit
clf = lgb.LGBMClassifier(**optimal_params, verbosity=2)
clf.fit(X, y)
test1 = test.sort_values('TransactionDT').drop(['TransactionDT', 'TransactionID'], axis=1)

