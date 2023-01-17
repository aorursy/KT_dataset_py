import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression


import lightgbm as lgb
import gc # memory 
from datetime import datetime # train time checking

pd.set_option('display.max_columns', 100)
#VALIDATION
VALIDATE = False  #validation using train_test_split
VALID_SIZE = 0.90 # simple validation using train_test_split

#CROSS-VALIDATION
VALIDATE_KFOLDS = True #cross-validation using KFolds
NUMBER_KFOLDS = 5 #number of KFolds for cross-validation

SAMPLE = True    #True: use train.sample (100,000 rows) False: use full training set (train)
RANDOM_STATE = 2018

MAX_ROUNDS = 1000 #lgb iterations
EARLY_STOP = 50  #lgb early stop 
OPT_ROUNDS = 650  #To be adjusted based on best validation rounds
skiprows = range(1,109903891) #
nrows = 75000000
#USE SAMPLE FROM FULL TRAIN SET
SAMPLE_SIZE = 1 # use a subsample of the train set
output_filename = 'submission.csv'

IS_LOCAL = False


if (IS_LOCAL):
    PATH = '../input/talkingdata-adtracking-fraud-detection/'
else:
    PATH = '../input/'
print(os.listdir(PATH))
dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }

train_cols = ['ip','app','device','os', 'channel', 'click_time', 'is_attributed']

if SAMPLE:
    trainset = pd.read_csv(PATH+"train_sample.csv", dtype=dtypes, usecols=train_cols)    
else:
    trainset = pd.read_csv(PATH+"train.csv", skiprows=skiprows, nrows=nrows,dtype=dtypes, usecols=train_cols)
    trainset = trainset.sample(frac=SAMPLE_SIZE)

testset = pd.read_csv(PATH+"test.csv")
print("train -  rows:",trainset.shape[0]," columns:", trainset.shape[1])
print("test -  rows:",testset.shape[0]," columns:", testset.shape[1])

trainset.head()

testset.head()
def missing_data(data):
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data(trainset)
missing_data(testset)
plt.figure()
fig, ax = plt.subplots(figsize=(6,6))
x = trainset['is_attributed'].value_counts().index.values
y = trainset["is_attributed"].value_counts().values
# Bar plot
# Order the bars descending on target mean
sns.barplot(ax=ax, x=x, y=y)
plt.ylabel('Number of values', fontsize=12)
plt.xlabel('is_attributed value', fontsize=12)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.show();
trainset.describe()
t1 = trainset.loc[trainset['is_attributed'] != 0.]
t0 = trainset.loc[trainset['is_attributed'] == 0.]

t0.describe()
t1.describe()
var = ['app','device','os','channel']

i = 0
t1 = trainset.loc[trainset['is_attributed'] != 0]
t0 = trainset.loc[trainset['is_attributed'] == 0]

sns.set_style('whitegrid')
plt.figure()
fig, ax = plt.subplots(1,4,figsize=(16,4))

for feature in var:
    i += 1
    plt.subplot(1,4,i)
    sns.kdeplot(t1[feature], bw=0.5,label="is_attributed = 1")
    sns.kdeplot(t0[feature], bw=0.5,label="is_attributed = 0")
    plt.ylabel('Density plot', fontsize=12)
    plt.xlabel(feature, fontsize=12)
    locs, labels = plt.xticks()
    plt.tick_params(axis='both', which='major', labelsize=12)
plt.show();
trainset['year'] = pd.to_datetime(trainset.click_time).dt.year
trainset['month'] = pd.to_datetime(trainset.click_time).dt.month
trainset['day'] = pd.to_datetime(trainset.click_time).dt.day
trainset['hour'] = pd.to_datetime(trainset.click_time).dt.hour
trainset['min'] = pd.to_datetime(trainset.click_time).dt.minute
trainset['sec'] = pd.to_datetime(trainset.click_time).dt.second
trainset.head()
trainset.describe()
var = ['day','hour']

i = 0
t1 = trainset.loc[trainset['is_attributed'] != 0]
t0 = trainset.loc[trainset['is_attributed'] == 0]

sns.set_style('whitegrid')
plt.figure()
fig, ax = plt.subplots(1,2,figsize=(12,4))

for feature in var:
    i += 1
    plt.subplot(1,2,i)
    sns.kdeplot(t1[feature], bw=0.5,label="is_attributed = 1")
    sns.kdeplot(t0[feature], bw=0.5,label="is_attributed = 0")
    plt.ylabel('Density plot', fontsize=12)
    plt.xlabel(feature, fontsize=12)
    locs, labels = plt.xticks()
    plt.tick_params(axis='both', which='major', labelsize=12)
plt.show();
var = ['day','hour']

for feature in var:
    fig, ax = plt.subplots(figsize=(16,6))
    # Calculate the percentage of target=1 per category value
    cat_perc = trainset[[feature, 'is_attributed']].groupby([feature],as_index=False).mean()
    cat_perc.sort_values(by='is_attributed', ascending=False, inplace=True)
    # Bar plot
    #sns.barplot(ax=ax,x=feature, y='is_attributed', data=cat_perc, order=cat_perc[feature]) #for ordered bars
    sns.barplot(ax=ax,x=feature, y='is_attributed', data=cat_perc)
    plt.ylabel('Percent of `is_attributed` with value 1 [%]', fontsize=12)
    plt.xlabel(feature, fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.show();
#------------------------------------------------------------------------------
def show_max_clean(df,gp,agg_name,agg_type,show_max):
#------------------------------------------------------------------------------    
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    gc.collect()
    return( df )
    
#------------------------------------------------------------------------------
def perform_count( df, group_cols, agg_name, agg_type='uint32', show_max=False, show_agg=True ):
#------------------------------------------------------------------------------
    if show_agg:
        print( "Aggregating by ", group_cols , '...' )
    gp = df[group_cols][group_cols].groupby(group_cols).size().rename(agg_name).to_frame().reset_index()
    df = df.merge(gp, on=group_cols, how='left')
    return (show_max_clean(df,gp,agg_name,agg_type,show_max))

#------------------------------------------------------------------------------
def perform_countuniq( df, group_cols, counted, agg_name, agg_type='uint32', show_max=False, show_agg=True ):
#------------------------------------------------------------------------------    
    if show_agg:
        print( "Counting unique ", counted, " by ", group_cols , '...' )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].nunique().reset_index().rename(columns={counted:agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    return (show_max_clean(df,gp,agg_name,agg_type,show_max))

#------------------------------------------------------------------------------    
def perform_cumcount( df, group_cols, counted, agg_name, agg_type='uint32', show_max=False, show_agg=True ):
#------------------------------------------------------------------------------    
    if show_agg:
        print( "Cumulative count by ", group_cols , '...' )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].cumcount()
    df[agg_name]=gp.values
    return (show_max_clean(df,gp,agg_name,agg_type,show_max))

#------------------------------------------------------------------------------
def perform_mean( df, group_cols, counted, agg_name, agg_type='float32', show_max=False, show_agg=True ):
#------------------------------------------------------------------------------    
    if show_agg:
        print( "Calculating mean of ", counted, " by ", group_cols , '...' )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].mean().reset_index().rename(columns={counted:agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    return (show_max_clean(df,gp,agg_name,agg_type,show_max))

#------------------------------------------------------------------------------
def perform_var( df, group_cols, counted, agg_name, agg_type='float32', show_max=False, show_agg=True ):
#------------------------------------------------------------------------------    
    if show_agg:
        print( "Calculating variance of ", counted, " by ", group_cols , '...' )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].var().reset_index().rename(columns={counted:agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    return (show_max_clean(df,gp,agg_name,agg_type,show_max))
trainset = perform_countuniq( trainset, ['ip'], 'channel', 'X0', 'uint8', show_max=True ); gc.collect()
trainset = perform_cumcount( trainset, ['ip', 'device', 'os'], 'app', 'X1', show_max=True ); gc.collect()
trainset = perform_countuniq( trainset, ['ip', 'day'], 'hour', 'X2', 'uint8', show_max=True ); gc.collect()
trainset = perform_countuniq( trainset, ['ip'], 'app', 'X3', 'uint8', show_max=True ); gc.collect()
trainset = perform_countuniq( trainset, ['ip', 'app'], 'os', 'X4', 'uint8', show_max=True ); gc.collect()
trainset = perform_countuniq( trainset, ['ip'], 'device', 'X5', 'uint16', show_max=True ); gc.collect()
trainset = perform_countuniq( trainset, ['app'], 'channel', 'X6', show_max=True ); gc.collect()
trainset = perform_cumcount( trainset, ['ip'], 'os', 'X7', show_max=True ); gc.collect()
trainset = perform_countuniq( trainset, ['ip', 'device', 'os'], 'app', 'X8', show_max=True ); gc.collect()
trainset = perform_count( trainset, ['ip', 'day', 'hour'], 'ip_tcount', show_max=True ); gc.collect()
trainset = perform_count( trainset, ['ip', 'app'], 'ip_app_count', show_max=True ); gc.collect()
trainset = perform_count( trainset, ['ip', 'app', 'os'], 'ip_app_os_count', 'uint16', show_max=True ); gc.collect()
trainset = perform_var( trainset, ['ip', 'day', 'channel'], 'hour', 'ip_tchan_count', show_max=True ); gc.collect()
trainset = perform_var( trainset, ['ip', 'app', 'os'], 'hour', 'ip_app_os_var', show_max=True ); gc.collect()
trainset = perform_var( trainset, ['ip', 'app', 'channel'], 'day', 'ip_app_channel_var_day', show_max=True ); gc.collect()
trainset = perform_mean( trainset, ['ip', 'app', 'channel'], 'hour', 'ip_app_channel_mean_hour', show_max=True ); gc.collect()
trainset.head(5)
var = ['X0','X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7']

i = 0
t1 = trainset.loc[trainset['is_attributed'] != 0]
t0 = trainset.loc[trainset['is_attributed'] == 0]

sns.set_style('whitegrid')
plt.figure()
fig, ax = plt.subplots(2,4,figsize=(14,8))

for feature in var:
    i += 1
    plt.subplot(2,4,i)
    sns.kdeplot(t1[feature], bw=0.5,label="is_attributed = 1")
    sns.kdeplot(t0[feature], bw=0.5,label="is_attributed = 0")
    plt.ylabel('Density plot', fontsize=12)
    plt.xlabel(feature, fontsize=12)
    locs, labels = plt.xticks()
    plt.tick_params(axis='both', which='major', labelsize=12)
plt.show();
var = ['X8', 'ip_tcount', 'ip_app_count','ip_app_os_count', 
        'ip_tchan_count','ip_app_os_var','ip_app_channel_var_day', 'ip_app_channel_mean_hour']

i = 0
t1 = trainset.loc[trainset['is_attributed'] != 0]
t0 = trainset.loc[trainset['is_attributed'] == 0]

sns.set_style('whitegrid')
plt.figure()
fig, ax = plt.subplots(2,4,figsize=(14,8))

for feature in var:
    i += 1
    plt.subplot(2,4,i)
    sns.kdeplot(t1[feature], bw=0.5,label="is_attributed = 1")
    sns.kdeplot(t0[feature], bw=0.5,label="is_attributed = 0")
    plt.ylabel('Density plot', fontsize=12)
    plt.xlabel(feature, fontsize=12)
    locs, labels = plt.xticks()
    plt.tick_params(axis='both', which='major', labelsize=12)
plt.show();
var = ['app','device','os','channel']

# Bar plot
sns.set_style('whitegrid')

plt.figure()
fig, ax = plt.subplots(1,4,figsize=(16,4))
i = 0
for feature in var:
    i = i + 1
    plt.subplot(1,4,i)
    sns.kdeplot(trainset[feature], bw=0.5, label="train")
    sns.kdeplot(testset[feature], bw=0.5, label="test")
    plt.ylabel('Distribution', fontsize=12)
    plt.xlabel(feature, fontsize=12)
    locs, labels = plt.xticks()
    #plt.setp(labels, rotation=90)
    plt.tick_params(axis='both', which='major', labelsize=12)
plt.show();
testset['year'] = pd.to_datetime(testset.click_time).dt.year
testset['month'] = pd.to_datetime(testset.click_time).dt.month
testset['day'] = pd.to_datetime(testset.click_time).dt.day
testset['hour'] = pd.to_datetime(testset.click_time).dt.hour
testset['min'] = pd.to_datetime(testset.click_time).dt.minute
testset['sec'] = pd.to_datetime(testset.click_time).dt.second
testset.head()
testset.describe()
var = ['day','hour']

sns.set_style('whitegrid')

plt.figure()
fig, ax = plt.subplots(1,2,figsize=(12,4))
i = 0
for feature in var:
    i = i + 1
    plt.subplot(1,2,i)
    sns.kdeplot(trainset[feature], bw=0.5, label="train")
    sns.kdeplot(testset[feature], bw=0.5, label="test")
    plt.ylabel('Distribution', fontsize=12)
    plt.xlabel(feature, fontsize=12)
    locs, labels = plt.xticks()
    #plt.setp(labels, rotation=90)
    plt.tick_params(axis='both', which='major', labelsize=12)
plt.show();
testset = perform_countuniq( testset, ['ip'], 'channel', 'X0', 'uint8', show_max=True ); gc.collect()
testset = perform_cumcount( testset, ['ip', 'device', 'os'], 'app', 'X1', show_max=True ); gc.collect()
testset = perform_countuniq( testset, ['ip', 'day'], 'hour', 'X2', 'uint8', show_max=True ); gc.collect()
testset = perform_countuniq( testset, ['ip'], 'app', 'X3', 'uint8', show_max=True ); gc.collect()
testset = perform_countuniq( testset, ['ip', 'app'], 'os', 'X4', 'uint8', show_max=True ); gc.collect()
testset = perform_countuniq( testset, ['ip'], 'device', 'X5', 'uint16', show_max=True ); gc.collect()
testset = perform_countuniq( testset, ['app'], 'channel', 'X6', show_max=True ); gc.collect()
testset = perform_cumcount( testset, ['ip'], 'os', 'X7', show_max=True ); gc.collect()
testset = perform_countuniq( testset, ['ip', 'device', 'os'], 'app', 'X8', show_max=True ); gc.collect()
testset = perform_count( testset, ['ip', 'day', 'hour'], 'ip_tcount', show_max=True ); gc.collect()
testset = perform_count( testset, ['ip', 'app'], 'ip_app_count', show_max=True ); gc.collect()
testset = perform_count( testset, ['ip', 'app', 'os'], 'ip_app_os_count', 'uint16', show_max=True ); gc.collect()
testset = perform_var( testset, ['ip', 'day', 'channel'], 'hour', 'ip_tchan_count', show_max=True ); gc.collect()
testset = perform_var( testset, ['ip', 'app', 'os'], 'hour', 'ip_app_os_var', show_max=True ); gc.collect()
testset = perform_var( testset, ['ip', 'app', 'channel'], 'day', 'ip_app_channel_var_day', show_max=True ); gc.collect()
testset = perform_mean( testset, ['ip', 'app', 'channel'], 'hour', 'ip_app_channel_mean_hour', show_max=True ); gc.collect()
var = ['X0','X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7']

sns.set_style('whitegrid')

plt.figure()
fig, ax = plt.subplots(2,4,figsize=(14,8))
i = 0
for feature in var:
    i = i + 1
    plt.subplot(2,4,i)
    sns.kdeplot(trainset[feature], bw=0.5, label="train")
    sns.kdeplot(testset[feature], bw=0.5, label="test")
    plt.ylabel('Distribution', fontsize=12)
    plt.xlabel(feature, fontsize=12)
    locs, labels = plt.xticks()
    #plt.setp(labels, rotation=90)
    plt.tick_params(axis='both', which='major', labelsize=12)
plt.show();
var = ['X8', 'ip_tcount', 'ip_app_count','ip_app_os_count', 
       'ip_tchan_count', 'ip_app_os_var','ip_app_channel_var_day', 'ip_app_channel_mean_hour']

sns.set_style('whitegrid')

plt.figure()
fig, ax = plt.subplots(2,4,figsize=(14,8))
i = 0
for feature in var:
    i = i + 1
    plt.subplot(2,4,i)
    sns.kdeplot(trainset[feature], bw=0.5, label="train")
    sns.kdeplot(testset[feature], bw=0.5, label="test")
    plt.ylabel('Distribution', fontsize=12)
    plt.xlabel(feature, fontsize=12)
    locs, labels = plt.xticks()
    #plt.setp(labels, rotation=90)
    plt.tick_params(axis='both', which='major', labelsize=12)
plt.show();
start = datetime.now()

len_train = len(trainset)
gc.collect()

most_freq_hours_in_test_data = [4, 5, 9, 10, 13, 14]
least_freq_hours_in_test_data = [6, 11, 15]

def prep_data( df ):
    
    df['hour'] = pd.to_datetime(df.click_time).dt.hour.astype('uint8')
    df['day'] = pd.to_datetime(df.click_time).dt.day.astype('uint8')
    df.drop(['click_time'], axis=1, inplace=True)
    gc.collect()
    
    df['in_test_hh'] = (   3 
                         - 2*df['hour'].isin(  most_freq_hours_in_test_data ) 
                         - 1*df['hour'].isin( least_freq_hours_in_test_data ) ).astype('uint8')
    gp = df[['ip', 'day', 'in_test_hh', 'channel']].groupby(by=['ip', 'day', 'in_test_hh'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'nip_day_test_hh'})
    df = df.merge(gp, on=['ip','day','in_test_hh'], how='left')
    df.drop(['in_test_hh'], axis=1, inplace=True)
    df['nip_day_test_hh'] = df['nip_day_test_hh'].astype('uint32')
    del gp
    gc.collect()

    gp = df[['ip', 'day', 'hour', 'channel']].groupby(by=['ip', 'day', 'hour'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'nip_day_hh'})
    df = df.merge(gp, on=['ip','day','hour'], how='left')
    df['nip_day_hh'] = df['nip_day_hh'].astype('uint16')
    del gp
    gc.collect()
    
    gp = df[['ip', 'os', 'hour', 'channel']].groupby(by=['ip', 'os', 'hour'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'nip_hh_os'})
    df = df.merge(gp, on=['ip','os','hour'], how='left')
    df['nip_hh_os'] = df['nip_hh_os'].astype('uint16')
    del gp
    gc.collect()

    gp = df[['ip', 'app', 'hour', 'channel']].groupby(by=['ip', 'app',  'hour'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'nip_hh_app'})
    df = df.merge(gp, on=['ip','app','hour'], how='left')
    df['nip_hh_app'] = df['nip_hh_app'].astype('uint16')
    del gp
    gc.collect()

    gp = df[['ip', 'device', 'hour', 'channel']].groupby(by=['ip', 'device', 'hour'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'nip_hh_dev'})
    df = df.merge(gp, on=['ip','device','hour'], how='left')
    df['nip_hh_dev'] = df['nip_hh_dev'].astype('uint32')
    del gp
    gc.collect()

    df.drop( ['ip','day'], axis=1, inplace=True )
    gc.collect()
    return df
trainset = prep_data(trainset)
gc.collect()

params = {
          'boosting_type': 'gbdt',
          'objective': 'binary',
          'metric':'auc',
          'learning_rate': 0.1,
          'num_leaves': 9,  # we should let it be smaller than 2^(max_depth)
          'max_depth': 5,  # -1 means no limit
          'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
          'max_bin': 100,  # Number of bucketed bin for feature values
          'subsample': 0.9,  # Subsample ratio of the training instance.
          'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
          'colsample_bytree': 0.7,  # Subsample ratio of columns when constructing each tree.
          'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
          'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
          'nthread': 8,
          'verbose': 0,
          'scale_pos_weight':99.7, # because training data is extremely unbalanced 
         }

target = 'is_attributed'
predictors = ['app','device','os', 'channel', 'hour', 'nip_day_test_hh', 'nip_day_hh', 'nip_hh_os', 'nip_hh_app', 'nip_hh_dev']
categorical = ['app', 'device', 'os', 'channel', 'hour']

if VALIDATE:

    train_df, val_df = train_test_split(trainset, test_size=VALID_SIZE, random_state=RANDOM_STATE, shuffle=True )
    
    dtrain = lgb.Dataset(train_df[predictors].values, 
                         label=train_df[target].values,
                         feature_name=predictors,
                         categorical_feature=categorical)
    del train_df
    gc.collect()

    dvalid = lgb.Dataset(val_df[predictors].values,
                         label=val_df[target].values,
                         feature_name=predictors,
                         categorical_feature=categorical)
    del val_df
    gc.collect()

    evals_results = {}

    model = lgb.train(params, 
                      dtrain, 
                      valid_sets=[dtrain, dvalid], 
                      valid_names=['train','valid'], 
                      evals_result=evals_results, 
                      num_boost_round=MAX_ROUNDS,
                      early_stopping_rounds=EARLY_STOP,
                      verbose_eval=50, 
                      feval=None)

    del dvalid
    
elif VALIDATE_KFOLDS:
    kf = KFold(n_splits = NUMBER_KFOLDS, random_state = RANDOM_STATE, shuffle = True)
    for train_index, test_index in kf.split(trainset):
        train_X, valid_X = trainset.iloc[train_index], trainset.iloc[test_index]

        dtrain = lgb.Dataset(train_X[predictors].values, label=train_X[target].values,
                         feature_name=predictors, categorical_feature=categorical)
   
        dvalid = lgb.Dataset(valid_X[predictors].values, label=valid_X[target].values,
                         feature_name=predictors, categorical_feature=categorical)
    
        evals_results = {}
        model =  lgb.train(params, 
                      dtrain, 
                      valid_sets=[dtrain, dvalid], 
                      valid_names=['train','valid'], 
                      evals_result=evals_results, 
                      num_boost_round=MAX_ROUNDS,
                      early_stopping_rounds=EARLY_STOP,
                      verbose_eval=50, 
                      feval=None)
    
else:

    gc.collect()
    dtrain = lgb.Dataset(train_df[predictors].values, label=train_df[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical
                          )
    del train_df
    gc.collect()

    evals_results = {}

    model = lgb.train(params, 
                      dtrain, 
                      valid_sets=[dtrain], 
                      valid_names=['train'], 
                      evals_result=evals_results, 
                      num_boost_round=OPT_ROUNDS,
                      verbose_eval=50,
                      feval=None)
    
del dtrain
gc.collect()
test_cols = ['ip','app','device','os', 'channel', 'click_time', 'click_id']

test_df = prep_data(testset)
gc.collect()

sub = pd.DataFrame()
sub['click_id'] = test_df['click_id']
sub['is_attributed'] = model.predict(test_df[predictors])
sub.to_csv(output_filename, index=False, float_format='%.9f')
