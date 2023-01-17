import numpy as np

import pandas as pd

pd.set_option('max_columns', None)

import matplotlib.pyplot as plt

plt.style.use('ggplot')

import datetime

from scipy import stats

from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection import train_test_split, KFold

from wordcloud import WordCloud

from collections import Counter

from nltk.corpus import stopwords

from nltk.util import ngrams

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.preprocessing import StandardScaler,MinMaxScaler, Imputer,LabelEncoder

from sklearn.metrics import roc_auc_score



import gc

import re

import seaborn as sns

import os

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import xgboost as xgb

import lightgbm as lgb

from sklearn.linear_model import LogisticRegression

from sklearn import model_selection

from sklearn.metrics import accuracy_score

import json

import ast

# import shap

from urllib.request import urlopen

from PIL import Image

import time

from sklearn.metrics import mean_squared_error
user_info = pd.read_csv('../input/digix-data/user_info.csv',

                        names=['uId','age','gender','city','province','phoneType','carrier'])

user_info.dtypes

for col in user_info:

    if user_info[col].dtype == 'int64':

        user_info[col] = user_info[col].astype(np.int32)

        

ad_info = pd.read_csv('../input/digix-data/ad_info.csv',

                      names=['adId','billId','primId','creativeType','intertype','spreadAppId'])

for col in ad_info:

    if ad_info[col].dtype == 'int64':

        ad_info[col] = ad_info[col].astype(np.int32)

        

content_info = pd.read_csv('../input/digix-data/content_info.csv',

                           names=['contentId','firstClass','secondClass'])



test = pd.read_csv('../input/digix-test/test_20190518.csv',

                   names=['num','uId','adId','operTime','siteId','slotId','contentId','netType'])

for col in test:

    if test[col].dtype == 'int64':

        test[col] = test[col].astype(np.int32)

        

train_test_user = pd.read_csv('../input/digis-sampl/train_test_user.csv' ) # ,nrows=100000

train_test_user.dtypes

for col in train_test_user:

    if train_test_user[col].dtype == 'float64':

        train_test_user[col] = train_test_user[col].astype(np.float32)

train_test_user = train_test_user.drop_duplicates(keep='first').reset_index(drop=True)



train_test_user_0 = train_test_user[ train_test_user['label'] == 0 ]

train_test_user_1 = train_test_user[ train_test_user['label'] == 1 ]

print(train_test_user_0.shape,train_test_user_1.shape)



train_test_user_0 = train_test_user_0.sample(n = 1300000 , random_state = 42).reset_index(drop=True)

train_test_user_1 = train_test_user_1.sample(n = 550000 , random_state = 42).reset_index(drop=True)

print(train_test_user_0.shape,train_test_user_1.shape)



train_test_user=pd.concat([train_test_user_0,train_test_user_1]).reset_index(drop=True)

train_test_user = train_test_user.sample(frac = 1, random_state = 32).reset_index(drop=True)

print(train_test_user.shape)

print(train_test_user['label'].value_counts())



train = pd.DataFrame()

train = train_test_user.drop(['age','gender','city','province','phoneType','carrier'],axis=1)

train['label'].value_counts()

train['label'].astype(int).plot.hist();



train = pd.merge(train, ad_info, how='left', on=['adId'])

train = pd.merge(train, content_info, how='left', on=['contentId'])

train = pd.merge(train, user_info, how='left', on=['uId'])

test = pd.merge(test, ad_info, how='left', on=['adId'])

test = pd.merge(test, content_info, how='left', on=['contentId'])

test = pd.merge(test, user_info, how='left', on=['uId'])

user_info = pd.DataFrame()

def miss_col(data):        #列的缺失个数和缺失率

    col_total = data.isnull().sum().sort_values(ascending=False)#从大到小按顺序排每个特征缺失的个数

    col_percent =(data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)#从大到小按顺序排每个特征缺失率    

    return col_total, col_percent 

train_miss_rnum, train_miss_rper = miss_col(train)

user_info_miss_rnum, user_info_miss_rper = miss_col(user_info)

test_miss_rnum, test_miss_rper = miss_col(test)



  # Number of each type of column

train.dtypes.value_counts()

  # Number of unique classes in each object column

train.select_dtypes('object').apply(pd.Series.nunique, axis = 0)

 # 提取aid, uid的全局统计特征

train_len = len(train)

test_len = len(test)

    

df = pd.concat([train.drop(['label'], axis=1), test.drop(['num'], axis=1)], axis=0)

df = pd.merge(df, df.groupby(['uId'])['adId'].nunique().reset_index().rename(

            columns={'adId': 'uId_adId_nunique'}), how='left', on='uId')

    

df = pd.merge(df, df.groupby(['uId'])['siteId'].nunique().reset_index().rename(

            columns={'siteId': 'uId_siteId_nunique'}), how='left', on='uId')

    

df = pd.merge(df, df.groupby(['uId'])['slotId'].nunique().reset_index().rename(

            columns={'slotId': 'uId_slotId_nunique'}), how='left', on='uId')

    

df = pd.merge(df, df.groupby(['uId'])['netType'].nunique().reset_index().rename(

            columns={'netType': 'uId_netType_nunique'}), how='left', on='uId')

    

df = pd.merge(df, df.groupby(['uId'])['contentId'].nunique().reset_index().rename(

            columns={'contentId': 'uId_contentId_nunique'}), how='left', on='uId')



df = pd.merge(df, df.groupby(['adId'])['uId'].nunique().reset_index().rename(

            columns={'uId': 'adId_uId_nunique'}), how='left', on='adId')



df['uId_count'] = df.groupby('uId')['adId'].transform('count')

fea_columns = ['uId_adId_nunique', 'uId_siteId_nunique', 'uId_slotId_nunique',

                   'uId_netType_nunique','uId_contentId_nunique', 'adId_uId_nunique']

        

for col in fea_columns:

    bins = []

    for percent in [0, 20, 35, 50, 65, 85, 100]:

        bins.append(np.percentile(df[col], percent))

    df[col] =  np.digitize(df[col], bins, right=True)

        

count_bins = [10, 20, 40, 80, 100, 150, 200, 300, 500]

df['uId_count'] =  np.digitize(df['uId_count'], count_bins, right=True)

train[fea_columns] = df[fea_columns].iloc[:train_len]

test[fea_columns] = df[fea_columns].iloc[train_len: train_len+test_len]

train['uId_count'] = df['uId_count'].iloc[:train_len]

test['uId_count'] = df['uId_count'].iloc[train_len: train_len+test_len]

df = pd.DataFrame()  

 # 衍生时间特征

train['operTime_1'] = pd.to_datetime(train['operTime'])

test['operTime_1'] = pd.to_datetime(test['operTime'])

  ## getattr 返回属性值年份、月份、周几

def process_date(df):

    date_parts = [ 'month','weekday',  'weekofyear', 'hour','day', 'quarter']

    for part in date_parts:

        part_col = 'operTime' + "_" + part

        df[part_col] = getattr(df['operTime_1'].dt, part).astype(int)    

    return df



train = process_date(train)

test = process_date(test)



train = train.drop(['operTime'], axis=1)

train = train.drop(['operTime_1'], axis=1)

test = test.drop(['operTime'], axis=1)

test = test.drop(['operTime_1'], axis=1)
 ## siteId

train['siteId'].value_counts()



# Set the style of plots

plt.style.use('fivethirtyeight')

# Plot the distribution of Id in siteId

plt.hist(train['siteId'] , edgecolor = 'k', bins = 25)

plt.title('site Id'); plt.xlabel('siteId'); plt.ylabel('Count');



siteId_data_mean = train.groupby(['siteId'])['label'].mean()



train = train.join(pd.get_dummies(train['siteId'],prefix='siteId_'))

test = test.join(pd.get_dummies(test['siteId'],prefix='siteId_'))



 ## slotId

train['slotId'].value_counts()

list_of_slotId = list(train['slotId'].values)

Counter( list_of_slotId ).most_common(10)

top_count_slotId = [m[0] for m in Counter( list_of_slotId  ).most_common(10)]

for g in top_count_slotId:

    train['slotId_' + str(g)] = train['slotId'].apply(lambda x: 1 if g == x else 0)

    test['slotId_' + str(g)] = test['slotId'].apply(lambda x: 1 if g == x else 0)



 ## netType

train['netType'].value_counts()

netType_data = train.groupby(['siteId'])['label'].mean()

train = train.join(pd.get_dummies(train['netType'],prefix='netType_'))

test = test.join(pd.get_dummies(test['netType'],prefix='netType_'))



train = train.drop(['siteId','slotId','netType'], axis=1)

test = test.drop(['siteId','slotId','netType'], axis=1)

# user_info

train.isnull().any(axis=0)

test.isnull().any(axis=0)



age_groups_mean  = train.groupby(['age'])['label'].mean()



train['gender'].value_counts()



user_train = train[['uId', 'label']]

ftup=[('user_label_mean','mean'),('user_label_sum','sum')]

user_train_count = user_train.groupby('uId')['label'].agg(ftup).reset_index()

train = pd.merge(train, user_train_count, how='left', on=['uId'])

test = pd.merge(test, user_train_count, how='left', on=['uId'])



train['city'].value_counts()

train['city']  = train['city'].apply(lambda x: 0 if pd.isna(x) else x) 

test['city']  = test['city'].apply(lambda x: 0 if pd.isna(x) else x)

list_of_city= list(train['city'].values)

Counter( list_of_city ).most_common(10)

top_count_city= [m[0] for m in Counter(list_of_city).most_common(10)]

for g in top_count_city:

    train['city_' + str(g)] = train['city'].apply(lambda x: 1 if g == x else 0)

    test['city_' + str(g)] = test['city'].apply(lambda x: 1 if g == x else 0)



train['province'].value_counts()

train['province']  = train['province'].apply(lambda x: 0 if pd.isna(x) else x) 

test['province']  = test['province'].apply(lambda x: 0 if pd.isna(x) else x)

list_of_province= list(train['province'].values)

Counter( list_of_province ).most_common(15)

top_count_province= [m[0] for m in Counter(list_of_province).most_common(15)]

for g in top_count_province:

    train['province_' + str(g)] = train['province'].apply(lambda x: 1 if g == x else 0)

    test['province_' + str(g)] = test['province'].apply(lambda x: 1 if g == x else 0) 



train['phoneType'].value_counts()

train['phoneType']  = train['phoneType'].apply(lambda x: 0 if pd.isna(x) else x) 

test['phoneType']  = test['phoneType'].apply(lambda x: 0 if pd.isna(x) else x)

list_of_phoneType= list(train['phoneType'].values)

Counter( list_of_phoneType ).most_common(15)

top_count_phoneType= [m[0] for m in Counter(list_of_province).most_common(15)]

for g in top_count_phoneType:

    train['phoneType_' + str(g)] = train['phoneType'].apply(lambda x: 1 if g == x else 0)

    test['phoneType_' + str(g)] = test['phoneType'].apply(lambda x: 1 if g == x else 0)



train['carrier'].value_counts()

train = train.join(pd.get_dummies(train['carrier'],prefix='carrier_'))

test = test.join(pd.get_dummies(test['carrier'],prefix='carrier_')) 



train = train.drop(['city','province','phoneType','carrier'], axis=1)

test = test.drop(['city','province','phoneType','carrier'], axis=1)

  #X_train = pd.merge(train, user_info, how='left', on=['uId'])

  #X_test = pd.merge(test, user_info, how='left', on=['uId'])

# ad_info

ad_info.isnull().any(axis=0)



train['billId'].value_counts()

le = LabelEncoder()

le.fit(train['billId'])

train['billId']= le.transform(train['billId'])

test['billId']= le.transform(test['billId'])



train['primId'].value_counts()

train['primId']  = train['primId'].apply(lambda x: 0 if pd.isna(x) else x) 

test['primId']  = test['primId'].apply(lambda x: 0 if pd.isna(x) else x)

train['primId'].value_counts()

list_of_primId = list(train['primId'].values)

Counter( list_of_primId ).most_common(10)

top_count_primId = [m[0] for m in Counter( list_of_primId  ).most_common(10)]

for g in top_count_primId:

    train['primId_' + str(g)] = train['primId'].apply(lambda x: 1 if g == x else 0)

    test['primId_' + str(g)] = test['primId'].apply(lambda x: 1 if g == x else 0)   



train['creativeType'].value_counts()

train = train.join(pd.get_dummies(train['creativeType'],prefix='creativeType_'))

test = test.join(pd.get_dummies(test['creativeType'],prefix='creativeType_'))



train['intertype'].value_counts()

train = train.join(pd.get_dummies(train['intertype'],prefix='intertype_'))

test = test.join(pd.get_dummies(test['intertype'],prefix='intertype_'))



train['spreadAppId'].value_counts()

train['spreadAppId']  = train['spreadAppId'].apply(lambda x: 0 if pd.isna(x) else x) 

test['spreadAppId']  = test['spreadAppId'].apply(lambda x: 0 if pd.isna(x) else x)

list_of_spreadAppId = list(train['spreadAppId'].values)

top_count_spreadAppId = [m[0] for m in Counter( list_of_spreadAppId  ).most_common(10)]

for g in top_count_spreadAppId:

    train['spreadAppId_' + str(g)] = train['spreadAppId'].apply(lambda x: 1 if g == x else 0)

    test['spreadAppId_' + str(g)] = test['spreadAppId'].apply(lambda x: 1 if g == x else 0)  



train = train.drop(['primId','creativeType','intertype','spreadAppId'], axis=1)

test = test.drop(['primId','creativeType','intertype','spreadAppId'], axis=1)
# content_info

  ## firstClass

content_info.isnull().any(axis=0)

train['firstClass'].value_counts()



list_of_firstClass = list(content_info['firstClass'].values)

Counter( list_of_firstClass ).most_common(16)

top_count_firstClass = [m[0] for m in Counter( list_of_firstClass  ).most_common(16)]

for g in top_count_firstClass:

    train['firstClass_' + g] = train['firstClass'].apply(lambda x: 1 if g in str(x) else 0)

    test['firstClass_' + g] = test['firstClass'].apply(lambda x: 1 if g in str(x) else 0)



train['secondClass_na']  = train['secondClass'].apply(lambda x: 1 if pd.isna(x) else 0)                  

train['len_secondClass'] = train['secondClass'].apply(lambda x: 0 if pd.isna(x) else len(re.split('#',x)))  

test['secondClass_na']  = test['secondClass'].apply(lambda x: 1 if pd.isna(x) else 0)                  

test['len_secondClass'] = test['secondClass'].apply(lambda x: 0 if pd.isna(x) else len(re.split('#',x)))                 



train = train.drop(['firstClass','secondClass'], axis=1)

test = test.drop(['firstClass','secondClass'], axis=1)
 ## 对齐 train、test 数据集

train_labels = train['label']

test_num = test['num']

train,test = train.align(test, join = 'inner', axis = 1)

train['label'] = train_labels

test['num'] = test_num



train = train.drop(['uId','adId','contentId'], axis=1)

test = test.drop(['uId','adId','contentId'], axis=1)



print('Training Features shape: ', train.shape)

print('Testing Features shape: ', test.shape)



def miss_col(data):        #列的缺失个数和缺失率

    col_total = data.isnull().sum().sort_values(ascending=False)#从大到小按顺序排每个特征缺失的个数

    col_percent =(data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)#从大到小按顺序排每个特征缺失率

    

    return col_total, col_percent 

train_miss_rnum, train_miss_rper = miss_col(train)

test_miss_rnum, test_miss_rper = miss_col(test)



X = train.drop(['label'], axis=1)

y = train['label']

X_test = test.drop(['num'], axis=1)



X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1)

############################## 训练模型 ######################################



#### Baseline

# Feature names

features = list(X_train.columns)



# Median imputation of missing values,中值替换缺失值

imputer = Imputer(strategy = 'median')

# 归一化(0,1) 

scaler = MinMaxScaler(feature_range = (0, 1))



# Fit on the training data

imputer.fit(X_train)

# Transform both training and testing data

X_train = imputer.transform(X_train)

X_valid = imputer.transform(X_valid)

X_test = imputer.transform(X_test)



# Repeat with the scaler

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_valid = scaler.transform(X_valid)

X_test = scaler.transform(X_test)



print('Training data shape: ', X_train.shape)

print('Testing data shape: ', X_valid.shape)

print('Testing data shape: ', X_test.shape)



# Make the model with the specified regularization parameter

log_reg = LogisticRegression(C = 0.0001)

# Train on the training data

log_reg.fit(X_train, y_train)



log_reg_pred_train = log_reg.predict_proba(X_train)[:, 1]

log_reg_pred_valid = log_reg.predict_proba(X_valid)[:, 1]

log_reg_pred_test = log_reg.predict_proba(X_test)[:, 1]



baseline_train_auc = roc_auc_score(y_train, log_reg_pred_train)

print('The baseline model scores {:.5f} ROC AUC on the test set.'.format(baseline_train_auc))

baseline_valid_auc = roc_auc_score(y_valid, log_reg_pred_valid)

print('The baseline model scores {:.5f} ROC AUC on the valid set.'.format(baseline_valid_auc))

 # LGBMClassifier

print(train.shape,test.shape)

features = train

test_features = test

n_folds = 5



  # Extract the labels for training

labels = features['label']

test_features_num = test_features['num']

test_features = test_features.drop(columns = ['num'])    

 # Remove the ids and target

features = features.drop(columns = ['label'])

test_features = test_features    

print('Training Data Shape: ', features.shape)

print('Testing Data Shape: ', test_features.shape)

    

  # Extract feature names

feature_names = list(features.columns)    

  # Convert to np arrays

features = np.array(features)

test_features = np.array(test_features)   

  # Create the kfold object

k_fold = KFold(n_splits = n_folds, shuffle = True, random_state = 50)    

  # Empty array for feature importances

feature_importance_values = np.zeros(len(feature_names))    

  # Empty array for test predictions

test_predictions = np.zeros(test_features.shape[0])    

  # Empty array for out of fold validation predictions

out_of_fold_lgb = np.zeros(features.shape[0])    

  # Lists for recording validation and training scores

valid_scores = []

train_scores = []

      

  # Iterate through each fold

for train_indices, valid_indices in k_fold.split(features):

        

    # Training data for the fold

    train_features, train_labels = features[train_indices], labels[train_indices]

    # Validation data for the fold

    valid_features, valid_labels = features[valid_indices], labels[valid_indices]

        

    # Create the model

    model = lgb.LGBMClassifier(n_estimators=10000, objective = 'binary',  # class_weight = 'balanced',

                                learning_rate = 0.05, 

                               reg_alpha = 0.1, reg_lambda = 0.1, 

                               subsample = 0.8, random_state = 50)  #  random_state = 50, n_jobs = -1

        

    # Train the model

    model.fit(train_features, train_labels, eval_metric = 'auc',   #, categorical_feature = cat_indices

              eval_set = [(valid_features, valid_labels), (train_features, train_labels)],

              eval_names = ['valid', 'train'],

              early_stopping_rounds = 100, verbose = 200)

        

    # Record the best iteration

    best_iteration = model.best_iteration_

        

    # Record the feature importances

    feature_importance_values += model.feature_importances_ / k_fold.n_splits

        

    # Make predictions

    test_predictions += model.predict_proba(test_features, num_iteration = best_iteration)[:, 1] / k_fold.n_splits

        

    # Record the out of fold predictions

    out_of_fold_lgb[valid_indices] = model.predict_proba(valid_features, num_iteration = best_iteration)[:, 1]

        

    # Record the best score

    valid_score = model.best_score_['valid']['auc']

    train_score = model.best_score_['train']['auc']

        

    valid_scores.append(valid_score)

    train_scores.append(train_score)

        

    # Clean up memory

    gc.enable()

    del model, train_features, valid_features

    gc.collect()

        

  # Make the submission dataframe

submission_lgb = pd.DataFrame({'id': test_features_num, 'probability': test_predictions})    

  # Make the feature importance dataframe

feature_importances_lgb = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})    

  # Overall validation score

valid_auc_lgb = roc_auc_score(labels, out_of_fold_lgb)   

  # Add the overall scores to the metrics

valid_scores.append(valid_auc_lgb)

train_scores.append(np.mean(train_scores))   

  # Needed for creating dataframe of validation scores

fold_names = list(range(n_folds))

fold_names.append('overall')    

  # Dataframe of validation scores

metrics = pd.DataFrame({'fold': fold_names,

                        'train': train_scores,

                        'valid': valid_scores}) 



fi = feature_importances_lgb.sort_index(by='importance',ascending=False)

print('Baseline metrics')

print(metrics)

# Submission dataframe

submission_lgb['probability']  = submission_lgb['probability'].apply(lambda x: round(x,6))    

submission_lgb.to_csv('submission_lgb.csv', index = False)

print('feature_importances_lgb',fi)