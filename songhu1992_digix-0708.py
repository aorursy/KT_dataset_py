import numpy as np

import pandas as pd

pd.set_option('max_columns', None)

import matplotlib.pyplot as plt

plt.style.use('ggplot')

from sklearn.model_selection import train_test_split, KFold

from collections import Counter

from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import StandardScaler,MinMaxScaler, Imputer,LabelEncoder



import gc

import re

import plotly.offline as py

py.init_notebook_mode(connected=True)

import xgboost as xgb

import lightgbm as lgb

from sklearn.linear_model import LogisticRegression
train = pd.read_csv('../input/ctr-train-200/train_200.csv')

for col in train:

    if train[col].dtype == 'float64':

        train[col] = train[col].astype(np.float16)

for col in train:

    if train[col].dtype == 'int64':

        train[col] = train[col].astype(np.int16)



#train_0 = train[ train['label'] == 0 ]

#train_1 = train[ train['label'] == 1 ]

#train_0 = train_0.sample(n = 4800000 , random_state = 42).reset_index(drop=True)

#train=pd.concat([train_0,train_1]).reset_index(drop=True)

#train = train.sample(frac = 1, random_state = 32).reset_index(drop=True)



test = pd.read_csv('../input/ctr-test-200/test_100.csv') #,nrows=80000

for col in test:

    if test[col].dtype == 'float64':

        test[col] = test[col].astype(np.float16)

for col in test:

    if test[col].dtype == 'int64':

        test[col] = test[col].astype(np.int16)



print(train['label'].value_counts())

train['label'].astype(int).plot.hist();



def miss_col(data):        #列的缺失个数和缺失率

    col_total = data.isnull().sum().sort_values(ascending=False)#从大到小按顺序排每个特征缺失的个数

    col_percent =(data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)#从大到小按顺序排每个特征缺失率    

    return col_total, col_percent 

train_miss_rnum, train_miss_rper = miss_col(train)

test_miss_rnum, test_miss_rper = miss_col(test)
 # 提取aid, uid的全局统计特征

train_labels = train['label']

test_num = test['num']

train_len = len(train)

test_len = len(test)

    

df = pd.concat([train.drop(['label'], axis=1), test.drop(['num'], axis=1)], axis=0,sort=False)

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



# df['uId_count'] = df.groupby('uId')['adId'].transform('count')

fea_columns = ['uId_adId_nunique', 'uId_siteId_nunique', 'uId_slotId_nunique',

                   'uId_netType_nunique','uId_contentId_nunique', 'adId_uId_nunique']





for col in fea_columns:

    bins = []

    for percent in [0, 20, 35, 50, 65, 85, 100]:

        bins.append(np.percentile(df[col], percent))

    df[col] =  np.digitize(df[col], bins, right=True)



# count_bins = [10,20,30,40,50,60,70,80,90,100,120,140,160,180,200,300,400]

# df['uId_count'] =  np.digitize(df['uId_count'], count_bins, right=True)



# 衍生时间特征

df['operTime_1'] = pd.to_datetime(df['operTime'])

 

  ## getattr 返回属性值年份、月份、周几

def process_date(df):

    date_parts = [ 'hour'] # 'month','weekday',  'weekofyear','day', 

    for part in date_parts:

        part_col = 'operTime' + "_" + part

        df[part_col] = getattr(df['operTime_1'].dt, part).astype(np.int32)   

    return df



df = process_date(df)



df = df.drop(['operTime'], axis=1)

df = df.drop(['operTime_1'], axis=1)



 ## siteId

df['siteId'].value_counts()



df = df.join(pd.get_dummies(df['siteId'],prefix='siteId_'))

 ## slotId

df['slotId'].value_counts()

list_of_slotId = list(df['slotId'].values)

Counter( list_of_slotId ).most_common(10)

top_count_slotId = [m[0] for m in Counter( list_of_slotId  ).most_common(10)]

for g in top_count_slotId:

    df['slotId_' + str(g)] = df['slotId'].apply(lambda x: 1 if g == x else 0)



 ## netType

df['netType'].value_counts()

df = df.join(pd.get_dummies(df['netType'],prefix='netType_'))

# df = df.drop(['siteId','slotId','netType'], axis=1)
# user_info



df['user_info_na']=df[['age','gender','city','province','phoneType','carrier']].count(axis=1).astype(np.int8) 



df['gender']  = df['gender'].apply(lambda x: 0 if pd.isna(x) else x).astype(np.int8) 

df['age']  = df['age'].apply(lambda x: 7 if pd.isna(x) else x).astype(np.int8) 



df['age'].value_counts()

df['gender'].value_counts()



df['city'].value_counts()

df['city']  = df['city'].apply(lambda x: 0 if pd.isna(x) else x) 

list_of_city= list(df['city'].values)

Counter( list_of_city ).most_common(10)

top_count_city= [m[0] for m in Counter(list_of_city).most_common(10)]

for g in top_count_city:

    df['city_' + str(g)] = df['city'].apply(lambda x: 1 if g == x else 0).astype(np.int8) 



df['province'].value_counts()

df['province']  = df['province'].apply(lambda x: 0 if pd.isna(x) else x) 

list_of_province= list(df['province'].values)

Counter( list_of_province ).most_common(15)

top_count_province= [m[0] for m in Counter(list_of_province).most_common(10)]

for g in top_count_province:

    df['province_' + str(g)] = df['province'].apply(lambda x: 1 if g == x else 0).astype(np.int8) 



df['phoneType'].value_counts()

df['phoneType']  = df['phoneType'].apply(lambda x: 0 if pd.isna(x) else x) 

list_of_phoneType= list(df['phoneType'].values)

Counter( list_of_phoneType ).most_common(15)

top_count_phoneType= [m[0] for m in Counter(list_of_province).most_common(10)]

for g in top_count_phoneType:

    df['phoneType_' + str(g)] = df['phoneType'].apply(lambda x: 1 if g == x else 0).astype(np.int8) 

    

df['carrier']  = df['carrier'].apply(lambda x: 0 if pd.isna(x) else x) 

df['carrier'].value_counts()

df = df.join(pd.get_dummies(df['carrier'],prefix='carrier_'))



#df = df.drop(['city','province','phoneType','carrier'], axis=1)
# ad_info



df['billId'].value_counts()

le = LabelEncoder()

le.fit(df['billId'])

df['billId']= le.transform(df['billId'])



df['primId'].value_counts()

df['primId']  = df['primId'].apply(lambda x: 0 if pd.isna(x) else x) 

df['primId'].value_counts()

list_of_primId = list(df['primId'].values)

Counter( list_of_primId ).most_common(10)

top_count_primId = [m[0] for m in Counter( list_of_primId  ).most_common(10)]

for g in top_count_primId:

    df['primId_' + str(g)] = df['primId'].apply(lambda x: 1 if g == x else 0).astype(np.int8) 



df['creativeType'].value_counts()

df = df.join(pd.get_dummies(df['creativeType'],prefix='creativeType_'))



df['intertype'].value_counts()

df = df.join(pd.get_dummies(df['intertype'],prefix='intertype_'))



df['spreadAppId'].value_counts()

df['spreadAppId_na']  = df['spreadAppId'].apply(lambda x: 0 if pd.isna(x) else 1) 

df['spreadAppId']  = df['spreadAppId'].apply(lambda x: 0 if pd.isna(x) else x) 



# list_of_spreadAppId = list(df['spreadAppId'].values)

# top_count_spreadAppId = [m[0] for m in Counter( list_of_spreadAppId  ).most_common(6)]

# for g in top_count_spreadAppId:

#     df['spreadAppId_' + str(g)] = df['spreadAppId'].apply(lambda x: 1 if g == x else 0).astype(np.int8) 



# df = df.drop(['primId','creativeType','intertype','spreadAppId'], axis=1)
  ## firstClass

df['firstClass'].value_counts()



df['firstClass']  = df['firstClass'].apply(lambda x: 'nan' if pd.isna(x) else x)  

list_of_firstClass = list(df['firstClass'].values)

Counter( list_of_firstClass ).most_common(6)

top_count_firstClass = [m[0] for m in Counter( list_of_firstClass  ).most_common(6)]

for g in top_count_firstClass:

    df['firstClass_' + str(g)] = df['firstClass'].apply(lambda x: 1 if g in str(x) else 0).astype(np.int8) 



df['secondClass_na']  = df['secondClass'].apply(lambda x: 1 if pd.isna(x) else 0).astype(np.int8)                  

df['len_secondClass'] = df['secondClass'].apply(lambda x: 0 if pd.isna(x) else len(re.split('#',x))).astype(np.int8)  



df = df.drop(['firstClass','secondClass'], axis=1)



for col in df:

    if df[col].dtype == 'int64':

        df[col] = df[col].astype(np.int32)    

for col in df:

    if df[col].dtype == 'float64':

        df[col] = df[col].astype(np.float32)  

 ## 对齐 train、test 数据集

train = df.iloc[:train_len]

test = df.iloc[train_len: train_len+test_len].reset_index(drop=True)

df = pd.DataFrame()

for col in train.columns:

    if train[col].nunique() == 1:

        print(col)

        train = train.drop([col], axis=1)

        test = test.drop([col], axis=1)

        

train,test = train.align(test, join = 'inner', axis = 1)

train['label'] = train_labels

test['num'] = test_num



train = train.drop(['uId'], axis=1)

test = test.drop(['uId'], axis=1)



print('Training Features shape: ', train.shape)

print('Testing Features shape: ', test.shape)



def miss_col(data):        #列的缺失个数和缺失率

    col_total = data.isnull().sum().sort_values(ascending=False)#从大到小按顺序排每个特征缺失的个数

    col_percent =(data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)#从大到小按顺序排每个特征缺失率    

    return col_total, col_percent 

train_miss_rnum, train_miss_rper = miss_col(train)

test_miss_rnum, test_miss_rper = miss_col(test)

def miss_col(data):        #列的缺失个数和缺失率

    col_total = data.isnull().sum().sort_values(ascending=False)#从大到小按顺序排每个特征缺失的个数

    col_percent =(data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)#从大到小按顺序排每个特征缺失率    

    return col_total, col_percent 

train_miss_rnum, train_miss_rper = miss_col(train)

test_miss_rnum, test_miss_rper = miss_col(test)
train.dtypes
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

    model = lgb.LGBMClassifier(n_estimators=2000, objective = 'binary',  # class_weight = 'balanced',

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