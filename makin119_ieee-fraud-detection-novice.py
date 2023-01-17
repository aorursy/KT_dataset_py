import numpy as np

import pandas as pd

import os



import matplotlib.pyplot as plt

%matplotlib inline

plt.style.use('ggplot')

from tqdm import tqdm

from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder

from sklearn.decomposition import PCA,KernelPCA

from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import SimpleImputer,IterativeImputer

from sklearn.svm import NuSVR, SVR

from sklearn.metrics import mean_absolute_error

pd.options.display.precision = 15

from sklearn.metrics import roc_auc_score

import lightgbm as lgb

import xgboost as xgb

import time

import datetime

from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, GroupKFold, GridSearchCV, train_test_split, TimeSeriesSplit

from sklearn import metrics

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")



import json

import gc





def reduce_mem_usage(df, verbose=True):

    #reduce memory of data uesd

    start_mem = df.memory_usage(deep=True).sum() / 1024**2

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type != object:

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

    end_mem = df.memory_usage(deep=True).sum() / 1024**2

    if verbose: 

        print(f'Memory usage of dataframe is {start_mem} MB --> {end_mem} MB (Decreased by {100 * (start_mem - end_mem) / start_mem})')

    return df



%%time

folder_path = '../input/'

train_identity = pd.read_csv(f'{folder_path}train_identity.csv')

train_transaction = pd.read_csv(f'{folder_path}train_transaction.csv')

test_identity = pd.read_csv(f'{folder_path}test_identity.csv')

test_transaction = pd.read_csv(f'{folder_path}test_transaction.csv')

sub = pd.read_csv(f'{folder_path}sample_submission.csv')

# let's combine the data and work with the whole dataset

train = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')

test = pd.merge(test_transaction, test_identity, on='TransactionID', how='left')

identity_col=list(train_identity.columns)

trasac_col=list(train_transaction.columns)



del train_identity, train_transaction, test_identity, test_transaction

train=reduce_mem_usage(train)

test=reduce_mem_usage(test)

gc.collect()
#3.transactionDT



def datetime_trans(train,start_date='2017-11-30'):

    startdate=datetime.datetime.strptime(start_date,"%Y-%m-%d")

    train['TransactionDT']=train['TransactionDT'].fillna(train['TransactionDT'].mean())

    train['date']=train['TransactionDT'].apply(lambda x : datetime.timedelta(seconds=x)+startdate)

    train['weekday']=train['date'].apply(lambda x :x.weekday())#不适合单独使用

    train['month']=(train['date'].dt.year-2017)*12+train['date'].dt.month

    train['hour']=train['date'].apply(lambda x :x.hour)#可以使用

    train['day']=(train['date'].dt.year-2017)*365+train['date'].dt.dayofyear

    train['year_weekday']=train['date'].apply(lambda x : str(x.year)+'_'+str(x.weekday()))#有一定的偏度，但较为平坦

    train['weekday_hour']=train['date'].apply(lambda x :str(x.weekday())+'_'+str(x.hour))#波动性质较好

date_col=['weekday','month','day','hour','year_weekday','weekday_hour']

datetime_trans(train)

datetime_trans(test)

#it's most important trick ,card1-card3 and card5 seems like some serial number

def addNewFeatures(data): 

    data['uid'] = data['card1'].astype(str)+'_'+data['card2'].astype(str)



    data['uid2'] = data['uid'].astype(str)+'_'+data['card3'].astype(str)+'_'+data['card5'].astype(str)



    data['uid3'] = data['uid2'].astype(str)+'_'+data['addr1'].astype(str)+'_'+data['addr2'].astype(str)

    data['uid4'] = data['addr1'].astype(str)+'_'+data['addr2'].astype(str)

    data['D9'] = np.where(data['D9'].isna(),0,1)

    

    return data



train = addNewFeatures(train)

test = addNewFeatures(test)
agg_cols = ['card1','card2','card3','card5','uid','uid2','uid3','uid4']

def add_agg_col(col_prefix,agg_col,col_suffix='TransactionAmt'):

    if isinstance(agg_col,list):

        temp_df=pd.concat([train[[col_prefix,col_suffix]],test[[col_prefix,col_suffix]]])

        temp_df=temp_df.groupby(col_prefix)[col_suffix].agg(agg_col)

        for c in agg_col:

            new_col=col_prefix+'_'+c+'_'+col_suffix

            train[new_col]=train[col_prefix].map(temp_df[c])#problem is here temp_df.columns

            test[new_col]=test[col_prefix].map(temp_df[c])

    else:

        raise TypeError('agg_col must be List')



for i in agg_cols:

    add_agg_col(i,['mean','std'])

    print(f'{i} for [\'mean\',\'std\'] aggregate is done!')



train['id_02_to_mean_card1'] = train['id_02'] / train.groupby(['card1'])['id_02'].transform('mean')

train['id_02_to_mean_card4'] = train['id_02'] / train.groupby(['card4'])['id_02'].transform('mean')

train['id_02_to_std_card1'] = train['id_02'] / train.groupby(['card1'])['id_02'].transform('std')

train['id_02_to_std_card4'] = train['id_02'] / train.groupby(['card4'])['id_02'].transform('std')



test['id_02_to_mean_card1'] = test['id_02'] / test.groupby(['card1'])['id_02'].transform('mean')

test['id_02_to_mean_card4'] = test['id_02'] / test.groupby(['card4'])['id_02'].transform('mean')

test['id_02_to_std_card1'] = test['id_02'] / test.groupby(['card1'])['id_02'].transform('std')

test['id_02_to_std_card4'] = test['id_02'] / test.groupby(['card4'])['id_02'].transform('std')



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



train=train.replace([np.inf,-np.inf],np.nan)

test=test.replace([np.inf,-np.inf],np.nan)
#P and R emaildomain's np.nan is float type

def p_r_domain(train,test,col):

    train[col]=train[col].fillna('Nan').apply(lambda x : x.split('.')[0])

    test[col]=test[col].fillna('Nan').apply(lambda x : x.split('.')[0])

    

for c in ['P_emaildomain','R_emaildomain']:

    p_r_domain(train,test,c)
#add 'others' mark threshold=0.95 exclude 'na's

def add_others_mark(train,test,categ_col):

    temp_df=pd.concat([train[[categ_col]],test[[categ_col]]])

    series=temp_df[categ_col].value_counts(normalize=True).cumsum()

    others_index=list(series[series>0.95].index)

    if len(others_index)!=0:

        train[categ_col]=train[categ_col].apply(lambda x : 'others' if x in others_index else x)

        test[categ_col]=test[categ_col].apply(lambda x : 'others' if x in others_index else x)

        print(f'{categ_col}:{len(others_index)} of {len(series)} feature values has been replaced to \'others\'')

mail_col=['P_emaildomain', 'R_emaildomain',

          'DeviceInfo',

          'id_30','id_33']       



for c in mail_col:

    add_others_mark(train,test,c)

add_others_mark(train,test,mail_col[3])

#set frequency

freq_cols = ['card1','card2','card3','card5',

          'C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14',

          'D1','D2','D3','D4','D5','D6','D7','D8',

          'addr1','addr2',

          'dist1','dist2',

          'P_emaildomain', 'R_emaildomain',

          'DeviceInfo','DeviceType',

          'id_30','id_33',

          'uid','uid2','uid3','uid4'

         ]+date_col



def set_freq_col(train,test,col):

    prefix='_fq'

    temp_df=pd.concat([train[[col]],test[[col]]])

    fq=temp_df[col].value_counts(dropna=False)

    train[col+prefix]=train[col].map(fq)

    test[col+prefix]=test[col].map(fq)

    

for c in freq_cols:

    set_freq_col(train,test,c)

    



periods = ['month','year_weekday','weekday_hour']

uids = ['uid','uid2','uid3','uid4']

def set_uid_period(train,test,periods,uids):

    for period in periods:

        for col in uids:

            new_column = col + '_' + period



            temp_df = pd.concat([train[[col,period]], test[[col,period]]])

            temp_df[new_column] = temp_df[col].astype(str) + '_' + (temp_df[period]).astype(str)

            fq_encode = temp_df[new_column].value_counts()



            train[new_column] = (train[col].astype(str) + '_' + train[period].astype(str)).map(fq_encode)

            test[new_column]  = (test[col].astype(str) + '_' + test[period].astype(str)).map(fq_encode)



            train[new_column] /= train[period+'_fq']

            test[new_column]  /= test[period+'_fq']

            

set_uid_period(train,test,periods,uids)
#去掉na较多，单值占比较大的列。threshold=0.85

tr_na_count=train.isnull().sum()/len(train)

tr_drop_cols=[c for c in train.columns if tr_na_count[c]>0.85]

tr_big_cols=[c for c in train.columns if train[c].value_counts(normalize=True,dropna=False).values[0]>0.85]

drop_cols=list(set(tr_drop_cols+tr_big_cols))

drop_cols.remove('isFraud')

y_=train['isFraud']

train.drop(columns=drop_cols+['isFraud'],inplace=True)

test.drop(columns=drop_cols,inplace=True)

#15去掉多余的列。

excess_col=['date','TransactionDT','TransactionID']



train.drop(columns=excess_col,inplace=True)

test.drop(columns=excess_col,inplace=True)
#use sklearn.imputer object instead of pd.fillna()



def imputing_na(train,col):

    #use sklearn.imputer object instead of pd.fillna()

    if train[col].dtypes == object:

        imp=SimpleImputer(strategy='constant',fill_value='Nan').fit_transform(train[col].values.reshape(-1,1))

        train[col]=pd.Series(imp[:,0])

    else:

        imp=SimpleImputer(strategy='constant',fill_value=-999).fit_transform(train[col].values.reshape(-1,1))

        train[col]=pd.Series(imp[:,0])

for c in train.columns:

    imputing_na(train,c)

    imputing_na(test,c)

#label encoder categorical columns

numerical_cols = train.select_dtypes(exclude = 'object').columns

categorical_cols = train.select_dtypes(include = 'object').columns



def labelencoder(train,test,col):

    cod=list(train[col].values)+list(test[col].values)

    le=LabelEncoder().fit(cod)

    train[col]=le.transform(train[col])

    test[col]=le.transform(test[col])

    

for c in categorical_cols:

    labelencoder(train,test,c)

    

print(train.shape,test.shape)

#training LGBM

params = {'num_leaves': int((2**10)*0.72),

          'min_child_weight': 0.17,

          'feature_fraction': 0.72,

          'bagging_fraction': 0.72,

          'min_data_in_leaf': 179,

          'objective': 'binary',

          'max_depth': -1,

          'learning_rate': 0.006,

          "boosting_type": "gbdt",

          "bagging_seed": 13,

          "metric": 'auc',

          "verbosity": -1,

          'reg_alpha': 0.3299927210061127,

          'reg_lambda': 0.3885237330340494,

          'random_state': 4,

}
%%time



NFOLDS = 7

folds = KFold(n_splits=NFOLDS)



columns = train.columns

splits = folds.split(train, y_)

y_preds = np.zeros(test.shape[0])

y_oof = np.zeros(train.shape[0])

score = 0



feature_importances = pd.DataFrame()

feature_importances['feature'] = columns

  

for fold_n, (train_index, valid_index) in enumerate(splits):

    X_train, X_valid = train[columns].iloc[train_index], train[columns].iloc[valid_index]

    y_train, y_valid = y_.iloc[train_index], y_.iloc[valid_index]

    

    dtrain = lgb.Dataset(X_train, label=y_train)

    dvalid = lgb.Dataset(X_valid, label=y_valid)



    clf = lgb.train(params, dtrain, 10000, valid_sets = [dtrain, dvalid], verbose_eval=200, early_stopping_rounds=300)

    

    feature_importances[f'fold_{fold_n + 1}'] = clf.feature_importance()

    

    y_pred_valid = clf.predict(X_valid)

    y_oof[valid_index] = y_pred_valid

    print(f"Fold {fold_n + 1} | AUC: {roc_auc_score(y_valid, y_pred_valid)}")

    

    score += roc_auc_score(y_valid, y_pred_valid) / NFOLDS

    y_preds += clf.predict(test) / NFOLDS

    

    del X_train, X_valid, y_train, y_valid

    gc.collect()

    

print(f"\nMean AUC = {score}")


sub['isFraud']= y_preds
sub.to_csv('submission.csv', index=False)

feature_importances['average'] = feature_importances[[f'fold_{fold_n + 1}' for fold_n in range(folds.n_splits)]].mean(axis=1)



plt.figure(figsize=(16, 16))

sns.barplot(data=feature_importances.sort_values(by='average', ascending=False).head(50), x='average', y='feature');

plt.title('50 TOP feature importance over {} folds average'.format(NFOLDS));
lgb.create_tree_digraph(clf)