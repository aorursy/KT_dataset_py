# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import warnings

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import time

import datetime



from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder

from sklearn.metrics import roc_auc_score

import lightgbm as lgb

import xgboost as xgb

from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, GroupKFold, GridSearchCV, train_test_split, TimeSeriesSplit

from sklearn import metrics



import gc

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory





for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#import train datasets

train_transaction = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv', index_col='TransactionID')

train_identity = pd.read_csv('../input/ieee-fraud-detection/train_identity.csv', index_col='TransactionID')

# merge to create one dataset

train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)

#remove individual dataframes

del train_transaction, train_identity



#import test datasets

test_transaction = pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv', index_col='TransactionID')

test_identity = pd.read_csv('../input/ieee-fraud-detection/test_identity.csv', index_col='TransactionID')

# merge to create one dataset

test = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)

#remove individual dataframes

del test_transaction, test_identity
train.head()

train.shape #(590540, 433)
#basic assessment of nulls

na_columns = train.isna().sum()

#print(na_columns[na_columns==0]) #19

print(na_columns[na_columns>0]/train.shape[0]) #414
all_columns = train.columns

numericCols = train._get_numeric_data().columns #402

categoricalCols = list(set(all_columns) - set(numericCols)) #31
len(numericCols)
one_value_cols = [col for col in train.columns if train[col].nunique() <= 1] #none

one_value_cols_test = [col for col in test.columns if test[col].nunique() <= 1] #V107

one_value_cols == one_value_cols_test
plt.hist(train['TransactionDT'], label='train');

plt.hist(test['TransactionDT'], label='test');

plt.legend();

plt.title('Distribution of transactiond dates');
def datetime_trans(data,start_date='2017-11-30'):

    startdate=datetime.datetime.strptime(start_date,"%Y-%m-%d")

    data['TransactionDT']=data['TransactionDT'].fillna(data['TransactionDT'].mean())

    data['date']=data['TransactionDT'].apply(lambda x : datetime.timedelta(seconds=x)+startdate)

    data['weekday']=data['date'].apply(lambda x :x.weekday())

    data['month']=(data['date'].dt.year-2017)*12+train['date'].dt.month

    data['hour']=data['date'].apply(lambda x :x.hour)

    data['day']=(data['date'].dt.year-2017)*365+train['date'].dt.dayofyear

    data['year_weekday']=data['date'].apply(lambda x : str(x.year)+'_'+str(x.weekday()))

    data['weekday_hour']=data['date'].apply(lambda x :str(x.weekday())+'_'+str(x.hour))

date_col=['weekday','month','day','hour','year_weekday','weekday_hour']

datetime_trans(train)

datetime_trans(test)
def transaction_amount_details(data):

    data['TransactionAmt_to_mean_card1'] = data['TransactionAmt'] / data.groupby(['card1'])['TransactionAmt'].transform('mean')

    data['TransactionAmt_to_mean_card4'] = data['TransactionAmt'] / data.groupby(['card4'])['TransactionAmt'].transform('mean')

    data['TransactionAmt_to_std_card1'] = data['TransactionAmt'] / data.groupby(['card1'])['TransactionAmt'].transform('std')

    data['TransactionAmt_to_std_card4'] = data['TransactionAmt'] / data.groupby(['card4'])['TransactionAmt'].transform('std')

    

    

transaction_amount_details(train)

transaction_amount_details(test)
def email_details(data):

    data[['P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3']] = data['P_emaildomain'].str.split('.', expand=True)

    data[['R_emaildomain_1', 'R_emaildomain_2', 'R_emaildomain_3']] = data['R_emaildomain'].str.split('.', expand=True)

    

email_details(train)

email_details(test)





train.head()
def id_and_address(data): 

    data['uid'] = data['card1'].astype(str)+'_'+data['card2'].astype(str)



    data['uid2'] = data['uid'].astype(str)+'_'+data['card3'].astype(str)+'_'+data['card5'].astype(str)



    data['uid3'] = data['uid2'].astype(str)+'_'+data['addr1'].astype(str)+'_'+data['addr2'].astype(str)

    data['uid4'] = data['addr1'].astype(str)+'_'+data['addr2'].astype(str)

    data['D9'] = np.where(data['D9'].isna(),0,1)

    

id_and_address(train)

id_and_address(test)
train.head()
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
train.head()
def id02(data):

    data['id_02_to_mean_card1'] = data['id_02'] / data.groupby(['card1'])['id_02'].transform('mean')

    data['id_02_to_mean_card4'] = data['id_02'] / data.groupby(['card4'])['id_02'].transform('mean')

    data['id_02_to_std_card1'] = data['id_02'] / data.groupby(['card1'])['id_02'].transform('std')

    data['id_02_to_std_card4'] = data['id_02'] / data.groupby(['card4'])['id_02'].transform('std')

    

id02(train)

id02(test)
train.head()
def D15(data):

    data['D15_to_mean_card1'] = data['D15'] / data.groupby(['card1'])['D15'].transform('mean')

    data['D15_to_mean_card4'] = data['D15'] / data.groupby(['card4'])['D15'].transform('mean')

    data['D15_to_std_card1'] = data['D15'] / data.groupby(['card1'])['D15'].transform('std')

    data['D15_to_std_card4'] = data['D15'] / data.groupby(['card4'])['D15'].transform('std')

    data['D15_to_mean_addr1'] = data['D15'] / data.groupby(['addr1'])['D15'].transform('mean')

    data['D15_to_mean_addr2'] = data['D15'] / data.groupby(['addr2'])['D15'].transform('mean')

    data['D15_to_std_addr1'] = data['D15'] / data.groupby(['addr1'])['D15'].transform('std')

    data['D15_to_std_addr2'] = data['D15'] / data.groupby(['addr2'])['D15'].transform('std')

    

D15(train)

D15(test)
train.head()
def screen(data):

    data['screen_width'] = data['id_33'].str.split('x', expand=True)[0]

    data['screen_height'] = data['id_33'].str.split('x', expand=True)[1]



screen(train)

screen(test)
def device(data):

    data['device_name'] = data['DeviceInfo'].str.split('/', expand=True)[0]

    data['device_version'] = data['DeviceInfo'].str.split('/', expand=True)[1]

    

device(train)

device(test)
def browser_OS(data):

    data['OS_id_30'] = data['id_30'].str.split(' ', expand=True)[0]

    data['version_id_30'] = data['id_30'].str.split(' ', expand=True)[1]



    data['browser_id_31'] = data['id_31'].str.split(' ', expand=True)[0]

    data['version_id_31'] = data['id_31'].str.split(' ', expand=True)[1]

    

browser_OS(train)

browser_OS(test)
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
train=train.replace([np.inf,-np.inf],np.nan)

test=test.replace([np.inf,-np.inf],np.nan)
#dropping columns

tr_na_count=train.isnull().sum()/len(train) #nulls

tr_drop_cols=[c for c in train.columns if tr_na_count[c]>0.70] #nulls>85%

tr_big_cols=[c for c in train.columns if train[c].value_counts(normalize=True,dropna=False).values[0]>0.85] #if single value>85%

te_na_count=test.isnull().sum()/len(test) #nulls

te_drop_cols=[c for c in test.columns if te_na_count[c]>0.70] #nulls>85%

te_big_cols=[c for c in test.columns if test[c].value_counts(normalize=True,dropna=False).values[0]>0.85] #if single value>85%

drop_cols=list(set(tr_drop_cols+tr_big_cols+te_drop_cols+te_big_cols))

drop_cols.remove('isFraud')

print(len(drop_cols))

response=train['isFraud']

train.drop(columns=drop_cols+['isFraud'],inplace=True)

test.drop(columns=drop_cols,inplace=True)
train.head()
excess_col=['date','TransactionDT']



train.drop(columns=excess_col,inplace=True)

test.drop(columns=excess_col,inplace=True)
train.head()
test.head()
na_count_post=train.isna().sum()

print(na_count_post[na_count_post>0]/train.shape[0])
train.shape #(590540, 220)

test.shape #(506691, 220)
numerical_cols = train.select_dtypes(exclude = 'object').columns

categorical_cols = train.select_dtypes(include = 'object').columns
numerical_cols
categorical_cols=categorical_cols
train_ip=train.copy()

test_ip=test.copy()
train_ip.head()
for col in numerical_cols:

    train_ip[col] = train_ip[col].fillna(value=train_ip[col].mean(skipna=True))

    test_ip[col] = test_ip[col].fillna(value=test_ip[col].mean(skipna=True))
train_ip.head(10)
for col in categorical_cols:

    train_ip[col] = train_ip[col].fillna(train_ip[col].mode()[0])

    test_ip[col] = test_ip[col].fillna(test_ip[col].mode()[0])
train_ip.head(20)
na_count_post2=train_ip.isna().sum()

print(na_count_post[na_count_post2>0]/train_ip.shape[0])
def labelencoder(train,test,col):

    cod=list(train[col].values)+list(test[col].values)

    le=LabelEncoder().fit(cod)

    train[col]=le.transform(train[col])

    test[col]=le.transform(test[col])

    

for c in categorical_cols:

    labelencoder(train_ip,test_ip,c)
train_ip.head()
train_ip.shape
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



NFOLDS = 5

folds = KFold(n_splits=NFOLDS)



columns = train_ip.columns

splits = folds.split(train_ip, response)

y_preds = np.zeros(test_ip.shape[0])

y_oof = np.zeros(train_ip.shape[0])

score = 0



feature_importances = pd.DataFrame()

feature_importances['feature'] = columns

  

for fold_n, (train_index, valid_index) in enumerate(splits):

    X_train, X_valid = train_ip[columns].iloc[train_index], train_ip[columns].iloc[valid_index]

    y_train, y_valid = response.iloc[train_index], response.iloc[valid_index]

    

    dtrain = lgb.Dataset(X_train, label=y_train)

    dvalid = lgb.Dataset(X_valid, label=y_valid)



    clf = lgb.train(params, dtrain, 10000, valid_sets = [dtrain, dvalid], verbose_eval=200, early_stopping_rounds=300)

    

    feature_importances[f'fold_{fold_n + 1}'] = clf.feature_importance()

    

    y_pred_valid = clf.predict(X_valid)

    y_oof[valid_index] = y_pred_valid

    print(f"Fold {fold_n + 1} | AUC: {roc_auc_score(y_valid, y_pred_valid)}")

    

    score += roc_auc_score(y_valid, y_pred_valid) / NFOLDS

    y_preds += clf.predict(test_ip) / NFOLDS

    

    del X_train, X_valid, y_train, y_valid

    gc.collect()

    

print(f"\nMean AUC = {score}")
sub = pd.read_csv(f'../input/ieee-fraud-detection/sample_submission.csv')
sub['isFraud']= y_preds

sub.to_csv('submission.csv', index=False)