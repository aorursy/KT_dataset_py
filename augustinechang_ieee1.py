# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import multiprocessing

import warnings

import matplotlib.pyplot as plt

import seaborn as sns

import lightgbm as lgb

import gc

from time import time

import datetime

from tqdm import tqdm_notebook

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import KFold,StratifiedKFold

from sklearn.metrics import roc_auc_score

warnings.simplefilter('ignore')

sns.set()

%matplotlib inline

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)
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
%%time

warnings.simplefilter('ignore')

files = ['../input/ieee-fraud-detection/test_identity.csv', 

         '../input/ieee-fraud-detection/test_transaction.csv',

         '../input/ieee-fraud-detection/train_identity.csv',

         '../input/ieee-fraud-detection/train_transaction.csv',

         '../input/ieee-fraud-detection/sample_submission.csv']



def load_data(file):

    return reduce_mem_usage(pd.read_csv(file))



with multiprocessing.Pool() as pool:

    test_identity, test_transaction, train_identity, train_transaction, sample_submission = pool.map(load_data, files)
base_columns = list(train_transaction) + list(train_identity)
train_transaction.head()
########################### D9 and TransactionDT

# Let's add temporary "time variables" for aggregations

# and add normal "time variables"



# Also, seems that D9 column is an hour

# and it is the same as df['DT'].dt.hour

START_DATE = datetime.datetime.strptime('2017-11-30', '%Y-%m-%d')

for df in [train_transaction, test_transaction]:

    # Temporary

    df['DT'] = df['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds = x)))

    df['DT_M'] = (df['DT'].dt.year-2017)*12 + df['DT'].dt.month

    df['DT_W'] = (df['DT'].dt.year-2017)*52 + df['DT'].dt.weekofyear

    df['DT_D'] = (df['DT'].dt.year-2017)*365 + df['DT'].dt.dayofyear

    

    df['DT_hour'] = df['DT'].dt.hour

    df['DT_day_week'] = df['DT'].dt.dayofweek

    df['DT_day'] = df['DT'].dt.day

    

    # D9 column

    df['D9'] = np.where(df['D9'].isna(),0,1)


plt.hist(train_transaction['DT'], label='train');

plt.hist(test_transaction['DT'], label='test');

plt.legend();

plt.title('Distribution of transactiond dates');
########################### ProductCD and M4 Target mean

for col in ['ProductCD','M4']:

    temp_dict = train_transaction.groupby([col])['isFraud'].agg(['mean']).reset_index().rename(

                                                        columns={'mean': col+'_target_mean'})

    temp_dict.index = temp_dict[col].values

    temp_dict = temp_dict[col+'_target_mean'].to_dict()



    train_transaction[col+'_target_mean'] = train_transaction[col].map(temp_dict)

    test_transaction[col+'_target_mean']  = test_transaction[col].map(temp_dict)
train_transaction.head()
########################### Reset values for "noise" card1

i_cols = ['card1']



for col in i_cols: 

    valid_card = pd.concat([train_transaction[[col]], test_transaction[[col]]])

    valid_card = valid_card[col].value_counts()

    valid_card = valid_card[valid_card>2]

    valid_card = list(valid_card.index)



    train_transaction[col] = np.where(train_transaction[col].isin(test_transaction[col]), train_transaction[col], np.nan)

    test_transaction[col]  = np.where(test_transaction[col].isin(train_transaction[col]), test_transaction[col], np.nan)



    train_transaction[col] = np.where(train_transaction[col].isin(valid_card), train_transaction[col], np.nan)

    test_transaction[col]  = np.where(test_transaction[col].isin(valid_card), test_transaction[col], np.nan)
train_transaction.head()
########################### TransactionAmt



# Let's add some kind of client uID based on cardID ad addr columns

# The value will be very specific for each client so we need to remove it

# from final feature. But we can use it for aggregations.

train_transaction['uid'] = train_transaction['card1'].astype(str)+'_'+train_transaction['card2'].astype(str)

test_transaction['uid'] = test_transaction['card1'].astype(str)+'_'+test_transaction['card2'].astype(str)



train_transaction['uid2'] = train_transaction['uid'].astype(str)+'_'+train_transaction['card3'].astype(str)+'_'+train_transaction['card5'].astype(str)

test_transaction['uid2'] = test_transaction['uid'].astype(str)+'_'+test_transaction['card3'].astype(str)+'_'+test_transaction['card5'].astype(str)



train_transaction['uid3'] = train_transaction['uid2'].astype(str)+'_'+train_transaction['addr1'].astype(str)+'_'+train_transaction['addr2'].astype(str)

test_transaction['uid3'] = test_transaction['uid2'].astype(str)+'_'+test_transaction['addr1'].astype(str)+'_'+test_transaction['addr2'].astype(str)



# Check if the Transaction Amount is common or not (we can use freq encoding here)

# In our dialog with a model we are telling to trust or not to these values   

train_transaction['TransactionAmt_check'] = np.where(train_transaction['TransactionAmt'].isin(test_transaction['TransactionAmt']), 1, 0)

test_transaction['TransactionAmt_check']  = np.where(test_transaction['TransactionAmt'].isin(train_transaction['TransactionAmt']), 1, 0)



# For our model current TransactionAmt is a noise

# https://www.kaggle.com/kyakovlev/ieee-check-noise

# (even if features importances are telling contrariwise)

# There are many unique values and model doesn't generalize well

# Lets do some aggregations

i_cols = ['card1','card2','card3','card5','uid','uid2','uid3']



for col in i_cols:

    for agg_type in ['mean','std']:

        new_col_name = col+'_TransactionAmt_'+agg_type

        temp_df = pd.concat([train_transaction[[col, 'TransactionAmt']], test_transaction[[col,'TransactionAmt']]])

        #temp_df['TransactionAmt'] = temp_df['TransactionAmt'].astype(int)

        temp_df = temp_df.groupby([col])['TransactionAmt'].agg([agg_type]).reset_index().rename(

                                                columns={agg_type: new_col_name})

        

        temp_df.index = list(temp_df[col])

        temp_df = temp_df[new_col_name].to_dict()   

    

        train_transaction[new_col_name] = train_transaction[col].map(temp_df)

        test_transaction[new_col_name]  = test_transaction[col].map(temp_df)

           

# Small "hack" to transform distribution 

# (doesn't affect auc much, but I like it more)

# please see how distribution transformation can boost your score 

# (not our case but related)

# https://scikit-learn.org/stable/auto_examples/compose/plot_transformed_target.html

train_transaction['TransactionAmt'] = np.log1p(train_transaction['TransactionAmt'])

test_transaction['TransactionAmt'] = np.log1p(test_transaction['TransactionAmt'])    
train_transaction.head()
########################### 'P_emaildomain' - 'R_emaildomain'

p = 'P_emaildomain'

r = 'R_emaildomain'

uknown = 'email_not_provided'



for df in [train_transaction, test_transaction]:

    df[p] = df[p].fillna(uknown)

    df[r] = df[r].fillna(uknown)

    

    # Check if P_emaildomain matches R_emaildomain

    df['email_check'] = np.where((df[p]==df[r])&(df[p]!=uknown),1,0)



    df[p+'_prefix'] = df[p].apply(lambda x: x.split('.')[0])

    df[r+'_prefix'] = df[r].apply(lambda x: x.split('.')[0])



## Local test doesn't show any boost here, 

## but I think it's a good option for model stability 



## Also, we will do frequency encoding later
train_transaction.head()
########################### M columns (except M4)

# All these columns are binary encoded 1/0

# We can have some features from it

i_cols = ['M1','M2','M3','M5','M6','M7','M8','M9']



for df in [train_transaction, test_transaction]:

    df['M_sum'] = df[i_cols].sum(axis=1).astype(np.int8)

    df['M_na'] = df[i_cols].isna().sum(axis=1).astype(np.int8)
train_identity.head()
########################### Device info

for df in [train_identity, test_identity]:

    ########################### Device info

    df['DeviceInfo'] = df['DeviceInfo'].fillna('unknown_device').str.lower()

    df['DeviceInfo_device'] = df['DeviceInfo'].apply(lambda x: ''.join([i for i in x if i.isalpha()]))

    df['DeviceInfo_version'] = df['DeviceInfo'].apply(lambda x: ''.join([i for i in x if i.isnumeric()]))

    

    ########################### Device info 2

    df['id_30'] = df['id_30'].fillna('unknown_device').str.lower()

    df['id_30_device'] = df['id_30'].apply(lambda x: ''.join([i for i in x if i.isalpha()]))

    df['id_30_version'] = df['id_30'].apply(lambda x: ''.join([i for i in x if i.isnumeric()]))

    

    ########################### Browser

    df['id_31'] = df['id_31'].fillna('unknown_device').str.lower()

    df['id_31_device'] = df['id_31'].apply(lambda x: ''.join([i for i in x if i.isalpha()]))
train_identity.head()
########################### Merge Identity columns

temp_df = train_transaction[['TransactionID']]

temp_df = temp_df.merge(train_identity, on=['TransactionID'], how='left')

del temp_df['TransactionID']

train_transaction = pd.concat([train_transaction,temp_df], axis=1)

    

temp_df = test_transaction[['TransactionID']]

temp_df = temp_df.merge(test_identity, on=['TransactionID'], how='left')

del temp_df['TransactionID']

test_transaction = pd.concat([test_transaction,temp_df], axis=1)
train_transaction.head()
train_transaction['card1'].value_counts(dropna=False).to_dict()
########################### Freq encoding

i_cols = ['card1','card2','card3','card5',

          'C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14',

          'D1','D2','D3','D4','D5','D6','D7','D8',

          'addr1','addr2',

          'dist1','dist2',

          'P_emaildomain', 'R_emaildomain',

          'DeviceInfo','DeviceInfo_device','DeviceInfo_version',

          'id_30','id_30_device','id_30_version',

          'id_31_device',

          'id_33',

          'uid','uid2','uid3',

         ]



for col in i_cols:

    temp_df = pd.concat([train_transaction[[col]], test_transaction[[col]]])

    fq_encode = temp_df[col].value_counts(dropna=False).to_dict()   

    train_transaction[col+'_fq_enc'] = train_transaction[col].map(fq_encode)

    test_transaction[col+'_fq_enc']  = test_transaction[col].map(fq_encode)





for col in ['DT_M','DT_W','DT_D']:

    temp_df = pd.concat([train_transaction[[col]], test_transaction[[col]]])

    fq_encode = temp_df[col].value_counts().to_dict()

            

    train_transaction[col+'_total'] = train_transaction[col].map(fq_encode)

    test_transaction[col+'_total']  = test_transaction[col].map(fq_encode)

        



periods = ['DT_M','DT_W','DT_D']

i_cols = ['uid']

for period in periods:

    for col in i_cols:

        new_column = col + '_' + period

            

        temp_df = pd.concat([train_transaction[[col,period]], test_transaction[[col,period]]])

        temp_df[new_column] = temp_df[col].astype(str) + '_' + (temp_df[period]).astype(str)

        fq_encode = temp_df[new_column].value_counts().to_dict()

            

        train_transaction[new_column] = (train_transaction[col].astype(str) + '_' + train_transaction[period].astype(str)).map(fq_encode)

        test_transaction[new_column]  = (test_transaction[col].astype(str) + '_' + test_transaction[period].astype(str)).map(fq_encode)

        

        train_transaction[new_column] /= train_transaction[period+'_total']

        test_transaction[new_column]  /= test_transaction[period+'_total']
########################### Encode Str columns

# For all such columns (probably not)

# we already did frequency encoding (numeric feature)

# so we will use astype('category') here

for col in list(train_transaction):

    if train_transaction[col].dtype=='O':

        print(col)

        train_transaction[col] = train_transaction[col].fillna('unseen_before_label')

        test_transaction[col]  = test_transaction[col].fillna('unseen_before_label')

        

        train_transaction[col] = train_transaction[col].astype(str)

        test_transaction[col] = test_transaction[col].astype(str)

        

        le = LabelEncoder()

        le.fit(list(train_transaction[col])+list(test_transaction[col]))

        train_transaction[col] = le.transform(train_transaction[col])

        test_transaction[col]  = le.transform(test_transaction[col])

        

        train_transaction[col] = train_transaction[col].astype('category')

        test_transaction[col] = test_transaction[col].astype('category')
########################### Model Features 

## We can use set().difference() but the order matters

## Matters only for deterministic results

## In case of remove() we will not change order

## even if variable will be renamed

## please see this link to see how set is ordered

## https://stackoverflow.com/questions/12165200/order-of-unordered-python-sets

rm_cols = [

    'TransactionID','TransactionDT', # These columns are pure noise right now

    'isFraud',                          # Not target in features))

    'uid','uid2','uid3',             # Our new client uID -> very noisy data

    'bank_type',                     # Victims bank could differ by time

    'DT','DT_M','DT_W','DT_D',       # Temporary Variables

    'DT_hour','DT_day_week','DT_day',

    'DT_D_total','DT_W_total','DT_M_total',

    'id_30','id_31','id_33',

]
########################### Features elimination 

from scipy.stats import ks_2samp

features_check = []

columns_to_check = set(list(train_transaction)).difference(base_columns+rm_cols)

for i in columns_to_check:

    features_check.append(ks_2samp(test_transaction[i], train_transaction[i])[1])



features_check = pd.Series(features_check, index=columns_to_check).sort_values() 

features_discard = list(features_check[features_check==0].index)

print(features_discard)



# We will reset this list for now (use local test drop),

# Good droping will be in other kernels

# with better checking

features_discard = [] 



# Final features list

features_columns = [col for col in list(train_transaction) if col not in rm_cols + features_discard]
SEED = 42
########################### Model params

lgb_params = {

                    'objective':'binary',

                    'boosting_type':'gbdt',

                    'metric':'auc',

                    'n_jobs':-1,

                    'learning_rate':0.01,

                    'num_leaves': 2**8,

                    'max_depth':-1,

                    'tree_learner':'serial',

                    'colsample_bytree': 0.7,

                    'subsample_freq':1,

                    'subsample':0.7,

                    'n_estimators':800,

                    'max_bin':255,

                    'verbose':-1,

                    'seed': SEED,

                    'early_stopping_rounds':100, 

                } 
test_transaction.head()
test_transaction.head()
LOCAL_TEST = False
########################### Model

import lightgbm as lgb



def make_predictions(tr_df, tt_df, features_columns, target, lgb_params, NFOLDS=2):

    folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)



    X,y = tr_df[features_columns], tr_df[target]    

#     P,P_y = tt_df[features_columns], tt_df[target]  

    P = tt_df[features_columns] 



    tt_df = tt_df[['TransactionID']]    

    predictions = np.zeros(len(tt_df))

    

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y)):

        print('Fold:',fold_)

        tr_x, tr_y = X.iloc[trn_idx,:], y[trn_idx]

        vl_x, vl_y = X.iloc[val_idx,:], y[val_idx]

            

        print(len(tr_x),len(vl_x))

        tr_data = lgb.Dataset(tr_x, label=tr_y)



        if LOCAL_TEST:

            vl_data = lgb.Dataset(P, label=P_y) 

        else:

            vl_data = lgb.Dataset(vl_x, label=vl_y)  



        estimator = lgb.train(

            lgb_params,

            tr_data,

            valid_sets = [tr_data, vl_data],

            verbose_eval = 200,

        )   

        

        pp_p = estimator.predict(P)

        predictions += pp_p/NFOLDS



        if LOCAL_TEST:

            feature_imp = pd.DataFrame(sorted(zip(estimator.feature_importance(),X.columns)), columns=['Value','Feature'])

            print(feature_imp)

        

        del tr_x, tr_y, vl_x, vl_y, tr_data, vl_data

        gc.collect()

        

    tt_df['prediction'] = predictions

    

    return tt_df

## -------------------
lgb_params['learning_rate'] = 0.005

lgb_params['n_estimators'] = 1800

lgb_params['early_stopping_rounds'] = 100    

test_predictions = make_predictions(train_transaction, test_transaction, features_columns, 'isFraud', lgb_params, NFOLDS=3)
########################### Export

if not LOCAL_TEST:

    test_predictions['isFraud'] = test_predictions['prediction']

    test_predictions[['TransactionID','isFraud']].to_csv('submission.csv', index=False)