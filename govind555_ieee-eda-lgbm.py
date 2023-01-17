# General imports

import numpy as np

import pandas as pd

import os, warnings, datetime, math

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.preprocessing import LabelEncoder



warnings.filterwarnings('ignore')
########################### Helpers

#################################################################################

## -------------------

## Memory Reducer

# :df pandas dataframe to reduce size             # type: pd.DataFrame()

# :verbose                                        # type: bool

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

## -------------------
########################### DATA LOAD

#################################################################################

print('Load Data')

train_df = pd.read_csv('../input/train_transaction.csv')

test_df = pd.read_csv('../input/test_transaction.csv')

test_df['isFraud'] = 0



train_identity = pd.read_csv('../input/train_identity.csv')

test_identity = pd.read_csv('../input/test_identity.csv')
########################### Base check

#################################################################################



for df in [train_df, test_df, train_identity, test_identity]:

    original = df.copy()

    df = reduce_mem_usage(df)



    for col in list(df):

        if df[col].dtype!='O':

            if (df[col]-original[col]).sum()!=0:

                df[col] = original[col]

                #print('Bad transformation', col)
train_df.columns
train_identity.head()
temp_df = pd.concat([train_df, test_df])

percent_missing = (temp_df.isnull().mean() * 100) 

print(percent_missing.sort_values(ascending=False))
### Removing columns greater than a certain threshold

#thresh = 90

#reduced_col = (percent_missing[percent_missing<thresh])

#reduced_col.sort_values(ascending=False)
# numnerical columns

num_cols = train_df._get_numeric_data().columns

print('Total number of numerical columns are ', len(num_cols))

print(num_cols)

print('-------------------------------------------------------------------------------')

# categorical columns

cat_cols = set(train_df.columns) - set(num_cols)

print('Total number of categorical columns are', len(cat_cols))

print(cat_cols)
# Plotting categorical features

def make_categorical_plots(Vs):

    col = 4

    row = len(Vs)//col+1

    fig = plt.figure(figsize=(20,row*5))

    for i,v in enumerate(Vs):

        ax = plt.subplot(row,col,i+1)

        g1 = sns.barplot(x=v, y="isFraud", data=train_df, ax = ax)

        g1.legend()

        plt.title(v+" barplot w.r.t target", fontsize=16)

        g1.set_xlabel(v+ " values", fontsize=16)

        g1.set_ylabel("Probability", fontsize=16)

        #plt.title('Column: '+ str(v), fontsize=16)

        plt.subplots_adjust(hspace = 0.5)

    plt.show()
# Plotting desity plot for numerical features

def make_density_plots(Vs):

    col = 2

    row = len(Vs)//col+1

    fig = plt.figure(figsize=(20,row*5))

    for i,v in enumerate(Vs):

        ax = plt.subplot(row,col,i+1)

        g1 = sns.distplot(train_df[train_df['isFraud'] == 1][v].dropna(), label='Fraud',

                          ax=ax )

        g1 = sns.distplot(train_df[train_df['isFraud'] == 0][v].dropna(), label='NoFraud',

                              ax=ax)

        g1.legend()

        plt.title(v+" values distribution w.r.t target", fontsize=16)

        g1.set_xlabel(v+ " values", fontsize=16)

        g1.set_ylabel("Probability", fontsize=16)

        #plt.title('Column: '+ str(v), fontsize=16)

        plt.subplots_adjust(hspace = 0.5)

    plt.show()
# Function to plot histogram

def make_histogram_plots(Vs):

    col = 4

    row = len(Vs)//4+1

    plt.figure(figsize=(20,row*5))

    for i,v in enumerate(Vs):

        #print(v)

        plt.subplot(row,col,i+1)

        plt.title('Column: '+ str(v))

        h = plt.hist(train_df[v],bins=100)

        if len(h[0])>1: plt.ylim((0,np.sort(h[0])[-2]))

    plt.show()
import re
total_col = train_df.columns 

# Finding all columns starting with 'card'

r = re.compile("^card")

card_list = list(filter(r.match, total_col)) # Read Note

print(card_list)
make_density_plots(['card1', 'card2', 'card3','card5'])
# Finding all columns starting with 'addr'

r = re.compile("^addr")

addr_list = list(filter(r.match, total_col)) # Read Note

print(addr_list)
make_density_plots(addr_list)
# Finding all columns starting with 'V'

r = re.compile("^V")

V_list = list(filter(r.match, total_col)) # Read Note

print(V_list)
# Plotting first 10 values

make_density_plots(['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10'])
# col_corr = set() # Set of all the names of deleted columns

corr_matrix = train_df[V_list].corr()
####### Removing multi-correlated columns and retaining only that column which has high unique value, 

#### This conveys that it has high variance, and may be has high impact on the model

#### Here, first we are making correlation matrix and then finding the correlations of columns with other columns

#### which are highly correlated. Then removing all other columns except for the columns which has highest unique value. 

col_corr = set()

threshold = 0.80

for i in range(len(corr_matrix.columns)):

    high_corr = []

    high_corr.append(corr_matrix.columns[i])

    for j in range(i):

        mx = 0

        vx = corr_matrix.columns[i]

        #print('vx', vx)

        if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):

            high_corr.append(corr_matrix.columns[j])

            

    if len(high_corr)>1:

        #print('high_corr',high_corr )

        for col in high_corr:

            n = train_df[col].nunique()

            if n>mx:

                mx = n

                vx = col

        #print('vx', vx)        

        high_corr.remove(str(vx))

        col_corr.update(set(high_corr))

        #print('col_corr', col_corr)
print(list(col_corr))
# V_list_ = [x for x in V_list if x not in col_corr]

# print("Length of new V_list is", len(V_list_))

# print(V_list_)
# with threshold value of .80

# This is the output after finding the features which are highly correlated. 

# You can find how to get these columns 3 cells above

V_list_ = ['V1', 'V3', 'V5', 'V7', 'V9', 'V13', 'V20', 'V24', 'V26', 'V28', 'V30', 'V36', 'V38', 'V45',

           'V47', 'V54', 'V55', 'V56', 'V62', 'V67', 'V76', 'V78', 'V83', 'V87', 'V88', 'V107', 'V109',

           'V110', 'V113', 'V115', 'V116', 'V119', 'V121', 'V122', 'V125', 'V138', 'V140', 'V142', 

           'V147', 'V158', 'V160', 'V162', 'V166', 'V169', 'V174', 'V185', 'V198', 'V201', 'V209', 

           'V210', 'V220', 'V221', 'V223', 'V239', 'V240', 'V241', 'V251', 'V252', 'V260', 'V262',

           'V267', 'V271', 'V281', 'V282', 'V283', 'V286', 'V289', 'V291', 'V301', 'V303', 'V305', 

           'V307', 'V310', 'V325', 'V339']
#make_density_plots(['V_list_'])
# Plotting values

make_density_plots(['TransactionAmt'])
# Finding all columns starting with 'D'

r = re.compile("^D")

D_list = list(filter(r.match, total_col)) # Read Note

print(D_list)
make_density_plots(D_list)
# Finding all columns starting with 'M'

r = re.compile("^M")

M_list = list(filter(r.match, total_col)) # Read Note

print(M_list)
make_categorical_plots(['M1','M2','M3','M4','M5','M6','M7','M8','M9'])
make_categorical_plots(['addr1', 'addr2'])
train_identity.head()
train_identity.columns
temp_identity_df = pd.concat([train_identity, test_identity])

percent_missing = (temp_identity_df.isnull().mean() * 100) 

print(percent_missing.sort_values(ascending=False))
# numnerical columns

num_identity_cols = train_identity._get_numeric_data().columns

print('Total number of numerical columns are ', len(num_identity_cols))

print(num_identity_cols)

print('-------------------------------------------------------------------------------')

# categorical columns

cat_identity_cols = set(train_identity.columns) - set(num_identity_cols)

print('Total number of categorical columns are', len(cat_identity_cols))

print(cat_identity_cols)
train_df['TransactionAmt'] / train_df.groupby(['card1'])['TransactionAmt'].transform('mean')
train_df.head()
# Find the groups

#train_df.groupby(['card1']).groups
sns.countplot(x="ProductCD", data=train_df)
sns.countplot(x="M4", data=train_df)
train_df['card1'].nunique()
# Freq encoding

i_cols = ['card1','card2','M4','D9',

          'addr1','addr2','dist1','dist2',

          'P_emaildomain', 'R_emaildomain'

         ]



for col in i_cols:

    temp_df = pd.concat([train_df[[col]], test_df[[col]]])

    valid_card = temp_df[col].value_counts().to_dict()   

    train_df[col+'_fq_enc'] = train_df[col].map(valid_card)

    test_df[col+'_fq_enc']  = test_df[col].map(valid_card)
(train_df['card1'].astype(str)+'_'+train_df['card2'].astype(str)).nunique()
len(train_df)
TARGET = 'isFraud'

for col in ['card1','card2','addr1','addr2','M4']:

    temp_dict = train_df.groupby([col])[TARGET].agg(['mean']).reset_index().rename(

                                                        columns={'mean': col+'_target_mean'})

    temp_dict.index = temp_dict[col].values

    temp_dict = temp_dict[col+'_target_mean'].to_dict()



    train_df[col+'_target_mean'] = train_df[col].map(temp_dict)

    test_df[col+'_target_mean']  = test_df[col].map(temp_dict)
i_cols = V_list_



for df in [train_df, test_df]:

    df['V_sum'] = df[i_cols].sum(axis=1).astype(np.int8)

    df['V_na'] = df[i_cols].isna().sum(axis=1).astype(np.int8)
# Binary encoding for M columns

i_cols = ['M1','M2','M3','M5','M6','M7','M8','M9']



for df in [train_df, test_df]:

    df['M_sum'] = df[i_cols].sum(axis=1).astype(np.int8)

    df['M_na'] = df[i_cols].isna().sum(axis=1).astype(np.int8)
# Let's add some kind of client uID based on cardID ad addr columns

# The value will be very specific for each client so we need to remove it

# from final feature. But we can use it for aggregations.

train_df['uid'] = train_df['card1'].astype(str)+'_'+train_df['card2'].astype(str)

test_df['uid'] = test_df['card1'].astype(str)+'_'+test_df['card2'].astype(str)



train_df['uid2'] = train_df['uid'].astype(str)+'_'+train_df['addr1'].astype(str)+'_'+train_df['addr2'].astype(str)

test_df['uid2'] = test_df['uid'].astype(str)+'_'+test_df['addr1'].astype(str)+'_'+test_df['addr2'].astype(str)



train_df['uid3'] = train_df['uid2'].astype(str)+'_'+train_df['P_emaildomain'].astype(str)+'_'+train_df['R_emaildomain'].astype(str)

test_df['uid3'] = test_df['uid2'].astype(str)+'_'+test_df['P_emaildomain'].astype(str)+'_'+test_df['R_emaildomain'].astype(str)



# Check if the Transaction Amount is common or not (we can use freq encoding here)

# In our dialog with a model we are telling to trust or not to these values   

train_df['TransactionAmt_check'] = np.where(train_df['TransactionAmt'].isin(test_df['TransactionAmt']), 1, 0)

test_df['TransactionAmt_check']  = np.where(test_df['TransactionAmt'].isin(train_df['TransactionAmt']), 1, 0)

i_cols = ['card1','card2','uid','uid2','uid3']



for col in i_cols:

    for agg_type in ['mean','std']:

        new_col_name = col+'_TransactionAmt_'+agg_type

        temp_df = pd.concat([train_df[[col, 'TransactionAmt']], test_df[[col,'TransactionAmt']]])

        #temp_df['TransactionAmt'] = temp_df['TransactionAmt'].astype(int)

        temp_df = temp_df.groupby([col])['TransactionAmt'].agg([agg_type]).reset_index().rename(

                                                columns={agg_type: new_col_name})

        

        temp_df.index = list(temp_df[col])

        temp_df = temp_df[new_col_name].to_dict()   

    

        train_df[new_col_name] = train_df[col].map(temp_df)

        test_df[new_col_name]  = test_df[col].map(temp_df)
START_DATE = datetime.datetime.strptime('2017-11-30', '%Y-%m-%d')

for df in [train_df, test_df]:

    

    # Temporary variables for aggregation

    df['DT'] = df['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds = x)))

    df['DT_M'] = ((df['DT'].dt.year-2017)*12 + df['DT'].dt.month).astype(np.int8)

    df['DT_W'] = ((df['DT'].dt.year-2017)*52 + df['DT'].dt.weekofyear).astype(np.int8)

    df['DT_D'] = ((df['DT'].dt.year-2017)*365 + df['DT'].dt.dayofyear).astype(np.int16)

    

    df['DT_hour'] = (df['DT'].dt.hour).astype(np.int8)

    df['DT_day_week'] = (df['DT'].dt.dayofweek).astype(np.int8)

    df['DT_day_month'] = (df['DT'].dt.day).astype(np.int8)
df[['DT','DT_M','DT_W', 'DT_D','DT_hour', 'DT_day_week', 'DT_day_month' ]].head()
# Total transactions per timeblock

for col in ['DT_M','DT_W','DT_D', 'DT_hour', 'DT_day_week', 'DT_day_month']:

    temp_df = pd.concat([train_df[[col]], test_df[[col]]])

    fq_encode = temp_df[col].value_counts().to_dict()

            

    train_df[col+'_total'] = train_df[col].map(fq_encode)

    test_df[col+'_total']  = test_df[col].map(fq_encode)

    
train_df[['DT_M_total','DT_W_total','DT_D_total', 'DT_hour_total', 'DT_day_week_total', 'DT_day_month_total']].head()
V_col_remove = [col for col in V_list if col not in V_list_]

#print(V_col_remove)
for col in ['M1','M2','M3','M5','M6','M7','M8','M9']:

    train_df[col] = train_df[col].map({'T':1, 'F':0})

    test_df[col]  = test_df[col].map({'T':1, 'F':0})
train_df.dtypes
train_df['ProductCD'].head()

#.fillna('unseen_before_label')
train_df['ProductCD'].isnull().sum()
train_df['ProductCD'].count()
train_df['ProductCD'].fillna('unseen_before_label').head()
train_df.select_dtypes(include=['object'])
for col in list(train_df):

    if train_df[col].dtype=='object':

        print(col)
# Frequency encoding for categorical column and then doing categorical encoding

for col in list(train_df):

    if train_df[col].dtype=='object':

        print(col)

        train_df[col] = train_df[col].fillna('NaN')

        test_df[col]  = test_df[col].fillna('NaN')

        

        train_df[col] = train_df[col].astype(str)

        test_df[col] = test_df[col].astype(str)

        

        le = LabelEncoder()

        le.fit(list(train_df[col])+list(test_df[col]))

        train_df[col] = le.transform(train_df[col])

        test_df[col]  = le.transform(test_df[col])

        

        train_df[col] = train_df[col].astype('category')

        test_df[col] = test_df[col].astype('category')
train_df['R_emaildomain']
rm_cols = [



    'uid','uid2','uid3',            

    'DT','DT_M', 'DT_W', 'DT_D','DT_hour', 'DT_day_week', 'DT_day_month' # Already we have considered these

    

]



rm_cols.extend(V_col_remove)

#rm_cols
features_columns = [col for col in list(train_df) if col not in rm_cols]
len(train_df)
len(test_df)
features_columns.remove('isFraud')

features_columns
########################### Model

import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold

import gc

SEED = 10



def make_predictions(train_df, test_df, features_columns, target, lgb_params, NFOLDS=3):

    N_SPLITS = 10    

    folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)



    # Main Data

    X,y = train_df[features_columns], train_df[TARGET]



    # Test Data and expport DF

    P,P_y  = test_df[features_columns], test_df[TARGET]

    

    test_df = test_df[['TransactionID',target]]    

    predictions = np.zeros(len(test_df))



    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y, groups=y)):

        print('Fold:',fold_+1)

        tr_x, tr_y = X.iloc[trn_idx,:], y[trn_idx]    

        vl_x, vl_y = X.iloc[val_idx,:], y[val_idx]    

        train_data = lgb.Dataset(tr_x, label=tr_y)

        #valid_data = lgb.Dataset(vl_x, label=v_y)  

        

        if LOCAL_TEST:

            valid_data = lgb.Dataset(P, label=P_y) 

        else:

            valid_data = lgb.Dataset(vl_x, label=vl_y)  





        estimator = lgb.train(

                lgb_params,

                train_data,

                valid_sets = [train_data, valid_data],

                verbose_eval = 1000,

            )

        

        pp_p = estimator.predict(P)

        predictions += pp_p/NFOLDS

        

        if LOCAL_TEST:

            feature_imp = pd.DataFrame(sorted(zip(estimator.feature_importance(),X.columns)), columns=['Value','Feature'])

            print(feature_imp)

        

       # del tr_x, tr_y, vl_x, vl_y, train_data, valid_data

       # gc.collect()

        

    test_df['prediction']  = predictions

    

    return test_df       
LOCAL_TEST = False

# Model params

from sklearn import metrics

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

# Model Train

if LOCAL_TEST:

    lgb_params['learning_rate'] = 0.01

    lgb_params['n_estimators'] = 20000

    lgb_params['early_stopping_rounds'] = 100

    test_predictions = make_predictions(train_df, test_df, features_columns, TARGET, lgb_params)

    print(metrics.roc_auc_score(test_predictions[TARGET], test_predictions['prediction']))

else:

    lgb_params['learning_rate'] = 0.01

    lgb_params['n_estimators'] = 800

    lgb_params['early_stopping_rounds'] = 100    

    test_predictions = make_predictions(train_df, test_df, features_columns, TARGET, lgb_params, NFOLDS=2)
# submission

test_predictions['isFraud'] = test_predictions['prediction']

test_predictions[['TransactionID','isFraud']].to_csv('submission.csv', index=False)