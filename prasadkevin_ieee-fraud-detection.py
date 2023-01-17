#!rm -r /opt/conda/lib/python3.6/site-packages/lightgbm
#!git clone --recursive https://github.com/Microsoft/LightGBM
#!apt-get install -y -qq libboost-all-dev
'''%%bash

cd LightGBM

rm -r build

mkdir build

cd build

cmake -DUSE_GPU=1 -DOpenCL_LIBRARY=/usr/local/cuda/lib64/libOpenCL.so -DOpenCL_INCLUDE_DIR=/usr/local/cuda/include/ ..

make -j$(nproc)'''
#!cd LightGBM/python-package/;python3 setup.py install --precompile
# !mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

# !rm -r LightGBM
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# plotting

import matplotlib.pyplot as plt

from matplotlib import ticker

import seaborn as sns



import gc # garbage collection



from sklearn.preprocessing import StandardScaler, MinMaxScaler
plt.style.use('seaborn-whitegrid')
train_identity = pd.read_csv('../input/ieee-fraud-detection/train_identity.csv')

print('The training dataset for identity information has {0} columns and {1} rows'.format(*train_identity.shape))



train_transaction = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv')

print('The training dataset for transactions has {0} columns and {1} rows'.format(*train_transaction.shape))
test_identity = pd.read_csv('../input/ieee-fraud-detection/test_identity.csv')

print('The test dataset for identity information has {0} columns and {1} rows'.format(*test_identity.shape))



test_transaction = pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv')

print('The test dataset for identity information has {0} columns and {1} rows'.format(*test_transaction.shape))
train_idNA = train_identity.isnull().sum()

print('='*30+'Identity'+'='*30)

print('----------Columns with no missing values----------')

print(*train_idNA[train_idNA == 0].index, sep='\t')

print('----------Columns with missing values-------------')

print(*train_idNA[train_idNA > 0].index, sep='\t')



train_txnNA = train_transaction.isnull().sum()

print('\n========================transactions====================')

print('----------Columns with no missing values----------')

print(*train_txnNA[train_txnNA == 0].index, sep='\t')

print('----------Columns with missing values-------------')

print(*train_txnNA[train_txnNA > 0].index, sep='\t')
test_idNA = test_identity.isnull().sum()

print('========================Identity========================')

print('----------Columns with no missing values----------')

print(*test_idNA[test_idNA == 0].index, sep='\t')

print('----------Columns with missing values-------------')

print(*test_idNA[test_idNA > 0].index, sep='\t')



test_txnNA = test_transaction.isnull().sum()

print('\n========================transactions====================')

print('----------Columns with no missing values----------')

print(*test_txnNA[test_txnNA == 0].index, sep='\t')

print('----------Columns with missing values-------------')

print(*test_txnNA[test_txnNA > 0].index, sep='\t')
c_cols = [f'C{i}' for i in range(1,15)] # counts (Numeric)

d_cols = [f'D{i}' for i in range(1,16)] # timedeltas (Numeric)

m_cols = [f'M{i}' for i in range(1,10)] # matches (Boolean/ Categorical)

v_cols = [f'V{i}' for i in range(1,340)] # Numeric

id_cols = [f'id_{i:02d}' for i in range(1,39)] # id_12 to id_38 are catergorical columns

id_colC = id_cols[11:]



card_cols = [f'card{i}' for i in range(1,7)]
txn_cat_cols = ['ProductCD', 'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain'] + card_cols + m_cols

id_cat_cols = ['DeviceInfo', 'DeviceType'] + id_colC

cat_cols = txn_cat_cols + id_cat_cols
train =  train_transaction.merge(right= train_identity, on = 'TransactionID', how = 'left')

train.shape
test =  test_transaction.merge(right= test_identity, on = 'TransactionID', how = 'left')

test.shape
del train_identity, train_transaction, test_transaction, test_identity

gc.collect()
# take the number of decimals on the Transaction Amount field. More than two decimals may indicate transactions made overseas.

train['TransactionAmt_decimal_lenght'] = train['TransactionAmt'].astype(str).str.split('.', expand=True)[1].str.len()

test['TransactionAmt_decimal_lenght'] = test['TransactionAmt'].astype(str).str.split('.', expand=True)[1].str.len()
train['Txn_nulls'] = train.isna().sum(axis = 1)

test['Txn_nulls'] = test.isna().sum(axis = 1)
sns.distplot(train['Txn_nulls'][train['isFraud'] == 0], label = 'Not Fraud')

sns.distplot(train['Txn_nulls'][train['isFraud'] == 1], label = 'Is Fraud')

plt.legend()
train[cat_cols] = train[cat_cols].astype('category')

test[cat_cols] = test[cat_cols].astype('category')
def reduce_mem_usage2(df):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.        

    """

    start_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    

    for col in df.columns:

        col_type = df[col].dtype.name

        

        if col_type not in ('object', 'category'):

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

        else:

            df[col] = df[col].astype('category')

            try:

                df[col] = df[col].cat.add_categories('UNK').fillna('UNK')

            except ValueError:

                pass    



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    

    return df
def get_dataset_diff(trn, tst, cat_cols):

    for col in cat_cols:

        train_set = set(trn[col].cat.categories)

        test_set = set(tst[col].cat.categories)

        new_values = len(test_set - train_set)



        if new_values == 0:

            pass

        else:

            print ('**There are {} new values in the test set for the `{}` column'.format(new_values, col))
train = reduce_mem_usage2(train)

test = reduce_mem_usage2(test)
numerical_cols = train.select_dtypes('number').columns

categorical_cols = train.select_dtypes(exclude = 'number').columns



print ('There are {} numerical columns and {} categorical columns'.format(len(numerical_cols), len(categorical_cols)))
get_dataset_diff(train, test, categorical_cols)
def set_valid_cats(trn, test, cat_cols):

    for col in cat_cols:

        cat_values = trn[col].cat.categories

        test[col] = test[col].cat.set_categories(cat_values)

        

        if test[col].isna().sum() > 0:

            print ('Resetting categorical levels created {} null values'.format(test[col].isna().sum()))

        try:

            test[col] = test[col].cat.add_categories('UNK').fillna('UNK')

        except ValueError:

            test[col] = test[col].fillna('UNK')
set_valid_cats(train, test, categorical_cols)
# identify records having identity information

train['no_identity'] = train['id_01'].isna()*1

test['no_identity'] = test['id_01'].isna()*1
overall_fraud_rate = train['isFraud'].value_counts(normalize = True)[1]

overall_fraud_rate
train[card_cols].isna().sum()
fig, ax = plt.subplots(ncols= 3, nrows= 2, figsize = (20, 10))

ax = ax.ravel()



for i, col in enumerate(card_cols):

    if train[col].dtype.name == 'category':

        props = train.groupby(col, observed = True)['isFraud'].value_counts(dropna = False, normalize=True).unstack()

        props = props.sort_values(by=1, ascending = False).head(10)

        p = props.plot(kind='barh', stacked='True', ax = ax[i], legend = False)

        ax[i].vlines(1 - overall_fraud_rate, ymin= ax[i].get_ylim()[0], ymax= ax[i].get_ylim()[1], linestyle = ':' )

        ax[i].set_ylabel(col)

    else:

        sns.distplot(train[col][train['isFraud'] == 0] ,ax = ax[i], label = 'Not Fraud', hist = False,)

        sns.distplot(train[col][train['isFraud'] == 1] ,ax = ax[i], label = 'Is Fraud', hist = False)

        ax[i].set_xlabel(col)
for col in card_cols:

    card_count = train[col].value_counts().to_dict()

    train[col+'_count'] = train[col].map(card_count)

    test[col+'_count'] = test[col].map(card_count)
(train['TransactionDT'] / (3600*24)).describe()
plt.figure(figsize = (10, 5))

vals = plt.hist(train['TransactionDT'] / (3600*24), bins=182*24)

plt.xlim(1, 7)

plt.xlabel('Days')

plt.ylabel('Number of transactions')

plt.ylim(0,500)
vals = plt.hist(train['TransactionDT'] / (3600*24), bins=182*24)

plt.xlim(0, 3)

plt.xlabel('Days')

plt.ylabel('Number of transactions')

plt.ylim(0,500)
train['TransactionDays'] = train['TransactionDT'] / (60*60*24) - 9/24

test['TransactionDays'] = test['TransactionDT'] / (60*60*24) - 9/24
plt.figure(figsize = (10, 5))

vals = plt.hist(train['TransactionDays'], bins=182*24)

plt.xlim(120, 127)

plt.xlabel('Days')

plt.ylabel('Number of transactions')

plt.ylim(0,500)
train['TransactionHour'] = np.floor((train['TransactionDays'] % 1 )* 24)

train['TransactionDayofWeek'] = np.floor(train['TransactionDays'] % 7)



test['TransactionHour'] = np.floor((test['TransactionDays'] % 1 )* 24)

test['TransactionDayofWeek'] = np.floor(test['TransactionDays'] % 7)
weekday_freq = train['TransactionDayofWeek'].value_counts().to_dict()

train[col+'_weekfreq'] = train['TransactionDayofWeek'].map(weekday_freq)

test[col+'_weekfreq'] = test['TransactionDayofWeek'].map(weekday_freq)

    

hour_freq = train['TransactionHour'].value_counts().to_dict()

train[col+'_weekfreq'] = train['TransactionDayofWeek'].map(hour_freq)

test[col+'_weekfreq'] = test['TransactionDayofWeek'].map(hour_freq)
fraud_fracDay = train.groupby('TransactionDayofWeek')['isFraud'].mean()



plt.plot(fraud_fracDay)

plt.ylim(0.03, 0.04)

plt.hlines(y= overall_fraud_rate, xmin= 0, xmax= 6, linestyles= ':')
fraud_fracHr = train.groupby('TransactionHour')['isFraud'].mean()



plt.plot(fraud_fracHr)

plt.ylim(0.02, 0.12)
fig, ax = plt.subplots(figsize=(10, 6))

plot= sns.violinplot(x = train['isFraud'][train['TransactionAmt'] < 1000], y = train['TransactionAmt'][train['TransactionAmt'] < 1000].astype('float32'))
train.groupby(col)['TransactionAmt'].transform('mean')
for col in card_cols:

    if train[col].dtype.name == 'category':

        transaction_mean = train.groupby(col)['TransactionAmt'].transform('mean')

        train['TransactionAmt_to_mean_'+col] = train['TransactionAmt'] / transaction_mean

        test['TransactionAmt_to_mean_'+col] = test['TransactionAmt'] / transaction_mean
sns.violinplot(x = train['isFraud'][train['TransactionAmt'] < 1000], y = train['TransactionAmt_to_mean_card4'][train['TransactionAmt'] < 1000].astype(float))
numerical_ids = train[id_cols].select_dtypes('number').columns

categorical_ids = train[id_cols].select_dtypes(exclude = 'number').columns
fig, ax = plt.subplots(ncols= 4, nrows= 3, figsize = (20, 10))

ax = ax.ravel()



for i, col in enumerate(numerical_ids):

        

    sns.distplot(train[col][train['isFraud'] == 0] ,ax = ax[i], label = 'Not Fraud', hist = False)

    sns.distplot(train[col][train['isFraud'] == 1] ,ax = ax[i], label = 'Is Fraud', hist = False)

    ax[i].set_xlabel(col);
train[numerical_ids] = train[numerical_ids].fillna(-999)

test[numerical_ids] = test[numerical_ids].fillna(-999)
fig, ax = plt.subplots(ncols= 5, nrows= 6, figsize = (20, 20))

ax = ax.ravel()

##scaler = StandardScaler()



for i, col in enumerate(categorical_ids):

    ## if len(train[col].cat.categories) > 10:

        ## print ('column `{}` has more than ten unique levels'.format(col))

    props = train.groupby(col, observed = True)['isFraud'].value_counts(normalize=True).unstack()

    props = props.sort_values(by=1, ascending = False).head(10)

    p = props.plot(kind='barh', stacked='True', ax = ax[i], legend = False)

    ax[i].vlines(1 - overall_fraud_rate, ymin= ax[i].get_ylim()[0], ymax= ax[i].get_ylim()[1], linestyle = ':' )

    ax[i].set_ylabel(col)
fig, ax = plt.subplots(ncols= 5, nrows= 3, figsize = (20, 10))

ax = ax.ravel()



for i, col in enumerate(c_cols):



    sns.distplot(train[col][train['isFraud'] == 0] ,ax = ax[i], label = 'Not Fraud', hist = False)

    sns.distplot(train[col][train['isFraud'] == 1] ,ax = ax[i], label = 'Is Fraud', hist = False)

    ax[i].set_xlabel(col)
train[c_cols] = train[c_cols].fillna(-1)

test[c_cols] = test[c_cols].fillna(-1)
fig, ax = plt.subplots(ncols= 5, nrows= 3, figsize = (20, 10))

ax = ax.ravel()



for i, col in enumerate(d_cols):

    

    sns.distplot(train[col][train['isFraud'] == 0] ,ax = ax[i], label = 'Not Fraud', hist = False)

    sns.distplot(train[col][train['isFraud'] == 1] ,ax = ax[i], label = 'Is Fraud', hist = False)

    ax[i].set_xlabel(col)
train[d_cols]  = train[d_cols].fillna(-1)

test[d_cols]  = test[d_cols].fillna(-1)
fig, ax = plt.subplots(ncols= 3, nrows= 3, figsize = (20, 10))

ax = ax.ravel()

##scaler = StandardScaler()



for i, col in enumerate(m_cols):

    props = train.groupby(col)['isFraud'].value_counts(normalize=True).unstack()

    props = props.sort_values(by=1, ascending = False)

    p = props.plot(kind='barh', stacked='True', ax = ax[i], legend = False)

    ax[i].vlines(1 - overall_fraud_rate, ymin= ax[i].get_ylim()[0], ymax= ax[i].get_ylim()[1], linestyle = ':' )

    ax[i].set_ylabel(col)
for col in m_cols:

    count = train[col].value_counts().to_dict()

    train[col+'_count'] = train[col].map(count)

    test[col+'_count'] = test[col].map(count)
fig, ax = plt.subplots(figsize = (15, 8))

props = train.groupby('id_30')['isFraud'].value_counts(dropna = False, normalize=True).unstack()

props = props.sort_values(by=1, ascending = False).head(25)

p = props.plot(kind='bar', stacked='True', legend = False, ax = ax )

## plt.vlines(1 - overall_fraud_rate, linestyle = ':' )

plt.ylabel('id_30')
fig, ax = plt.subplots(figsize = (15, 8))

props = train.groupby('id_31')['isFraud'].value_counts(dropna = False, normalize=True).unstack()

props = props.sort_values(by=1, ascending = False).head(25)

p = props.plot(kind='bar', stacked='True', legend = False, ax = ax )

## plt.vlines(1 - overall_fraud_rate, linestyle = ':' )

plt.ylabel('id_31')
fig, ax = plt.subplots(figsize = (15, 8))

props = train.groupby('id_33')['isFraud'].value_counts(dropna = False, normalize=True).unstack()

props = props.sort_values(by=1, ascending = False).head(25)

p = props.plot(kind='bar', stacked='True', legend = False, ax = ax)

## plt.vlines(1 - overall_fraud_rate, linestyle = ':' )

plt.ylabel('id_33')
fig, ax = plt.subplots(figsize = (15, 8))

props = train.groupby('DeviceInfo')['isFraud'].value_counts(dropna = False, normalize=True).unstack()

props = props.sort_values(by=1, ascending = False).head(50)

p = props.plot(kind='bar', stacked='True', legend = False, ax = ax )

## plt.vlines(1 - overall_fraud_rate, linestyle = ':' )

plt.ylabel('DeviceInfo')
fig, ax = plt.subplots(figsize = (15, 8))

props = train.groupby('DeviceType')['isFraud'].value_counts(dropna = False, normalize=True).unstack()

props = props.sort_values(by=1, ascending = False)

p = props.plot(kind='bar', stacked='True', legend = False, ax = ax )

## plt.vlines(1 - overall_fraud_rate, linestyle = ':' )

plt.ylabel('DeviceInfo')
## For the purchaser email take the top level domain

train['topDomain_P_emaildomain'] = train['P_emaildomain'].str.split('.', expand=True)[0].astype('category')

test['topDomain_P_emaildomain'] = test['P_emaildomain'].str.split('.', expand=True)[0].astype('category')
train_categories = train['topDomain_P_emaildomain'].cat.categories

test['topDomain_P_emaildomain'] = test['topDomain_P_emaildomain'].cat.set_categories(train_categories)
P_email_freq = train['topDomain_P_emaildomain'].value_counts().to_dict()



train['P_topDomain_freq'] = train['topDomain_P_emaildomain'].map(P_email_freq)

test['P_topDomain_freq'] = test['topDomain_P_emaildomain'].map(P_email_freq)
fig, ax = plt.subplots(figsize = (15, 8))

props = train.groupby('topDomain_P_emaildomain')['isFraud'].value_counts(dropna = False, normalize=True).unstack()

props = props.sort_values(by=1, ascending = False).head(50)

p = props.plot(kind='bar', stacked='True', legend = False, ax = ax )

## plt.vlines(1 - overall_fraud_rate, linestyle = ':' )

plt.ylabel('Top Level Domain for Purchaser email')
train['topDomain_R_emaildomain'] = train['R_emaildomain'].str.split('.', expand=True)[0].astype('category')

test['topDomain_R_emaildomain'] = test['R_emaildomain'].str.split('.', expand=True)[0].astype('category')
train_categories = train['topDomain_R_emaildomain'].cat.categories

test['topDomain_R_emaildomain'] = test['topDomain_R_emaildomain'].cat.set_categories(train_categories)
R_email_freq = train['topDomain_R_emaildomain'].value_counts().to_dict()



train['R_topDomain_freq'] = train['topDomain_R_emaildomain'].map(R_email_freq)

test['R_topDomain_freq'] = test['topDomain_R_emaildomain'].map(R_email_freq)
fig, ax = plt.subplots(figsize = (15, 8))

props = train.groupby('topDomain_R_emaildomain')['isFraud'].value_counts(dropna = False, normalize=True).unstack()

props = props.sort_values(by=1, ascending = False).head(50)

p = props.plot(kind='bar', stacked='True', legend = False, ax = ax )

## plt.vlines(1 - overall_fraud_rate, linestyle = ':' )

plt.ylabel('Top Level Domain for recipient email')
for col in ('addr1', 'addr2'):

    count = train[col].value_counts().to_dict()

    train[col+'_freq']  = train[col].map(count)

    test[col+'_freq']  = test[col].map(count)
train[['dist1', 'dist2']].isna().sum()
sns.distplot(train['dist1'][train['isFraud'] == 0], label = 'Not Fraud', hist = False)

sns.distplot(train['dist1'][train['isFraud'] == 1], label = 'Is Fraud', hist = False)
sns.distplot(train['dist2'][train['isFraud'] == 0], label = 'Not Fraud', hist = False)

sns.distplot(train['dist2'][train['isFraud'] == 1], label = 'Is Fraud', hist = False)
train['dist1'].min()
train['missing_dist1'] = train['dist1'].isna()*1

test['missing_dist1'] = test['dist1'].isna()*1



train['dist1'].fillna(0, inplace = True)

test['dist1'].fillna(0, inplace = True)



train['missing_dist2'] = train['dist2'].isna()*1

test['missing_dist2'] = test['dist2'].isna()*1



train['dist2'].fillna(0, inplace = True)

test['dist2'].fillna(0, inplace = True)
sns.distplot(train['V201_card1_std'][train['isFraud'] == 0], label = 'Not Fraud', hist = False)

sns.distplot(train['V201_card1_std'][train['isFraud'] == 1], label = 'Is Fraud', hist = False)
group_mean = train.groupby('card1')['isFraud'].transform('mean')

train['V201_card1_mean'] = train['V201']*group_mean

test['V201_card1_mean'] = test['V201']*group_mean



group_dev = train.groupby('card1')['isFraud'].transform('std')

train['V201_card1_std'] = train['V201']*group_dev

test['V201_card1_std'] = test['V201']*group_dev



group_mean = train.groupby('card1')['isFraud'].transform('mean')

train['C1_card1_mean'] = train['C1']*group_mean

test['C1_card1_mean'] = test['C1']*group_mean



group_mean = train.groupby('card1')['isFraud'].transform('mean')

train['C13_card1_mean'] = train['C13']*group_mean

test['C13_card1_mean'] = test['C13']*group_mean



train['card1_addr1'] = train['card1'].astype('str')+ '_' + train['addr1'].astype('str')

test['card1_addr1'] = test['card1'].astype('str')+ '_' + test['addr1'].astype('str')



train['card2_addr1'] = train['card2'].astype('str')+ '_' + train['addr1'].astype('str')

test['card2_addr1'] = test['card2'].astype('str')+ '_' + test['addr1'].astype('str')



group_mean = train.groupby('card1')['isFraud'].transform('mean')

train['dist1_card1_mean'] = train['dist1']*group_mean

test['dist1_card1_mean'] = test['dist1']*group_mean
train['missing_rich_feat'] = train[v_cols].isna().sum(axis = 1)

test['missing_rich_feat'] = test[v_cols].isna().sum(axis = 1)



sns.distplot(train['missing_rich_feat'][train['isFraud'] == 0], label = 'Not Fraud', hist = False)

sns.distplot(train['missing_rich_feat'][train['isFraud'] == 1], label = 'Is Fraud', hist = False)
train[v_cols] = train[v_cols].fillna(-1)

test[v_cols] = test[v_cols].fillna(-1)
kw_cbar = {'vmax':1, 'vmin':-1, 'cmap': 'RdYlGn'}

corr = train[['isFraud'] +id_cols[:11]].corr('spearman')

sns.clustermap(corr, **kw_cbar)
corr = train[['isFraud'] +d_cols].corr('spearman')

sns.clustermap(corr, **kw_cbar)
corr = train[['isFraud'] +c_cols].corr('spearman')

sns.clustermap(corr, **kw_cbar)
train.isna().sum()[train.isna().sum() > 0]
test.isna().sum()[test.isna().sum() > 0]
train = train.sort_values(by = 'TransactionDT')

test = test.sort_values(by = 'TransactionDT')
features = [col for col in train.columns if col != 'isFraud']

X_train = train[features].copy()

X_test = test[features].copy()



y_train = train['isFraud']



#del train, test

#gc.collect()



X_train_ids = X_train[['TransactionID']].copy()

X_test_ids = X_test[['TransactionID']].copy()



X_train.drop(columns = 'TransactionID', inplace = True)

X_test.drop(columns = 'TransactionID', inplace = True)
categorical_cols = X_train.select_dtypes(exclude = 'number').columns

categorical_cols
from category_encoders import TargetEncoder
target_encoder = TargetEncoder(cols= categorical_cols.tolist(), smoothing = 10, return_df= True)

X_train = target_encoder.fit_transform(X = X_train, y = y_train)

X_test = target_encoder.transform(X = X_test)
X_train = reduce_mem_usage2(X_train)

X_test = reduce_mem_usage2(X_test)
X_train.isna().sum()[X_train.isna().sum() > 0]
import lightgbm as lgb

# import xgboost as xgb

# from xgboost import plot_importance

from sklearn.model_selection import KFold, TimeSeriesSplit

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_score

# from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

# from hyperopt import space_eval



from scipy.stats import uniform

from time import time

import datetime
pos_weight = sum(y_train == 0) / sum(y_train == 1)

pos_weight
params = {

    'objective': 'binary',

    'metric': 'auc',

    'n_estimators' : 1000,

    'boosting_type': 'gbdt',

    'num_leaves': 256,

    'min_data_in_leaf': 40,

    'max_depth': 20,

    'bagging_fraction': 0.7,

    'feature_fraction': 0.7,

    'lambda_l1': 0.6,

    'lambda_l2': 1.10,

    'scale_pos_weight': pos_weight}
clf = lgb.LGBMClassifier(**params)

clf.fit(X_train,y_train)

prediction= clf.predict_proba(X_test)[:,1]
lgb.plot_importance(clf, importance_type= 'gain', max_num_features= 50, precision = 2, figsize= (20,15))
lgb.plot_importance(clf, importance_type= 'split', max_num_features= 50, precision = 2, figsize= (20,15))
X_test_ids['isFraud'] = prediction

X_test_ids.to_csv('Submission_lightgbm.csv', index = False)
'''params = {

 'silent': 1,

 'colsample_bytree': 0.8,

 'subsample': 0.8,

 'n_estimators': 1000,

 'learning_rate': 0.05,

 'max_leaves': 72,

 'objective': 'binary',

 'max_depth': 9,

 'min_child_weight': 1,

 'eval_metric': 'auc',

 ##'tree_method': 'gpu_hist',

 'importance_type': 'weight'}'''
'''splits = 5

folds = TimeSeriesSplit(n_splits=splits)



aucs = list()

feature_importances = pd.DataFrame()

feature_importances['feature'] = X_train.columns



training_start_time = time()



for fold_n, (trn_idx, val_idx) in enumerate(folds.split(X_train)):

    clf = xgb.XGBClassifier(**params, verbosity = 0)

    

    X_trn, X_val = X_train.iloc[trn_idx], X_train.iloc[val_idx]

    y_trn, y_val = y_train.iloc[trn_idx], y_train.iloc[val_idx]

    clf.fit(X_trn,y_trn, early_stopping_rounds = 100, eval_set = [(X_trn,y_trn), (X_val, y_val)],  verbose= 500)

    del X_trn, y_trn

    

    val=clf.predict_proba(X_val)[:,1]

    

    feature_importances['fold_{}'.format(fold_n + 1)] = clf.feature_importances_

    

    del clf, X_val

    print('ROC accuracy: {}'.format(roc_auc_score(y_val, val)))

    del val,y_val



    gc.collect()'''

'''feature_importances['average'] = feature_importances.mean(axis=1)



plt.figure(figsize=(16, 16))

sns.barplot(data=feature_importances.sort_values(by='average', ascending=False).head(50), x='average', y='feature')

plt.title('50 TOP feature importance over cv folds average');'''
'''clf = xgb.XGBClassifier(**params, verbosity = 0)

clf.fit(X_train, y_train)'''
'''#fig, ax = plt.subplots(figsize=(20, 10))

xgb.plot_importance(clf.booster_, max_num_features = 50)'''
X_test_ids['isFraud'] = prediction

X_test_ids.to_csv('Submission_Xgb_baseline.csv', index = False)
'''params = {

    'max_bin' : 63,

    'n_estimators' : 1000,

    'learning_rate': 0.01,

    'min_data_in_leaf' : 50,

    'num_leaves' : 100,

    'sparse_threshold' : 1.0,

    ## 'device' : 'gpu',

    'save_binary': True,

    'seed' : 42,

    'feature_fraction_seed': 42,

    'bagging_seed' : 42,

    'drop_seed' : 42,

    'data_random_seed' : 42,

    'objective' : 'binary',

    'boosting_type' : 'gbdt',

    'verbose' : 0,

    'metric' : 'auc',

    'is_unbalance' : True,

    'boost_from_average' : False,

}'''
'''splits = 5

folds = TimeSeriesSplit(n_splits=splits)



aucs = list()

feature_importances = pd.DataFrame()

feature_importances['feature'] = X_train.columns



training_start_time = time()



for fold_n, (trn_idx, val_idx) in enumerate(folds.split(X_train)):

    cv_fold_start_time = time()

    print ('** Training fold {}'.format(fold_n + 1))

    X_trn, X_val = X_train.iloc[trn_idx], X_train.iloc[val_idx]

    y_trn, y_val = y_train.iloc[trn_idx], y_train.iloc[val_idx]

    eval_set  = [(X_trn,y_trn), (X_val, y_val)]

    

    clf = lgb.LGBMClassifier(**params)

    clf.fit(X_trn, y_trn,

            eval_set=eval_set, 

            eval_metric="auc",

            early_stopping_rounds=100,

            verbose= 500)

    del X_trn, y_trn

    

    val=clf.predict_proba(X_val)[:,1]

    

    feature_importances['fold_{}'.format(fold_n + 1)] = clf.feature_importances_

    

    del clf, X_val

    print('ROC accuracy: {}'.format(roc_auc_score(y_val, val)))

    del val,y_val



    gc.collect()

    

    cv_fold_end_time = time()

    print ('fold completed in {}s'.format(cv_fold_end_time - cv_fold_start_time))'''
feature_importances['average'] = feature_importances.mean(axis=1)



plt.figure(figsize=(16, 16))

sns.barplot(data=feature_importances.sort_values(by='average', ascending=False).head(50), x='average', y='feature')

plt.title('50 TOP feature importance over cv folds average');
'''for fold_n, (trn_idx, val_idx) in enumerate(folds.split(X_train)):

    cv_fold_start_time = time()

    train = lgb.Dataset(X_train.iloc[trn_idx], y_train.iloc[trn_idx])

    valid = lgb.Dataset(X_train.iloc[val_idx], y_train.iloc[val_idx])

    valid_sets  = [train, valid]

    

    clf = lgb.train(params, train_set = train, num_boost_round = 1000, valid_sets = valid_sets, 

                   early_stopping_rounds= 100, verbose_eval = 500,)

    del train, valid, trn_idx

    

    val=clf.predict(X_train.iloc[val_idx], raw_score= True)

    

    feature_importances['fold_{}'.format(fold_n + 1)] = clf.feature_importances_

     

    print('ROC accuracy: {}'.format(roc_auc_score(y_train.iloc[val_idx], val)))

    del clf, val,val_idx



    gc.collect()

    

    cv_fold_end_time = time()

    print ('fold completed in {}s'.format(cv_fold_end_time - cv_fold_start_time))'''
'''def objective(space):

    print ('='*30 + 'New Run' + '='*30)

    auc_scores = np.zeros(5)

    ## Do cross validation

    for fold_n, (trn_idx, val_idx) in enumerate(folds.split(X_train)):

        clf = xgb.XGBClassifier(n_estimators = 1000,

                                tree_method= 'gpu_hist',

                                max_depth = int(space['max_depth']),

                                min_child_weight = space['min_child_weight'],

                                subsample = space['subsample'],

                                num_leaves = space['num_leaves'],

                                feature_fraction = space['feature_fraction'],

                                max_delta_step = space['max_delta_step'],

                                colsample_bytree = 0.8,

                                importance_type = 'weight'

                               )

    

        X_trn, X_val = X_train.iloc[trn_idx], X_train.iloc[val_idx]

        y_trn, y_val = y_train.iloc[trn_idx], y_train.iloc[val_idx]

        eval_set  = [(X_trn,y_trn), (X_val, y_val)]



        clf.fit(X_trn, y_trn,

            eval_set=eval_set, eval_metric="auc",

            early_stopping_rounds=100,

            verbose= False)

        

        del X_trn, y_trn, eval_set

        gc.collect()

        

        pred = clf.predict_proba(X_val)[:,1]

        auc = roc_auc_score(y_val, pred)      

        auc_scores[fold_n] = auc

        print ("Score for fold {}: {}".format(fold_n+1, auc))        

        

        del pred, auc, clf

        gc.collect()

        

    print ("**Average AUC across all folds: {}".format(auc_scores.mean()))

    return{'loss':1-auc_scores.mean(), 'status': STATUS_OK }





space ={

        'max_depth': hp.quniform("max_depth", 3, 10, 1),

        'min_child_weight': hp.quniform ('min_child', 1, 10, 1),

        'subsample': hp.uniform ('subsample', 0.8, 1),

        'num_leaves': hp.choice('num_leaves', list(range(20, 100, 10))),

        'feature_fraction': hp.uniform('feature_fraction', 0.4, .8),

        'max_delta_step': hp.quniform ('max_delta_step', 1, 10, 1)}'''
'''trials = Trials()

best = fmin(fn=objective,

            space=space,

            algo=tpe.suggest,

            max_evals=30,

            trials=trials)'''
'''best_params = space_eval(space, best)

print (best_params)'''
'''clf = xgb.XGBClassifier(**best_params, 

                        n_estimators = 1000,

                        tree_method= 'gpu_hist',

                        colsample_bytree = 0.8)

clf.fit(X_train,y_train)

prediction_opt= clf.predict_proba(X_test)[:,1]



X_test_ids['isFraud'] = prediction_opt

X_test_ids.to_csv('Submission_hyperopt.csv', index = False)'''