import numpy as np

import pandas as pd

pd.set_option("display.max_columns", 500)

import matplotlib.pyplot as plt

import seaborn as sns

import datetime

from sklearn.preprocessing import LabelEncoder

import xgboost as xgb

from xgboost import XGBClassifier

from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold, KFold

from sklearn.metrics import roc_auc_score

from hyperopt import fmin, hp, tpe, space_eval

import lightgbm as lgb



import gc

import warnings

warnings.filterwarnings('ignore')
train_identity = pd.read_csv('../input/train_identity.csv', index_col='TransactionID')

train_transaction = pd.read_csv('../input/train_transaction.csv', index_col='TransactionID')



train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)



gc.enable()

del train_transaction, train_identity

gc.collect()



print(train.shape)

train.head()
train.info()
# From kernel https://www.kaggle.com/gemartin/load-data-reduce-memory-usage

def reduce_mem_usage(df):

    for col in df.columns:

        col_type = df[col].dtype

        

        if col_type != object:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[: 3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min  and c_max < np.iinfo(np.int16).max:

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

    

    end_mem = df.memory_usage().sum()/ 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    

    return df
%%time

train = reduce_mem_usage(train)
id_cols = [c for c in train.columns if 'id' in c]

C_cols = [c for c in train.columns if 'C' in c]

D_cols = [c for c in train.columns if c.startswith('D')]

card_cols = [c for c in train.columns if 'card' in c]

id_category_cols = id_cols[11: ]

id_numeric_cols = id_cols[: 11]



for col in ['addr1', 'addr2']:

    train[col] = train[col].astype('object')

for col in card_cols:

    train[col] = train[col].astype('object')

for col in id_category_cols:

    train[col] = train[col].astype('object')
train['TransactionAmt'] = train['TransactionAmt'].astype(float)

train['TransAmtLog'] = np.log(train['TransactionAmt'])

train['TransAmtDemical'] = train['TransactionAmt'].astype('str').str.split('.', expand=True)[1].str.len()



plt.figure(figsize=(15, 10))

plt.suptitle('Transaction Values Distribution', fontsize=22)

plt.subplot(221)

g = sns.distplot(train['TransactionAmt'])

g.set_title('Transaction Amount Distribution')

g.set_xlabel('')

g.set_ylabel('Probability', fontsize=15)



plt.subplot(222)

g1 = sns.distplot(np.log(train['TransactionAmt']))

g1.set_title('Transaction Amount Log Distribution')

g1.set_xlabel('')

g.set_ylabel('Probability', fontsize=15)



plt.figure(figsize=(15, 10))



plt.subplot(212)

g4 = plt.scatter(range(train[train['isFraud'] == 0].shape[0]), 

                np.sort(train[train['isFraud'] == 0]['TransactionAmt'].values), 

                label='NoFraud', alpha=.2)

g4 = plt.scatter(range(train[train['isFraud'] == 1].shape[0]), 

                np.sort(train[train['isFraud'] == 1]['TransactionAmt'].values), 

                label='Fraud', alpha=.2)

g4 = plt.title('ECDF \nFRAUD and NO FRAUD Transaction Amount Distribution', fontsize=18)

g4 = plt.xlabel('Index')

g4 = plt.ylabel('Amount Distribution', fontsize=15)

g4 = plt.legend()



plt.show()
train.describe()
def transform_email(df):

    for col in ['R_emaildomain', 'P_emaildomain']:

        col1 = col.replace('domain', 'Corp')

        df[col1] = df[col]

        df.loc[df[col].isin(['gmail.com', 'gmail']), col1] = 'Google'

        df.loc[df[col].isin(['yahoo.com', 'yahoo.com.mx',  'yahoo.co.uk', 'yahoo.co.jp', 

                                 'yahoo.de', 'yahoo.fr', 'yahoo.es', 'yahoo.com.mx', 

                                 'ymail.com']), col1] = 'Yahoo'

        df.loc[df[col].isin(['hotmail.com','outlook.com','msn.com', 'live.com.mx', 'hotmail.es', 

                                 'hotmail.co.uk', 'hotmail.de', 'outlook.es', 'live.com', 'live.fr', 

                                 'hotmail.fr']), col1] = 'Microsoft'

        df.loc[df[col].isin(['aol.com', 'verizon.net']), col1] = 'Verizon'

        df.loc[df[col].isin(['att.net', 'sbcglobal.net', 'bellsouth.net']), col1] = 'AT&T'

        df.loc[df[col].isin(['icloud.com', 'mac.com', 'me.com']), col1] = 'Apple'

        df.loc[df[col1].isin(df['R_emailCorp'].value_counts()\

                                      [df['R_emailCorp'].value_counts() <= 1000].index), col1] = 'Others'



        col2 = col.replace('domain', 'Google')

        df[col2] = df[col1].str.contains('Google') * 1

        

        col3 = col.replace('domain', '_prefix')

        df[col3] = df[col].str.split('.').str[0]

        

        col4 = col.replace('domain', '_suffix')

        df[col4] = df[col].str.split('.').str[-1]

    

    return df
def transform_TransactionDT(df):

    START_DATE = '2017-12-01'

    start_date = datetime.datetime.strptime(START_DATE, '%Y-%m-%d')

    df['TransactionDT'] = df['TransactionDT'].apply(lambda x: (start_date + datetime.timedelta(seconds=x)))



    df['Year'] = df['TransactionDT'].dt.year

    df['Weekday'] = df['TransactionDT'].dt.dayofweek

    df['Hour'] = df['TransactionDT'].dt.hour

    df['Day'] = df['TransactionDT'].dt.day

    

    return df
def transform_id_cols(df):

    # Dealing with id_30

    df['id_30'] = df['id_30'].replace('nan', np.nan)

    df['System'] = df['id_30'].astype('str').str.split('.', expand=True)[0].str.split('_', expand=True)[0]

    df['SystemCorp'] = df['System'].str.split(expand=True)[0]

    

    # Dealing with id_31

    df['Browser'] = df['id_31'].str.replace(r'\d+.?\d*', '')

    

    df['LastestBrowser'] = df['id_31']

    df.loc[df['LastestBrowser'].isin(['samsung browser 7.0', 'opera 53.0', 'mobile safari 10.0', 'chrome 63.0 for android', 

                                       'google search application 49.0', 'firefox 60.0', 'edge 17.0', 'chrome 69.0', 

                                       'chrome 67.0 for android', 'chrome 64.0', 'chrome 63.0 for ios', 'chrome 65.0', 

                                       'chrome 64.0 for android', 'chrome 64.0 for ios', 'chrome 66.0', 

                                       'chrome 65.0 for android', 'chrome 65.0 for ios', 'chrome 66.0 for android', 

                                       'chrome 66.0 for ios']), 'LastestBrowser'] = 1

    df.loc[df['LastestBrowser'].str.len() > 1, 'LastestBrowser'] = 0

    

    df['BrowserCorp'] = df['id_31']

    df.loc[df['BrowserCorp'].str.contains('samsung', case=False, na=False), 'BrowserCorp'] = 'Samsung'

    df.loc[df['BrowserCorp'].str.contains('safari', case=False, na=False), 'BrowserCorp'] = 'Apple'

    df.loc[df['BrowserCorp'].str.contains('chrome|google', case=False, na=False), 'BrowserCorp'] = 'Google'

    df.loc[df['BrowserCorp'].str.contains('firefox', case=False, na=False), 'BrowserCorp'] = 'Mozilla'

    df.loc[df['BrowserCorp'].str.contains('edge|ie|microsoft', case=False, na=False, regex=True), 'BrowserCorp'] = 'Microsoft'

    df.loc[df['BrowserCorp'].isin(df['BrowserCorp'].value_counts()\

                                  [df['BrowserCorp'].value_counts()< 1000].index), ['BrowserCorp']] = 'other'

    

    # Dealing with id_33

    df['DisplayWidth'] = df['id_33'].str.split('x', expand=True)[0].astype(float)

    df['DisplayHeight'] = df['id_33'].str.split('x', expand=True)[1].astype(float)

    

    # Dealing with DeviceInfo

    df['DeviceType1'] = df['DeviceInfo'].str.split('-', expand=True)[0]

    df['DeviceType2'] = df['DeviceType1'].str.split(' ', expand=True)[0]

           

    return df
def transform_number(df):

    df['C1_min_C5'] = (df['C1'] - df['C5']).apply(lambda x: 0 if x <= 0 else 1)

    

    for col in ['addr1__addr2', 'addr1__card3', 'P_emaildomain__id_33', 'R_emaildomain__id_33', 

                'card6__addr2', 'id_32__V146', 'id_19__id_20']:

        col1, col2 = col.split('__')

        df[col] = df[col1].astype('str') + '_' + df[col2].astype('str')

        df.loc[df[col].str.contains('nan', na=True), col] = np.nan

       

    for col in card_cols:

        df[col] = df[col].fillna(-999)

    

    df['count_card3_P_emaildomain'] = df.groupby('card3').P_emaildomain.transform('count')

    df['count_card3_R_emaildomain'] = df.groupby('card3').R_emaildomain.transform('count')

    df['count_card6_P_emaildomain'] = df.groupby('card6').P_emaildomain.transform('count')

    df['count_card6_R_emaildomain'] = df.groupby('card6').R_emaildomain.transform('count')

    

    return df
test_identity = pd.read_csv('../input/test_identity.csv', index_col='TransactionID')

test_transaction = pd.read_csv('../input/test_transaction.csv', index_col='TransactionID')



test = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)



test['TransactionAmt'] = test['TransactionAmt'].astype(float)

test['TransAmtLog'] = np.log(test['TransactionAmt'])

test['TransAmtDemical'] = test['TransactionAmt'].astype('str').str.split('.', expand=True)[1].str.len()



for col in ['addr1', 'addr2']:

    test[col] = test[col].astype('object')

for col in card_cols:

    test[col] = test[col].astype('object')

for col in id_category_cols:

    test[col] = test[col].astype('object')



del test_transaction, test_identity

gc.collect()



print(test.info())



test = reduce_mem_usage(test)
%%time

train = transform_email(train)

train = transform_TransactionDT(train)

train = transform_id_cols(train)

train = transform_number(train)



test = transform_email(test)

test = transform_TransactionDT(test)

test = transform_id_cols(test)

test = transform_number(test)



print(train.shape)

print(test.shape)
# encoding_cols = ['card1','card2','card3','card5', 

#                  'C1','C2','C5','C6','C9','C11','C12','C13','C14', 

#                  'addr1','addr2','dist1', 

#                  'P_emaildomain', 'R_emaildomain',

#                  'id_01','id_02','id_03','id_05','id_06','id_09', 

#                  'id_11','id_13','id_14','id_17','id_19','id_20','id_30',

#                  'id_31','id_33','Browser','DeviceInfo','DeviceType1','System']

# for col in encoding_cols:

#     temp_df = pd.concat([train[[col]], test[[col]]])

#     fq_encode = temp_df[col].value_counts().to_dict()   

#     train[col+'_fq_enc'] = train[col].map(fq_encode)

#     test[col+'_fq_enc']  = test[col].map(fq_encode)

    

#     del temp_df

#     gc.collect()
cols_to_drop = ['V194', 'V195', 'V247', 'V142', 'V141', 'V191', 'V173', 'M1', 'V325', 'V138', 'V1', 'V41', 'V14','V65', 

                'V27', 'V240', 'V98', 'V105', 'V316', 'V301', 'V123', 'V136', 'V113', 'id_18', 'V110', 'V115', 'V108', 

                'V124', 'V104', 'V241', 'V191', 'V186', 'V252', 'V174', 'V172', 'V181', 'id_36', 'id_29', 'id_28', 'V185', 

                'id_10', 'V223', 'id_04', 'V242', 'V248', 'V223', 'V161', 'V226', 'V276', 'V278', 'id_35', 'id_25', 'V7', 

                'id_21', 'id_24', 'V295', 'V133', 'V299', 'V305', 'V286', 'V319', 'V321', 'V284', 'V116', 'V300', 'id_23', 

                'V311', 'V114', 'V118', 'V121', 'V106', 'V122', 'V125', 'V137', 'id_26', 'V103', 'V318', 'V120', 'id_27', 

                'V119', 'V293', 'V290', 'id_07', 'V132', 'dist2', 'V296', 'D7', 'V135', 'V297', 'V111', 'V298', 'V101', 

                'V309', 'id_08', 'V109', 'C3', 'V102', 'V107', 'V117', 'V112', 'V320', 'V134', 'id_22', 'V129', 'V281',  

                'V235', 'V159', 'V312', 'V274', 'V263', 'V33', 'V199', 'V327', 'V190', 'V322', 'V266', 'V207', 'V336', 

                'V160', 'V153', 'V157', 'V126', 'V273', 'V28', 'V91']
train.drop(cols_to_drop, axis=1, inplace=True)

test.drop(cols_to_drop, axis=1, inplace=True)



print(train.shape)

print(test.shape)
le = LabelEncoder()

for col in train.select_dtypes(include=['object', 'category']).columns:

    le.fit(list(train[col].astype(str).values) + list(test[col].astype(str).values))

    train[col] = le.transform(list(train[col].astype(str).values))

    test[col] = le.transform(list(test[col].astype(str).values))
X_train = train.drop(['isFraud', 'TransactionDT'], axis=1)

y_train = train['isFraud']

X_test = test.drop(['TransactionDT'], axis=1)

del train



X_train = reduce_mem_usage(X_train)

X_test = reduce_mem_usage(X_test)



gc.collect()
%%time

params = {

    'learning_rate': 0.05, 

    'max_depth': 9, 

    'gamma': 0.1, 

    'alpha': 4, 

    'subsample': 0.9, 

    'colsample_bytree': 0.9, 

    'Missing': -999

}



cv_scores = []

splits = 5

y_preds = np.zeros(len(X_test))

XGB_feature_importances = np.zeros(X_train.shape[1])

skf = StratifiedKFold(n_splits=splits)

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):

    clf = XGBClassifier(

        n_estimators=500,

        **params,

        tree_method='gpu_hist',

        early_stopping_rounds=100,

        random_state=4

    )



    X_tr, X_vl = X_train.iloc[tr_idx, :], X_train.iloc[val_idx, :]

    y_tr, y_vl = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        

    clf.fit(X_tr, y_tr)

        

    y_pred_train = clf.predict_proba(X_vl)[:,1]

    score = roc_auc_score(y_vl, y_pred_train)

    print("FOLD: ",fold,' AUC {}'.format(score))

    cv_scores.append(score)

    

    y_preds += clf.predict_proba(X_test)[:,1] / splits

    XGB_feature_importances += clf.feature_importances_ / splits

    

    del X_tr, X_vl, y_tr, y_vl, clf, y_pred_train    

    gc.collect()



print('CV Score : Mean - %.7g | Std - %.7g |Min - %.7g | Max - %.7g' % (np.mean(cv_scores), np.std(cv_scores),

                                    np.min(cv_scores), np.max(cv_scores)))



XGB_feature_importances = pd.Series(XGB_feature_importances, X_train.columns).sort_values(ascending=False)



plt.figure(figsize=(12, 6))

XGB_feature_importances[: 50].plot(kind='bar', title='Feature Importances')

plt.ylabel('Feature Importances Score')

plt.show()
zero_features = XGB_feature_importances[XGB_feature_importances.values == 0.0].index

print('There are %d features with 0.0 importance' % len(zero_features))

print(zero_features)
low_importance = XGB_feature_importances[300: -1]

low_importance
XGB_submission = pd.read_csv('../input/sample_submission.csv', index_col='TransactionID')

XGB_submission['isFraud'] = y_preds

XGB_submission.to_csv('XGB_fraud_detection.csv')

XGB_feature_importances.to_csv('XGB_feature_importances.csv')