import numpy as np
import pandas as pd
import warnings
warnings.simplefilter('ignore')
df_application_train = pd.read_csv('../input/application_train.csv')
df_application_train.head()
df_application_test = pd.read_csv('../input/application_test.csv')
df_application_test.head()
print(df_application_train.shape, df_application_test.shape)
for c in df_application_train.columns[df_application_train.dtypes == 'object']:
    c_train = set(df_application_train[c].unique())
    c_test = set(df_application_test[c].unique())
    diff = c_train ^ c_test
    if len(diff) > 0:
        print('feature ' + c + ' has different values: ', diff)
df_application_train['CODE_GENDER'] = df_application_train['CODE_GENDER'] \
                                                        .map(lambda x: x if x != 'XNA' else np.nan)
df_application_train['NAME_INCOME_TYPE'] = df_application_train['NAME_INCOME_TYPE'] \
                                                        .map(lambda x: x if x != 'Maternity leave' else np.nan)
df_application_train['NAME_FAMILY_STATUS'] = df_application_train['NAME_FAMILY_STATUS'] \
                                                        .map(lambda x: x if x != 'Unknown' else np.nan)
for c in df_application_train.columns[df_application_train.dtypes == 'object']:
    d = df_application_train[c].value_counts()
    if df_application_train[c].nunique() == 2:
        d[0] = 0
        d[1] = 1
            
    df_application_train[c] = df_application_train[c].map(d)
    df_application_test[c] = df_application_test[c].map(d)
features_with_small_variance = df_application_train.columns[(df_application_train.std(axis = 0) < .01).values]
df_application_train[features_with_small_variance].describe().T
features_with_small_variance = df_application_test.columns[(df_application_test.std(axis = 0) < .01).values]
df_application_test[features_with_small_variance].describe().T
df_application_train.drop(features_with_small_variance, axis = 1, inplace = True)
df_application_test.drop(features_with_small_variance, axis = 1, inplace = True)
print(df_application_train.shape, df_application_test.shape)
df_bureau_balance = pd.read_csv('../input/bureau_balance.csv')
df_bureau_balance.head()
for val in df_bureau_balance['STATUS'].unique():
    df_bureau_balance['STATUS_' + val] = (df_bureau_balance['STATUS'] == val).map(int)

df_bureau_balance.drop('STATUS', axis = 1, inplace = True)
df_bureau_balance.head() 
features_with_small_variance = df_bureau_balance.columns[(df_bureau_balance.std(axis = 0) < .01).values]
print(len(features_with_small_variance))
df_bureau_balance.info(null_counts = True)
df_bureau = pd.read_csv('../input/bureau.csv')
df_bureau.head()
for val in df_bureau['CREDIT_ACTIVE'].unique():
    df_bureau['CREDIT_ACTIVE_' + val] = (df_bureau['CREDIT_ACTIVE'] == val).map(int)
    
df_bureau.drop('CREDIT_ACTIVE', axis = 1, inplace = True)
for val in df_bureau['CREDIT_CURRENCY'].unique():
    df_bureau['CREDIT_CURRENCY_' + val] = (df_bureau['CREDIT_CURRENCY'] == val).map(int)
    
df_bureau.drop('CREDIT_CURRENCY', axis = 1, inplace = True)
for val in df_bureau['CREDIT_TYPE'].unique():
    df_bureau['CREDIT_TYPE_' + val] = (df_bureau['CREDIT_TYPE'] == val).map(int)
    
df_bureau.drop('CREDIT_TYPE', axis = 1, inplace = True)
features_with_small_variance = df_bureau.columns[(df_bureau.std(axis = 0) < .01).values]
print(len(features_with_small_variance))
df_bureau[features_with_small_variance].describe().T
df_bureau.drop(features_with_small_variance, axis = 1, inplace = True)
print(df_bureau.shape)
df_bureau.info(null_counts = True)
for c in df_bureau_balance.drop('SK_ID_BUREAU', axis = 1).columns:
    res = df_bureau_balance.groupby(by = 'SK_ID_BUREAU')[c].mean()
    df_bureau['Balance_' + str(c)] = df_bureau['SK_ID_BUREAU'] \
                                                        .map(lambda x: res[x] if x in res.index else np.nan)
        
df_bureau.head()
features_with_small_variance = df_bureau.columns[(df_bureau.std(axis = 0) < .01).values]
print(len(features_with_small_variance))
df_bureau[features_with_small_variance].describe().T
df_bureau.drop(features_with_small_variance, axis = 1, inplace = True)
print(df_bureau.shape)
df_bureau.info(null_counts = True)
df_installments_payments = pd.read_csv('../input/installments_payments.csv')
df_installments_payments.head()
features_with_small_variance = df_installments_payments.columns[(df_installments_payments.std(axis = 0) < .01).values]
print(len(features_with_small_variance))
df_installments_payments.info(null_counts = True)
df_credit_card_balance = pd.read_csv('../input/credit_card_balance.csv')
df_credit_card_balance.head()
for val in df_credit_card_balance['NAME_CONTRACT_STATUS'].unique():
    df_credit_card_balance['NAME_CONTRACT_STATUS_' + val] = (df_credit_card_balance['NAME_CONTRACT_STATUS'] == val).map(int)
    
df_credit_card_balance.drop('NAME_CONTRACT_STATUS', axis = 1, inplace = True)
features_with_small_variance = df_credit_card_balance.columns[(df_credit_card_balance.std(axis = 0) < .01).values]
print(len(features_with_small_variance))
df_credit_card_balance[features_with_small_variance].describe().T
df_credit_card_balance.drop(features_with_small_variance, axis = 1, inplace = True)
print(df_credit_card_balance.shape)
df_credit_card_balance.info(null_counts = True)
df_POS_CASH_balance = pd.read_csv('../input/POS_CASH_balance.csv')
df_POS_CASH_balance.head()
for val in df_POS_CASH_balance['NAME_CONTRACT_STATUS'].unique():
    df_POS_CASH_balance['NAME_CONTRACT_STATUS_' + val] = (df_POS_CASH_balance['NAME_CONTRACT_STATUS'] == val).map(int)
    
df_POS_CASH_balance.drop('NAME_CONTRACT_STATUS', axis = 1, inplace = True)
features_with_small_variance = df_POS_CASH_balance.columns[(df_POS_CASH_balance.std(axis = 0) < .01).values]
print(len(features_with_small_variance))
df_POS_CASH_balance[features_with_small_variance].describe().T
df_POS_CASH_balance.drop(features_with_small_variance, axis = 1, inplace = True)
print(df_POS_CASH_balance.shape)
df_POS_CASH_balance.info(null_counts = True)
df_previous_application = pd.read_csv('../input/previous_application.csv')
df_previous_application.head()
for c in df_previous_application.columns[df_previous_application.dtypes == 'object']:
    if df_previous_application[c].nunique() == 2:
        d = df_previous_application[c].value_counts()
        df_previous_application[c] = df_previous_application[c].map({d[0]: 0, d[1]: 1})
    else:        
        for val in df_previous_application[c].unique():
            df_previous_application[str(c) + '_' + str(val)] = (df_previous_application[c] == val).map(int)
        df_previous_application.drop(c, axis = 1, inplace = True)
features_with_small_variance = df_previous_application.columns[(df_previous_application.std(axis = 0) < .01).values]
print(len(features_with_small_variance))
df_previous_application[features_with_small_variance].describe().T
df_previous_application.drop(features_with_small_variance, axis = 1, inplace = True)
print(df_previous_application.shape)
df_previous_application.info()
for c in df_installments_payments.drop(['SK_ID_PREV', 'SK_ID_CURR'], axis = 1).columns:
    res = df_installments_payments.groupby(by = 'SK_ID_PREV')[c].mean()
    df_previous_application['IP_' + c] = df_previous_application['SK_ID_PREV'] \
                                                            .map(lambda x: res[x] if x in res.index else np.nan)

df_previous_application.head()
for c in df_credit_card_balance.drop(['SK_ID_PREV', 'SK_ID_CURR'], axis = 1).columns:
    res = df_credit_card_balance.groupby(by = 'SK_ID_PREV')[c].mean()
    df_previous_application['CCB_' + c] = df_previous_application['SK_ID_PREV'] \
                                                            .map(lambda x: res[x] if x in res.index else np.nan)

df_previous_application.head()
for c in df_POS_CASH_balance.drop(['SK_ID_PREV', 'SK_ID_CURR'], axis = 1).columns:
    res = df_POS_CASH_balance.groupby(by = 'SK_ID_PREV')[c].mean()
    df_previous_application['POS_' + c] = df_previous_application['SK_ID_PREV'] \
                                                            .map(lambda x: res[x] if x in res.index else np.nan)

df_previous_application.head()
#for c in df_bureau.drop(['SK_ID_CURR', 'SK_ID_BUREAU'], axis = 1).columns:
#    res = df_bureau.groupby(by = 'SK_ID_CURR')[c].mean()
#    df_application_train['Bureau_' + str(c)] = df_bureau['SK_ID_CURR'] \
#                                                        .map(lambda x: res[x] if x in res.index else np.nan)
#    df_application_test['Bureau_' + str(c)] = df_bureau['SK_ID_CURR'] \
#                                                        .map(lambda x: res[x] if x in res.index else np.nan)
#for c in df_previous_application.drop(['SK_ID_CURR', 'SK_ID_PREV'], axis = 1).columns:
#    res = df_previous_application.groupby(by = 'SK_ID_CURR')[c].mean()
#    df_application_train['Prev_' + str(c)] = df_previous_application['SK_ID_CURR'] \
#                                                        .map(lambda x: res[x] if x in res.index else np.nan)
#    df_application_test['Prev_' + str(c)] = df_previous_application['SK_ID_CURR'] \
#                                                        .map(lambda x: res[x] if x in res.index else np.nan)
#df_application_train.head()
#df_application_test.head()
#print(df_application_train.shape, df_application_test.shape)
#print(len(np.intersect1d(df_application_train.columns, df_application_test.columns)))