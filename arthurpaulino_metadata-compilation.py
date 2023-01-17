import pandas as pd

pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 1000)

application_train = pd.read_csv('../input/application_train.csv')
application_test = pd.read_csv('../input/application_test.csv')
application_test['TARGET'] = 2

columns = list(application_train.columns)
columns.remove('SK_ID_CURR')
columns.remove('TARGET')
columns = ['SK_ID_CURR'] + columns + ['TARGET']

application_train = application_train[columns]
application_test = application_test[columns]

application = pd.concat([application_train, application_test], ignore_index=True)
application = application.sort_values('SK_ID_CURR').reset_index(drop=True)

bureau = pd.read_csv('../input/bureau.csv')
bureau_balance = pd.read_csv('../input/bureau_balance.csv')
bureau = bureau.sort_values('SK_ID_CURR').reset_index(drop=True)

credit_card_balance = pd.read_csv('../input/credit_card_balance.csv')
installments_payments = pd.read_csv('../input/installments_payments.csv')
pos_cash_balance = pd.read_csv('../input/POS_CASH_balance.csv')
previous_application = pd.read_csv('../input/previous_application.csv')
credit_card_balance = credit_card_balance.sort_values(['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE']).reset_index(drop=True)
installments_payments = installments_payments.sort_values(['SK_ID_CURR', 'SK_ID_PREV', 'NUM_INSTALMENT_NUMBER']).reset_index(drop=True)
pos_cash_balance = pos_cash_balance.sort_values(['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE']).reset_index(drop=True)
previous_application = previous_application.sort_values(['SK_ID_CURR', 'SK_ID_PREV', 'DAYS_DECISION']).reset_index(drop=True)
print('Train size: {}'.format(application_train.shape[0]))
print('Test size: {}'.format(application_test.shape[0]))
print('Total # of SK_ID_CURR: {}'.format(application['SK_ID_CURR'].nunique()))
application.head(30)
df = application
missing_data = pd.DataFrame(df.isnull().sum()/df.isnull().count()*100).rename(columns={0:'missing data %'})
dtype = pd.DataFrame({'dtype':[df[column].dtype for column in df.columns]}, index=df.columns)
pd.concat([missing_data, dtype], axis=1, sort=True).sort_values('missing data %', ascending = False)
print('Total # of SK_ID_CURR: {}'.format(bureau['SK_ID_CURR'].nunique()))
print('Total # of SK_ID_BUREAU: {}'.format(bureau['SK_ID_BUREAU'].nunique()))
bureau.head(30)
df = bureau
missing_data = pd.DataFrame(df.isnull().sum()/df.isnull().count()*100).rename(columns={0:'missing data %'})
dtype = pd.DataFrame({'dtype':[df[column].dtype for column in df.columns]}, index=df.columns)
pd.concat([missing_data, dtype], axis=1, sort=True).sort_values('missing data %', ascending = False)
bureau_balance = bureau_balance.merge(bureau[['SK_ID_CURR', 'SK_ID_BUREAU']].drop_duplicates(subset=['SK_ID_BUREAU']),
                                      on='SK_ID_BUREAU',
                                      how='inner')

print('Total # of SK_ID_CURR: {}'.format(bureau_balance['SK_ID_CURR'].nunique()))
print('Total # of SK_ID_BUREAU: {}'.format(bureau_balance['SK_ID_BUREAU'].nunique()))
bureau_balance.head(30)
df = bureau_balance
missing_data = pd.DataFrame((df.isnull().sum()/df.isnull().count()*100).sort_values(ascending = False)).rename(columns={0:'missing data %'})
dtype = pd.DataFrame({'dtype':[df[column].dtype for column in df.columns]}, index=df.columns)
pd.concat([missing_data, dtype], axis=1, sort=True)
print('Total # of SK_ID_CURR: {}'.format(len(previous_application['SK_ID_CURR'].unique())))
print('Total # of SK_ID_PREV: {}'.format(len(previous_application['SK_ID_PREV'].unique())))
previous_application.head(30)
df = previous_application
missing_data = pd.DataFrame(df.isnull().sum()/df.isnull().count()*100).rename(columns={0:'missing data %'})
dtype = pd.DataFrame({'dtype':[df[column].dtype for column in df.columns]}, index=df.columns)
pd.concat([missing_data, dtype], axis=1, sort=True).sort_values('missing data %', ascending = False)
print('Total # of SK_ID_CURR: {}'.format(pos_cash_balance['SK_ID_CURR'].nunique()))
print('Total # of SK_ID_PREV: {}'.format(pos_cash_balance['SK_ID_PREV'].nunique()))
pos_cash_balance.head(30)
df = pos_cash_balance
missing_data = pd.DataFrame(df.isnull().sum()/df.isnull().count()*100).rename(columns={0:'missing data %'})
dtype = pd.DataFrame({'dtype':[df[column].dtype for column in df.columns]}, index=df.columns)
pd.concat([missing_data, dtype], axis=1, sort=True).sort_values('missing data %', ascending = False)
print('Total # of SK_ID_CURR: {}'.format(len(installments_payments['SK_ID_CURR'].unique())))
print('Total # of SK_ID_PREV: {}'.format(len(installments_payments['SK_ID_PREV'].unique())))
installments_payments.head(30)
df = installments_payments
missing_data = pd.DataFrame(df.isnull().sum()/df.isnull().count()*100).rename(columns={0:'missing data %'})
dtype = pd.DataFrame({'dtype':[df[column].dtype for column in df.columns]}, index=df.columns)
pd.concat([missing_data, dtype], axis=1, sort=True).sort_values('missing data %', ascending = False)
print('Total # of SK_ID_CURR: {}'.format(len(credit_card_balance['SK_ID_CURR'].unique())))
print('Total # of SK_ID_PREV: {}'.format(len(credit_card_balance['SK_ID_PREV'].unique())))
credit_card_balance.head(30)
df = credit_card_balance
missing_data = pd.DataFrame(df.isnull().sum()/df.isnull().count()*100).rename(columns={0:'missing data %'})
dtype = pd.DataFrame({'dtype':[df[column].dtype for column in df.columns]}, index=df.columns)
pd.concat([missing_data, dtype], axis=1, sort=True).sort_values('missing data %', ascending = False)