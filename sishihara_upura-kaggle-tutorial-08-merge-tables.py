import pandas as pd





application_train = pd.read_csv('../input/home-credit-default-risk/application_train.csv')

application_train.head()
bureau = pd.read_csv('../input/home-credit-default-risk/bureau.csv')

bureau.head()
previous_loan_counts = bureau.groupby('SK_ID_CURR', as_index=False)['SK_ID_BUREAU'].count().rename(columns={'SK_ID_BUREAU': 'previous_loan_counts'})

previous_loan_counts.head()
application_train = pd.merge(application_train, previous_loan_counts, on='SK_ID_CURR', how='left')



# 欠損値の処理

application_train['previous_loan_counts'].fillna(0, inplace=True)

application_train.head()