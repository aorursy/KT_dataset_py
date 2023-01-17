import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
!pip install featuretools
import featuretools as ft
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

es = ft.EntitySet("clients")
i=11
app_train = pd.read_csv('../input/application_train.csv').sort_values('SK_ID_CURR').reset_index(drop = True).loc[:i, :]
app_test = pd.read_csv('../input/application_test.csv').sort_values('SK_ID_CURR').reset_index(drop = True).loc[:i, :]
bureau = pd.read_csv('../input/bureau.csv').sort_values(['SK_ID_CURR', 'SK_ID_BUREAU']).reset_index(drop = True).loc[:i, :]
bureau_balance = pd.read_csv('../input/bureau_balance.csv').sort_values('SK_ID_BUREAU').reset_index(drop = True).loc[:i, :]
cash = pd.read_csv('../input/POS_CASH_balance.csv').sort_values(['SK_ID_CURR', 'SK_ID_PREV']).reset_index(drop = True).loc[:i, :]
credit = pd.read_csv('../input/credit_card_balance.csv').sort_values(['SK_ID_CURR', 'SK_ID_PREV']).reset_index(drop = True).loc[:i, :]
previous = pd.read_csv('../input/previous_application.csv').sort_values(['SK_ID_CURR', 'SK_ID_PREV']).reset_index(drop = True).loc[:i, :]
installments = pd.read_csv('../input/installments_payments.csv').sort_values(['SK_ID_CURR', 'SK_ID_PREV']).reset_index(drop = True).loc[:i, :]
app_train['set'] = 'train'
app_test['set'] = 'test'
app_test["TARGET"] = np.nan

app = app_train.append(app_test, ignore_index = True, sort=True)
app_train['set'] = 'train'
app_test['set'] = 'test'
app_test["TARGET"] = np.nan

app = app_train.append(app_test, ignore_index = True, sort=True)

es = es.entity_from_dataframe(entity_id = 'app', dataframe = app, index = 'SK_ID_CURR')
es = es.entity_from_dataframe(entity_id = 'bureau', dataframe = bureau, index = 'SK_ID_BUREAU')
es = es.entity_from_dataframe(entity_id = 'previous', dataframe = previous, index = 'SK_ID_PREV')

es = es.entity_from_dataframe(entity_id = 'bureau_balance', dataframe = bureau_balance, make_index = True, index = 'bureaubalance_index')
es = es.entity_from_dataframe(entity_id = 'cash', dataframe = cash, make_index = True, index = 'cash_index')
es = es.entity_from_dataframe(entity_id = 'installments', dataframe = installments, make_index = True, index = 'installments_index')
es = es.entity_from_dataframe(entity_id = 'credit', dataframe = credit,make_index = True, index = 'credit_index')
# Relationship between app and bureau
r_app_bureau = ft.Relationship(es['app']['SK_ID_CURR'], es['bureau']['SK_ID_CURR'])

# Relationship between bureau and bureau balance
r_bureau_balance = ft.Relationship(es['bureau']['SK_ID_BUREAU'], es['bureau_balance']['SK_ID_BUREAU'])

# Relationship between current app and previous apps
r_app_previous = ft.Relationship(es['app']['SK_ID_CURR'], es['previous']['SK_ID_CURR'])

# Relationships between previous apps and cash, installments, and credit
r_previous_cash = ft.Relationship(es['previous']['SK_ID_PREV'], es['cash']['SK_ID_PREV'])
r_previous_installments = ft.Relationship(es['previous']['SK_ID_PREV'], es['installments']['SK_ID_PREV'])
r_previous_credit = ft.Relationship(es['previous']['SK_ID_PREV'], es['credit']['SK_ID_PREV'])
# Add in the defined relationships
es = es.add_relationships([r_app_bureau, r_bureau_balance, r_app_previous,
                           r_previous_cash, r_previous_installments, r_previous_credit])
# Print out the EntitySet
es