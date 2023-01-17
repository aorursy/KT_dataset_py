# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
pos_cash_balance = pd.read_csv('../input/POS_CASH_balance.csv')
bureau_balance = pd.read_csv('../input/bureau_balance.csv')
application_train = pd.read_csv('../input/application_train.csv')
previous_application = pd.read_csv('../input/previous_application.csv')
installments_payments = pd.read_csv('../input/installments_payments.csv')
credit_card_balance = pd.read_csv('../input/credit_card_balance.csv')
bureau = pd.read_csv('../input/bureau.csv')
application_test =pd.read_csv('../input/application_test.csv')
#column_description = pd.read_csv('../input/HomeCredit_columns_description.csv')
#printing the number of rows and columns
print('number of rows and coumns in application_test :',application_test.shape,'\nNo of columns having na values',application_test.isna().any().sum(),'\n')
print('number of rows and coumns in pos_cash_balance :',pos_cash_balance.shape,'\nNo of columns having na values',pos_cash_balance.isna().any().sum(),'\n')
print('number of rows and coumns in bureau_balance :',bureau_balance.shape,'\nNo of columns having na values',bureau_balance.isna().any().sum(),'\n')
print('number of rows and coumns in application_train :',application_train.shape,'\nNo of columns having na values',application_train.isna().any().sum(),'\n')
print('number of rows and coumns in previous_application :',previous_application.shape,'\nNo of columns having na values',previous_application.isna().any().sum(),'\n')
print('number of rows and coumns in installments_payments :',installments_payments.shape,'\nNo of columns having na values',installments_payments.isna().any().sum(),'\n')
print('number of rows and coumns in credit_card_balance :',credit_card_balance.shape,'\nNo of columns having na values',credit_card_balance.isna().any().sum(),'\n')
print('number of rows and coumns in bureau',bureau.shape,'\nNo of columns having na values',bureau.isna().any().sum(),'\n')
#print('application_train')
pd.DataFrame(application_train.dtypes.value_counts())
pd.DataFrame(application_test.dtypes.value_counts())
pd.DataFrame(pos_cash_balance.dtypes.value_counts())
pd.DataFrame(bureau_balance.dtypes.value_counts())
pd.DataFrame(previous_application.dtypes.value_counts())
pd.DataFrame(installments_payments.dtypes.value_counts())
pd.DataFrame(bureau.dtypes.value_counts())
pd.DataFrame(credit_card_balance.dtypes.value_counts())
target_correlation = application_train.corr()['TARGET']
filtered_attributes = pd.DataFrame(target_correlation[np.abs(target_correlation)>0.01])

application_train_filtered = application_train[['SK_ID_CURR']+list(filtered_attributes.index)]
application_train_filtered.head()
application_train.head(2)
bureau.head(2)
print(len(application_train_filtered['SK_ID_CURR'].unique()))
print(len(bureau['SK_ID_CURR'].unique()))
bureau_balance.head(2)
bureau[bureau['SK_ID_CURR']==application_train.iloc[0]['SK_ID_CURR']]
bureau_balance[bureau_balance['SK_ID_BUREAU']==bureau[bureau['SK_ID_CURR']==application_train.iloc[0]['SK_ID_CURR']].iloc[0]['SK_ID_BUREAU']]
print('total elements in application_train ',len(application_train))
print('unique SK_ID_CURR in application_train ',len(application_train['SK_ID_CURR'].unique()))
print('unique SK_ID_CURR in bureau ',len(bureau['SK_ID_CURR'].unique()))
print('unique SK_ID_CURR in bureau_balance ',len(bureau_balance['SK_ID_BUREAU'].unique()))
num = pd.Series(application_train['SK_ID_CURR'].unique()).isin(pd.Series(bureau['SK_ID_CURR'].unique())).sum()
print("Total number of SK_ID_CURR in bureau that are in application_train: ",num)
print("So the number of missing id's in bureau are :")

previous_application.head(2)
print(len(previous_application['SK_ID_CURR'].unique()))
previous_application[previous_application['SK_ID_CURR']==application_train.iloc[0]['SK_ID_CURR']]
pos_cash_balance[pos_cash_balance['SK_ID_PREV']==1038818]
installments_payments[installments_payments['SK_ID_PREV']==1038818]
credit_card_balance[credit_card_balance['SK_ID_PREV']==1038818]
credit_card_balance.head()
