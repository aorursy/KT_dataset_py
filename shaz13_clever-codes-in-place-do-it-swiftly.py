import numpy as np 
import pandas as pd 
import os
print(os.listdir("../input"))
data = {}
data['application_train'] = pd.read_csv('../input/application_train.csv')
data['POS_CASH_balance'] = pd.read_csv('../input/POS_CASH_balance.csv')
data['bureau_balance'] = pd.read_csv('../input/bureau_balance.csv')
data['previous_application'] = pd.read_csv('../input/previous_application.csv')
data['installments_payments'] = pd.read_csv('../input/installments_payments.csv')
data['credit_card_balance'] = pd.read_csv('../input/credit_card_balance.csv')
data['sample_submission'] = pd.read_csv('../input/sample_submission.csv')
data['application_test'] = pd.read_csv('../input/application_test.csv')
data['bureau'] = pd.read_csv('../input/bureau.csv')

data['application_train'].head()
def mr_inspect(df):
    """Returns a inspection dataframe"""
    print ("Length of dataframe:", len(df))
    inspect_dataframe = pd.DataFrame({'dtype': df.dtypes, 'Unique values': df.nunique() ,
                 'Number of missing values': df.isnull().sum() ,
                  'Percentage missing': (df.isnull().sum() / len(df)) * 100
                 }).sort_values(by='Number of missing values', ascending = False)
    return inspect_dataframe
mr_inspect(data['credit_card_balance'])
def get_num_cols(df):
    """Returns list of columns that are numeric"""
    return list(df.select_dtypes(include=['int']).columns)

def get_cat_cols(df):
    """Returns list of columns that are non-numeric"""
    return list(df.select_dtypes(exclude=['int']).columns)

get_num_cols(data['application_train'])[:15]
get_cat_cols(data['application_train'])[:15]