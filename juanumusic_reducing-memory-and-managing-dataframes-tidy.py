import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import svm # Support vector Machines

from sklearn import neighbors # K Nearest Neighbors

from sklearn.model_selection import cross_val_score # Cross validation



# Any results you write to the current directory are saved as output.
# Stores the input path to where the csv files are located

INPUT_PATH = '../input/'
# Initialize the dataframes dictonary

dict_dfs = {}



# Read the csvs into the dictonary

dict_dfs['members'] = pd.read_csv(INPUT_PATH + 'members.csv', parse_dates=['registration_init_time','expiration_date'], dtype={'city': np.int8, 'bd': np.int16, 'registered_via': np.int8})

dict_dfs['train'] = pd.read_csv(INPUT_PATH + 'train.csv', dtype={'is_churn' : np.int8})

dict_dfs['predict'] = pd.read_csv(INPUT_PATH + 'sample_submission_zero.csv', dtype={'is_churn' : np.int8})

dict_dfs['transactions'] = pd.read_csv(INPUT_PATH + 'transactions.csv', parse_dates=['transaction_date','membership_expire_date'], dtype={'payment_method_id': np.int8, 'payment_plan_days': np.int16, 'plan_list_price': np.int16, 'actual_amount_paid': np.int16, 'is_auto_renew': np.int8, 'is_cancel': np.int8})

#dict_dfs['user_logs'] = pd.read_csv(INPUT_PATH + 'user_logs.csv') # TOO MUCH MEMORY

def get_memory_usage_datafame():

    "Returns a dataframe with the memory usage of each dataframe."

    

    # Dataframe to store the memory usage

    df_memory_usage = pd.DataFrame(columns=['DataFrame','Memory MB','Records'])



    # For each dataframe

    for key, value in dict_dfs.items():

    

        # Get the memory usage of the dataframe

        mem_usage = value.memory_usage(index=True).sum()

        mem_usage = mem_usage / 1024**2

    

        # Append the memory usage to the result dataframe

        df_memory_usage = df_memory_usage.append({'DataFrame': key, 'Memory MB': mem_usage,'Records': len(value)}, ignore_index=True)

    

    # return the dataframe

    return df_memory_usage
get_memory_usage_datafame()
# In case we run the cell twice

if 'registration_init_time' in dict_dfs['members'].columns:

    

    # Split registration init date into 3 columns

    dict_dfs['members']['registration_init_year'] = dict_dfs['members'].registration_init_time.dt.year.astype(np.int16)

    dict_dfs['members']['registration_init_month'] = dict_dfs['members'].registration_init_time.dt.month.astype(np.int8)

    dict_dfs['members']['registration_init_date'] = dict_dfs['members'].registration_init_time.dt.day.astype(np.int8)

    

    # Drop the registration init date 

    dict_dfs['members'] = dict_dfs['members'].drop('registration_init_time', axis=1)



# In case we run the cell twice

if 'expiration_date' in dict_dfs['members'].columns:

    

    # Split the expiration date into 3 columns

    dict_dfs['members']['expiration_year'] = dict_dfs['members'].expiration_date.dt.year.astype(np.int16)

    dict_dfs['members']['expiration_month'] = dict_dfs['members'].expiration_date.dt.month.astype(np.int8)

    dict_dfs['members']['expiration_date'] = dict_dfs['members'].expiration_date.dt.day.astype(np.int8)

    

    # Drop the expiration date 

    dict_dfs['members'] = dict_dfs['members'].drop('expiration_date', axis=1)
get_memory_usage_datafame()
# In case we run the cell more than once

if 'membership_expire_date' in dict_dfs['transactions']:

    # Split membership_expire_date into 3 columns

    dict_dfs['transactions']['membership_expire_year'] = dict_dfs['transactions'].membership_expire_date.dt.year.astype(np.int16)

    dict_dfs['transactions']['membership_expire_month'] = dict_dfs['transactions'].membership_expire_date.dt.month.astype(np.int8)

    dict_dfs['transactions']['membership_expire_date'] = dict_dfs['transactions'].membership_expire_date.dt.day.astype(np.int8)

    

    # Drop the registration init date 

    dict_dfs['transactions'] = dict_dfs['transactions'].drop('membership_expire_date', axis=1)

    

# In case we run the cell more than once

if 'transaction_date' in dict_dfs['transactions']:

    # Split membership_expire_date into 3 columns

    dict_dfs['transactions']['transaction_year'] = dict_dfs['transactions'].transaction_date.dt.year.astype(np.int16)

    dict_dfs['transactions']['transaction_month'] = dict_dfs['transactions'].transaction_date.dt.month.astype(np.int8)

    dict_dfs['transactions']['transaction_date'] = dict_dfs['transactions'].transaction_date.dt.day.astype(np.int8)

    

    # Drop the registration init date 

    dict_dfs['transactions'] = dict_dfs['transactions'].drop('transaction_date', axis=1)

    
get_memory_usage_datafame()
# Merge members to the train and test dataframes

dict_dfs['train'] = pd.merge(dict_dfs['train'], dict_dfs['members'], on='msno')

dict_dfs['predict'] = pd.merge(dict_dfs['predict'], dict_dfs['members'], on='msno')
# Set the gender values

gender = {'male':1, 'female':2}

# Map the int values to the gender columns of test and predict dataframes

dict_dfs['train'].gender = dict_dfs['train'].gender.map(gender)

dict_dfs['predict'].gender = dict_dfs['predict'].gender.map(gender)



# Set the NaN to 0 and convert the type to int8

dict_dfs['train'].gender = dict_dfs['train'].gender.fillna(0).astype(np.int8)

dict_dfs['predict'].gender = dict_dfs['predict'].gender.fillna(0).astype(np.int8)
get_memory_usage_datafame()
# Get the Y labels from the is_churn column

Y = dict_dfs['train']['is_churn']

# Drop the is_churn column from the train dataframe and store it on the X dataframe

X = dict_dfs['train'].drop(['is_churn','msno'], axis=1)