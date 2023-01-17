import os



import numpy as np

import pandas as pd # ça c'est vraiment...
df_train = pd.read_csv("../input/train_users_2.csv")

df_train.sample(n=5)
df_test = pd.read_csv("../input/test_users.csv")

df_test.sample(n=5)
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)

df_all.head(n=5)
df_all.drop('date_first_booking', axis=1, inplace=True)
df_all.head(n=5)
df_all['date_account_created'] = pd.to_datetime(df_all['date_account_created'], format='%Y-%m-%d')

df_all.head(n=5)
df_all['timestamp_first_active'] = pd.to_datetime(df_all['timestamp_first_active'], format='%Y%m%d%H%M%S')

df_all.head(n=5)
def remove_age_outliers(x, min_value=15, max_value=90):

    if np.logical_or(x<=min_value, x>=max_value):

        return np.nan

    else:

        return x
df_all['age'] = df_all['age'].apply(lambda x: remove_age_outliers(x) if(not np.isnan(x)) else np.nan)
df_all['age'].fillna(-1, inplace=True)
df_all.sample(n=5)
df_all.age = df_all.age.astype(int)

df_all.sample(n=5)
def check_NaN_values_in_df(df):

    for col in df:

        nan_count = df[col].isnull().sum()

        if nan_count != 0:

            print(col + " => " + str(nan_count) + " NaN value(s)")
check_NaN_values_in_df(df_all)
df_all["first_affiliate_tracked"].fillna(-1, inplace=True)

check_NaN_values_in_df(df_all)
df_all = df_all[df_all["date_account_created"] > '2013-02-01']

df_all.sample(n=5)
if not os.path.exists("output"):

    os.makedirs("output")

    

df_all.to_csv("output/cleaned.csv", sep=",", index=False)
from datetime import datetime

import sklearn as sk



df_all = pd.read_csv(

    "output/cleaned.csv", 

    dtype={

        'country_destination': str

    }

)



# We transform again the date column into datetime

df_all['date_account_created'] = pd.to_datetime(df_all['date_account_created'], format='%Y-%m-%d %H:%M:%S')

df_all['timestamp_first_active'] = pd.to_datetime(df_all['timestamp_first_active'], format='%Y-%m-%d %H:%M:%S')



# Check for NaN Values => We must find: country_destination => 62096 NaN Values

check_NaN_values_in_df(df_all) 



df_all.sample(n=5) # Only display a few lines and not the whole dataframe
# Home made One Hot Encoding function

def convert_to_binary(df, column_to_convert):

    categories = list(df[column_to_convert].drop_duplicates())



    for category in categories:

        cat_name = str(category).replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_").replace("-", "").lower()

        col_name = column_to_convert[:5] + '_' + cat_name[:10]

        df[col_name] = 0

        df.loc[(df[column_to_convert] == category), col_name] = 1



    return df

columns_to_convert = [

    'gender', 

    'signup_method', 

    'signup_flow', 

    'language', 

    'affiliate_channel', 

    'affiliate_provider', 

    'first_affiliate_tracked', 

    'signup_app', 

    'first_device_type', 

    'first_browser'

]



# One Hot Encoding

for column in columns_to_convert:

    df_all = convert_to_binary(df=df_all, column_to_convert=column)

    df_all.drop(column, axis=1, inplace=True)

    

df_all.sample(n=5)
# Add new date related fields

df_all['day_account_created'] = df_all['date_account_created'].dt.weekday

df_all['month_account_created'] = df_all['date_account_created'].dt.month

df_all['quarter_account_created'] = df_all['date_account_created'].dt.quarter

df_all['year_account_created'] = df_all['date_account_created'].dt.year

df_all['hour_first_active'] = df_all['timestamp_first_active'].dt.hour

df_all['day_first_active'] = df_all['timestamp_first_active'].dt.weekday

df_all['month_first_active'] = df_all['timestamp_first_active'].dt.month

df_all['quarter_first_active'] = df_all['timestamp_first_active'].dt.quarter

df_all['year_first_active'] = df_all['timestamp_first_active'].dt.year

df_all['created_less_active'] = (df_all['date_account_created'] - df_all['timestamp_first_active']).dt.days



# Drop unnecessary columns

columns_to_drop = ['date_account_created', 'timestamp_first_active', 'date_first_booking', 'country_destination']

for column in columns_to_drop:

    if column in df_all.columns:

        df_all.drop(column, axis=1, inplace=True)



print ("Dataframe Shape:", df_all.shape)

df_all.sample(n=5)
df_sessions = pd.read_csv("../input/sessions.csv")

print ("DF Session Shape:", df_sessions.shape)

df_sessions.head(n=5) # Only display a few lines and not the whole dataframe

# Determine primary device

sessions_device = df_sessions.loc[:, ['user_id', 'device_type', 'secs_elapsed']]

aggregated_lvl1 = sessions_device.groupby(['user_id', 'device_type'], as_index=False, sort=False).aggregate(np.sum)

idx = aggregated_lvl1.groupby(['user_id'], sort=False)['secs_elapsed'].transform(max) == aggregated_lvl1['secs_elapsed']

df_primary = pd.DataFrame(aggregated_lvl1.loc[idx , ['user_id', 'device_type', 'secs_elapsed']])

df_primary.rename(columns = {'device_type':'primary_device', 'secs_elapsed':'primary_secs'}, inplace=True)

df_primary = convert_to_binary(df=df_primary, column_to_convert='primary_device')

df_primary.drop('primary_device', axis=1, inplace=True)



df_primary.sample(n=5)
# Determine Secondary device

remaining = aggregated_lvl1.drop(aggregated_lvl1.index[idx])

idx = remaining.groupby(['user_id'], sort=False)['secs_elapsed'].transform(max) == remaining['secs_elapsed']

df_secondary = pd.DataFrame(remaining.loc[idx , ['user_id', 'device_type', 'secs_elapsed']])

df_secondary.rename(columns = {'device_type':'secondary_device', 'secs_elapsed':'secondary_secs'}, inplace=True)

df_secondary = convert_to_binary(df=df_secondary, column_to_convert='secondary_device')

df_secondary.drop('secondary_device', axis=1, inplace=True)



df_secondary.sample(n=5)
# Count occurrences of value in a column

def convert_to_counts(df, id_col, column_to_convert):

    id_list = df[id_col].drop_duplicates()

    

    df_counts = df.loc[:,[id_col, column_to_convert]]

    df_counts['count'] = 1

    df_counts = df_counts.groupby(by=[id_col, column_to_convert], as_index=False, sort=False).sum()

    

    new_df = df_counts.pivot(index=id_col, columns=column_to_convert, values='count')

    new_df = new_df.fillna(0)

    

    # Rename Columns

    categories = list(df[column_to_convert].drop_duplicates())

    for category in categories:

        cat_name = str(category).replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_").replace("-", "").lower()

        col_name = column_to_convert + '_' + cat_name

        new_df.rename(columns = {category:col_name}, inplace=True)

        

    return new_df
# Aggregate and combine actions taken columns



session_actions = df_sessions.loc[:,['user_id', 'action', 'action_type', 'action_detail']]

columns_to_convert = ['action', 'action_type', 'action_detail']



session_actions = session_actions.fillna('not provided')

first = True



for column in columns_to_convert:

    print("Converting " + column + " column...")

    current_data = convert_to_counts(df=session_actions, id_col='user_id', column_to_convert=column)



    # If first loop, current data becomes existing data, otherwise merge existing and current

    if first:

        first = False

        actions_data = current_data

    else:

        actions_data = pd.concat([actions_data, current_data], axis=1, join='inner')

        

actions_data.sample(n=5)
# Merge device datasets

df_primary.set_index('user_id', inplace=True)

df_secondary.set_index('user_id', inplace=True)

device_data = pd.concat([df_primary, df_secondary], axis=1, join="outer")



# Merge device and actions datasets

combined_results = pd.concat([device_data, actions_data], axis=1, join='outer')

df_sessions = combined_results.fillna(0)



# Merge user and session datasets

df_all.set_index('id', inplace=True)

df_all = pd.concat([df_all, df_sessions], axis=1, join='inner')



df_all.head(n=5)