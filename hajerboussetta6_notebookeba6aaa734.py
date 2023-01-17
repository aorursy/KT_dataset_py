import os



import numpy as np

import pandas as pd
df_train = pd.read_csv("../input/train_users_2.csv")

df_train.sample(n=5) # Only display a few lines and not the whole dataframe


df_test = pd.read_csv("../input/test_users.csv")

df_test.sample(n=5) # Only display a few lines and not the whole dataframe
# Combine into one dataset

df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)

df_all.head(n=5) # Only display a few lines and not the whole dataframe
df_all['date_account_created'] = pd.to_datetime(df_all['date_account_created'], format='%Y-%m-%d')

df_all['timestamp_first_active'] = pd.to_datetime(df_all['timestamp_first_active'], format='%Y%m%d%H%M%S')



df_all.head(n=5) # Only display a few lines and not the whole dataframe
# Remove date_first_booking column

df_all.drop('date_first_booking', axis=1, inplace=True)


# Remove outliers function

def remove_outlier(x, min_val=15, max_val=90):

    if np.isnan(x):

        return np.nan

    elif np.logical_or(x<=min_val, x>=max_val):

        return np.nan

    else:

        return x
# Fixing age column

df_all['age'] = df_all['age'].apply(lambda x: remove_outlier(x, min_val=15, max_val=90))

df_all['age'].fillna(-1, inplace=True)



df_all.sample(n=5)
df_all.age = df_all.age.astype(int)

df_all.sample(n=5)
def check_NaN_Values_in_df(df):

    # searching for NaN values is all the columns

    for col in df:

        nan_count = df[col].isnull().sum()



        if nan_count != 0:

            print (col + " => "+  str(nan_count) + " NaN Values")

            

check_NaN_Values_in_df(df_all)
df_all['first_affiliate_tracked'].fillna(-1, inplace=True)



# We check there is no more NaN except in country_destination column

check_NaN_Values_in_df(df_all) 



df_all.sample(n=5)
df_all = df_all[df_all['date_account_created'] > '2013-02-01']

df_all.sample(n=5)
# We create the output directory if necessary

if not os.path.exists("output"):

    os.makedirs("output")

    

# We export to csv

df_all.to_csv("output/cleaned.csv", sep=',', index=False)