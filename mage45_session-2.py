import os



import numpy as np

import pandas as pd



print(":)")
df_train = pd.read_csv("../input/train_users_2.csv")

df_train.sample(n=5)
df_test = pd.read_csv("../input/test_users.csv")

df_train.sample(n=5)
df_all = pd.concat((df_train, df_test), axis = 0, ignore_index = True)
df_all.sample(n=20)
# df_all.query('country_destination != "NaN"')



df_all.drop('date_first_booking', axis = 1, inplace = True)
df_all.sample(n=5)
df_all['date_account_created'] = pd.to_datetime(df_all['date_account_created'], format="%Y-%m-%d")
df_all['timestamp_first_active'] = pd.to_datetime(df_all['timestamp_first_active'], format="%Y%m%d%H%M%S")
df_all.sample(n=5)
def suppr_ages_incorrects(x, min_value=15, max_value=105):

    if np.logical_or(x<= min_value, x>=max_value):

        return np.nan

    else:

        return x
df_all['age'] = df_all['age'].apply(lambda x: suppr_ages_incorrects(x, 15, 100))



# df_all.query('age > 100')
df_all.age.fillna(-1, inplace = True)



df_all.age = df_all.age.astype(int)
df_all.sample(n=5)
def NaN_Values_in_df(df):

    for col in df:

        nb_nan = df[col].isnull().sum()

        if nb_nan != 0:

            print(col + " => " + str(nb_nan) + "NaNs")
NaN_Values_in_df(df_all)
df_all.first_affiliate_tracked.fillna(-1, inplace = True)
df_all.drop('timestamp_first_active', axis=1, inplace=True)
df_all.drop('language', axis=1, inplace=True)
df_all = df_all[df_all['date_account_created'] > '2013-01-01']

df_all.sample(n=5)
df_all.count()
if not os.path.exists('output'):

    os.makedirs('output')



df_all.to_csv("output/cleaned.csv", sep=',', index=False)