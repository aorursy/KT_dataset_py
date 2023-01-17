import os



import numpy as np

import pandas as pd
df_train = pd.read_csv("../input/train_users_2.csv")

df_train.sample(n=5) #only display a few lines and not the whole dataframe
df_test = pd.read_csv("../input/train_users_2.csv")

df_test.sample(n=5)
#Combine into one dataset

df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)

df_all.head(n=5) #only display a few lines and not the whole dataframe
df_all.drop('date_first_booking', axis=1, inplace=True)

df_all.sample(n=5)
df_all['date_account_created'] = pd.to_datetime(df_all['date_account_created'], format='%Y-%m-%d')

df_all.sample(n=5)
df_all['timestamp_first_active'] = pd.to_datetime(df_all['timestamp_first_active'], format='%Y%m%d')

df_all.sample(n=5)
def remove_age_outliers(x, min_values=15, max_value=90):

    if np.logical_or(x<=min_values, x>=max_value):

        return  np.nan

    else:

           return x
#we create the output directory

if not os.path.exists("output"):

    os.makedirs("output")



#we export to csv

df_all.to_csv("output/cleaned.csv", sep=",", index=False)