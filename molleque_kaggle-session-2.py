import os



import numpy as np

import pandas as pd

df_train = pd.read_csv("../input/train_users_2.csv")

df_train.sample(n=5)





#Quoi qu'est ce que c'est qu'un pipeline?

#- C'est un tube qui enchaine les traitement de données, étape par étape. Il exécute la routine que nous créons nous aussi étape par étape.

df_test = pd.read_csv("../input/test_users.csv")

df_test.sample(n=5)

df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)

df_all.head(n=5)

#concatene 2 tableaux l'un au dessus de l'autre en ignorant l'index
#supprimer la cellule 1ere reseration

df_all.drop('date_first_booking',axis=1, inplace=True)

df_all.sample(n=5)
df_all['date_account_created'] = pd.to_datetime(df_all['date_account_created'], format = '%Y-%m-%d')
df_all['timestamp_first_active'] = pd.to_datetime(df_all['timestamp_first_active'], format = '%Y%m%d%H%M%S')
df_all.sample(n=5)

def remove_age_outliers(x, min_value=15, max_value=90):

    if np.logical_or(x<=min_value, x>=max_value):

        return np.nan

    else:

        return x
#fonction lambda

#lambda x: True if(X==0) else False

#              applique a toutes les lignes du tableau

#               |

#              \/

df_all['age'] = df_all['age'].apply(lambda x: remove_age_outliers(x) if(not np.isnan(x)) else x)

df_all['age'].fillna(-1, inplace=True)

df_all['age'].sample(n=5)
df_all.age = df_all.age.astype(int)

df_all['age'].sample(n=5)
def check_NaN_Values_in_df(df):

    for col in df:

        nan_count = df[col].isnull().sum()

        

        if nan_count != 0:

            print (col+" -> " + str(nan_count) + "NaN Values")

        #for x in df
check_NaN_Values_in_df(df_all)

df_all['first_affiliate_tracked'].fillna(-1, inplace=True)
check_NaN_Values_in_df(df_all)

df_all.sample(n=5)
df_all.drop('language',axis=1, inplace=True)

df_all.sample(n=5)
df_all = df_all[df_all['date_account_created'] > '2013-02-01']

df_all.sample(n=5)
if  not os.path.exists("output"):

    os.makedirs("ouput")

    

df_all.to_csv("output/cleanned.csv", sep=',', index=False)