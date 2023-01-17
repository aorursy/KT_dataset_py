import os

import numpy as np

import pandas as pd
#definition d'un dataframe

df_train = pd.read_csv("../input/train_users_2.csv")

df_train.sample(n=5) 

#df_train.head(n=5) 
#charge les data de test

#pr traitement simult des data de tests et de train

df_test = pd.read_csv("../input/test_users.csv")

df_test.sample(n=5) 
df_all = pd.concat((df_train,df_test),axis=0,ignore_index=True)

df_all.head(n=5)
df_all.drop('date_first_booking',axis=1,inplace=True)# on supprime la colonne

df_all.sample(n=5)
#clean format des dates

df_all['date_account_created'] = pd.to_datetime(df_all['date_account_created'], format='%Y-%m-%d')

df_all.sample(n=5)
#format du time stamp

df_all['timestamp_first_active'] = pd.to_datetime(df_all['timestamp_first_active'], format='%Y%m%d%H%M%S')

df_all.sample(n=5)
#suppression des data outliers (résa entre 0 et 15 ans par exemple) (façon simple sans se compliquer la vie)

def remove_age_outliers(x, min_value=15, max_value=90):

    if np.logical_or(x<=min_value, x>=max_value):

        return np.nan

    else:

        return x

    
#df_all['age'].apply(lambda x:   remove_age_outliers(x)) #crash en python 2, on peut comparer un nan avec un num

df_all['age']=df_all['age'].apply(lambda x:   remove_age_outliers(x) if(not np.isnan(x))else x)

df_all['age'].fillna(-1, inplace=True) #fonctionne ici, mais pas forcément pour un autre projet

df_all.sample(n=5)
#conversion age en entier

df_all.age = df_all.age.astype(int)  #equivalent à df_all['age']

df_all.sample(n=5)
def check_Nan_Values_in_df(df):

    for col in df:

        nan_count = df[col].isnull().sum()

        

        if nan_count != 0:

            print(col + "=>"+str(nan_count)+ " Nan Values")
check_Nan_Values_in_df(df_all)
#pas normal d'avoir des Nan sur first affiliate tracked

df_all['first_affiliate_tracked'].fillna(-1, inplace=True)

check_Nan_Values_in_df(df_all)

df_all.sample(n=5)
#on dégage le time stamp car redondant avec la date et heure min sec inutile (redondance 99% du temps)

df_all.drop('timestamp_first_active',axis=1, inplace=True)

#on dégage la langue, pour essayer

df_all.drop('language',axis=1, inplace=True)

df_all.sample(n=5)
#on dégage ceux avant février 2013 (retrait des early outliers)

#on pourrait laisser janvier, potentiellement pour capter les effets saisonnniers

df_all = df_all[df_all['date_account_created']>'2013-02-01']

df_all.sample(n=5)

#enregistrement du nouveau csv propre

if not os.path.exists("output"):

    os.makedirs("output")

    

df_all.to_csv("output/cleaned.csv", sep=',', index=False)