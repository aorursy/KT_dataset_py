import os

import numpy as np

import pandas as pd
df_train=pd.read_csv("../input/train_users_2.csv")#chargement donnes d'entrainement

df_train.head(n=5) #Only first lines
df_train=pd.read_csv("../input/train_users_2.csv")

df_train.sample(n=5) #Only few lines
df_test=pd.read_csv("../input/test_users.csv")#chargement donnes de tests

df_test.sample(n=5) 
#regroupe les deux tableaux ignore index permet de pas avoir deux fois les numeros de lignes

df_all=pd.concat((df_train, df_test), axis=0, ignore_index=True)

df_all.head(n=5)
#supprime la colonne premiere date de resa

df_all.drop('date_first_booking',axis=1,inplace=True)
df_all.sample(n=5)
df_all['date_account_created']=pd.to_datetime(df_all['date_account_created'],format='%Y-%m-%d')

df_all.sample(n=5)
#modifie le format de la date et du temps pour unifier

df_all['timestamp_first_active']=pd.to_datetime(df_all['timestamp_first_active'], format='%Y%m%d%H%M%S')
df_all.sample(n=5)
#enleve les ages absurdes

def remove_age_outliers(x,min_value=15, max_value=90):

    if np.logical_or(x<=min_value,x>=max_value):

        return np.nan

    else:

        return x
#si x est un nan on le renvoie sinon on applique la fonction

df_all['age'].apply(lambda x: remove_age_outliers(x) if (not np.isnan(x)) else x)
df_all['age'].fillna(-1,inplace=True) #met -1 au lieu de nan fillna modifie directemetn le dataframe en destructif
df_all.sample(n=5)
df_all.age=df_all.age.astype(int) #transforme le type de age en int

df_all.sample(n=5)
#combien de valeur nan il reste dans la tableau

def check_NaN_Values_in_df(df):

    for col in df:

        nan_count = df[col].isnull().sum()

        

        if nan_count !=0:

            print(col +"=>"+str(nan_count)+" NaN values")
check_NaN_Values_in_df(df_all)
df_all['first_affiliate_tracked'].fillna(-1,inplace=True)
check_NaN_Values_in_df(df_all)

df_all.sample(5)
df_all.drop('timestamp_first_active',axis=1,inplace=True)#car meme date entre crea et timestamp

df_all.sample(n=5)
df_all.drop('language',axis=1,inplace=True)#car tous le monde parle anglais attention des fois pas judicieux

df_all.sample(n=5)
df_all = df_all[df_all['date_account_created'] > '2013-02-01']

df_all.sample(n=5)

if not os.path.exists("output"):

    os.makedirs("output")

    

df_all.to_csv("output/cleaned.csv",sep=',',index=False)
df_all.sample(n=5)