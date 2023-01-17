import os # appel système

import numpy as np # manipulations matricielles, un peu d'algèbre linéaire

import pandas as pd # manipulation de tableau, jointure SQL, etc.
df_train = pd.read_csv("../input/train_users_2.csv")

df_train.sample(n=5) # Ne montre que quelques lignes choisie au hasard
df_test = pd.read_csv("../input/test_users.csv")

df_test.sample(n=5)
#On combine les 2 tableaux



df_all = pd.concat((df_train, df_test), axis = 0, ignore_index = True)

# on importe pas l'index car pandas numérote les lignes et on ne veut pas que ça collisione

df_all.head(n=5)
# On supprime la colonne de 1ère réservation qui est embêtante

df_all.drop('date_first_booking', axis = 1, inplace = True)
df_all.sample(n=5)
df_all['date_account_created'] = pd.to_datetime(df_all['date_account_created'], format = '%Y-%m-%d')

df_all['timestamp_first_active'] = pd.to_datetime(df_all['timestamp_first_active'], format = '%Y%m%d%H%M%S')



df_all.sample(n=5)
def remove_age_outliers(x, min_value=15, max_value=90): #operations logiques sur des tableaux

    if np.logical_or(x<=min_value, x>=max_value):

        return np.nan

    else:

        return x

    
df_all['age'] = df_all['age'].apply(lambda x: remove_age_outliers(x) if(not np.isnan(x)) else x)

# Pandas accepte qu'on applique une fonction sur toutes les valeurs d'une ligne ou d'une colonne

# est-ce que naN est superieur ou égal à 90? Comparaison pas toujours possible



df_all['age'].head(n=20)
# on remplace les NaN par -1

df_all['age'].fillna(-1, inplace=True)

df_all.head(n=10)
# L'age est écrit comme n réel ! Conversion en entier.



df_all.age = df_all.age.astype(int)

df_all.age.sample(n=10)
def check_NaN_values_in_df(df):

    for col in df: # col va être chacune des colonnes

        nan_count = df[col].isnull().sum() #nombre de valeurs nulles

        

        if nan_count != 0:

            print(col + " => " + str(nan_count) + " NaN values") #nan_count is int => string
check_NaN_values_in_df(df_all)
df_all['first_affiliate_tracked'].fillna(-1, inplace = True)
check_NaN_values_in_df(df_all)

df_all.sample(n=5)
df_all.drop('timestamp_first_active', axis = 1, inplace = True)

df_all.sample(n=5)
# Il faut faire attention avec ce que l'on supprime. Néanmoins,

# il se peut qu'on retire de grosses informations (patterns)

df_all.drop('language', axis = 1, inplace = True)

df_all.sample(n=5)
df_all = df_all[df_all['date_account_created'] > '2013-02-01']

df_all.sample(n=5)
#creation du directory si nécessaire

if not os.path.exists("output"):

    os.makedirs("output")

    

#exportation en CSV

df_all.to_csv("output/cleaned.csv", sep = ",", index = False)