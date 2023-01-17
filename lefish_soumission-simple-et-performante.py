import datetime

import numpy as np

import os

import pandas as pd

from lightgbm import LGBMRegressor

from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
# Import des données

def ImportData():

    # donnes app

    path = '/kaggle/input/hackathon-data-science-avisia/train_booking.csv'

    train = pd.read_csv(path)

    # donnees val

    path = '/kaggle/input/hackathon-data-science-avisia/test_booking.csv'

    val = pd.read_csv(val)

    

    return train, val
# Traitement de la cible et des ID

def Cible(train, val, cible):

    # Id

    train = train.drop(columns = ['id'])

    id_val = val['id']

    val = val.drop(columns = ['id'])

    

    # Cible

    y_train = train[cible]

    train = train.drop(columns = [cible])

    

    # Suppression lignes où la cible est vide

    train = train[~(y_train.isnull())].reset_index(drop = True)

    y_train = y_train[~(y_train.isnull())]



    return train, val, y_train, id_val
# Convertir en date les dates stockées sous forme de texte

def TextToDate(train, val, list_col):

    for col in list_col:

        train[col] = pd.to_datetime(train[col].str[:10])

        val[col] = pd.to_datetime(val[col].str[:10])

    return(train, val)
# Autres traitements spécifiques sur les variables

def TreatVariables(train, val, dates, split):

    # ID sont totalement disjoints entre train et val donc suppression

    train = train.drop(columns = ['host_id'])

    val = val.drop(columns = ['host_id'])

    

    # Dates : conversion en année, mois, jour, heure, ancienneté

    for col in dates:

        train[col + '_an'] = train[col].dt.year

        val[col + '_an'] = val[col].dt.year

        train[col + '_mois'] = train[col].dt.month

        val[col + '_mois'] = val[col].dt.month

        train[col + '_jour'] = train[col].dt.day

        val[col + '_jour'] = val[col].dt.day

        train[col + '_heure'] = train[col].dt.hour

        val[col + '_heure'] = val[col].dt.hour

        train[col + '_ecart'] = (datetime.datetime.now() - train[col]).dt.days

        val[col + '_ecart'] = (datetime.datetime.now() - val[col]).dt.days

        train = train.drop(columns = [col])

        val = val.drop(columns = [col])

    

    # Parse colonnes texte

    for var in split:

        # nb d'equipements

        train[var + '_nb'] = train[var].map(lambda row: len(str(row).split(',')))

        val[var + '_nb'] = val[var].map(lambda row: len(str(row).split(',')))

        

        # ensemble des modalites de la variable string à spliter

        modalites = list(train[var].astype(str).to_numpy().flat)

        modalites = list(set(','.join(modalites).split(','))) # rendre unique

        

        # encoding

        for modalite in modalites:

            col = var + '_' + ''.join(e for e in modalite if e.isalnum())

            train[col] = train[var].str.contains(modalite).astype(float)

            val[col] = val[var].str.contains(modalite).astype(float)

        

        train = train.drop(columns = [var])

        val = val.drop(columns = [var])

    

    return train, val
# Suppression variables pas utilisées

def Remove(train, val):

    # Suppression variables texte

    texte = ['name','description','neighborhood_overview','notes','transit']

    texte += ['access','interaction','house_rules']

    train = train.drop(columns = texte)

    val = val.drop(columns = texte)

    

    # Suppression variables geo

    geo = ['latitude_booking', 'longitude_booking', 'geolocation', 'geopoint_announce']

    train = train.drop(columns = geo)

    val = val.drop(columns = geo)

    

    return train, val
# Encodage One Hot (binaire)

def OneHot(train, val):

    # Variables catégorielles

    categorielles = train.dtypes[train.dtypes == 'object'].index.tolist()

    

    # Boucle sur les variables

    for var in categorielles:

        

        # Boucle sur les modalités et leur fréquence

        for modalite, pct in (train[var].value_counts(dropna = False) / train.shape[0]).iteritems():

            

            # Si effectif suffisant et modalité dans l'ech de val

            if (pct >= 0.025) & (modalite in val[var].unique()):

                # Test nullité

                if modalite == modalite:

                    col = var + '_' + ''.join(e for e in modalite if e.isalnum() and e not in ['É', 'é', 'ô'])

                    train[col] = (train[var] == modalite).astype(float)

                    val[col] = (val[var].astype(str) == modalite).astype(float)

                    

                else:

                    col = var + '_' + str(mod)

                    train[col] = train[var].isnull().astype(float)

                    val[col] = val[var].isnull().astype(float)

    

    # Suppression variables categorielles

    train = train.drop(columns = categorielles)

    val = val.drop(columns = categorielles)

    

    return train, val
%%time

# Import data & pretraitements

print('---    1. Import Data    ---\n')

train_raw, val_raw = ImportData()



print('---   2. Pretraitement   ---')

print('---       Cible & id     ---')

train, val, y_train, id_val = Cible(train_raw, val_raw, 'price')



print('---    Texte vers Date   ---')

train, val = TextToDate(train, val, ['host_since', 'first_review', 'last_review'])



print('---   Feat engineering   ---')

train, val = TreatVariables(train, val,

                            ['host_since', 'first_review', 'last_review'],

                            ['host_verifications', 'amenities'])



print('---     Col inutiles     ---')

train, val = Remove(train, val)



print('---   One Hot Encoding   ---')

train, val = OneHot(train, val)



print('\n---------------------------')

print('Done.')
# Fit du modele

lgbm_v1 = LGBMRegressor(num_leaves = 10, learning_rate = 0.1, n_estimators = 1000,

                        colsample_bytree = 0.2, subsample = 0.2)

lgbm_v1.fit(train, y_train)
# Prédiction

predict_v1 = lgbm_v1.predict(val)



# Soumission

sub_v1 = pd.DataFrame()

sub_v1['Id'] = id_val

sub_v1['Predicted'] = predict_v1.tolist()

sub_v1.to_csv('soumission_v1.csv', index = False, sep = ',', decimal = '.')