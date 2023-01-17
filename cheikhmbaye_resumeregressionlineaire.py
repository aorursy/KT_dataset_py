



import os

#donnes

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import pandas as pd

import numpy as np

#random forest pour regressor

from sklearn.ensemble import RandomForestRegressor

#Pour le split feature et target

from sklearn.model_selection import train_test_split

#Pour regression au lieu de accuracy c'est mean_square_error

from sklearn.metrics import mean_squared_error

#Ne pas afficher le warning lors du fit par exemple

#Import pour la cross_validation

from sklearn.model_selection  import cross_val_score

#import random forest pour regression

from sklearn.ensemble import RandomForestRegressor

import warnings

warnings.filterwarnings('ignore')
# Fichier de train

X_train = pd.read_csv("../input/train.csv")

#Fichier de test

X_test = pd.read_csv("../input/test.csv")
submission = pd.read_csv("../input/sampleSubmission.csv")

submission["count"] = 195
#Convertir notre fichier en csv

submission.to_csv('submission.csv', index=False)
X_test.columns
## Definition foncyion features et target



def split_dataset(df, features, target='count'):

    X = df[features]

    y = df[target]

    return X, y

## Fonction split date

def date_split(df_train, df_test, date='datetime'):

    ##Traitement_df_train

    cols=df_train[date]

    date_cols=pd.to_datetime(cols)

    df_train['year'] = date_cols.dt.year

    df_train['month'] = date_cols.dt.month

    df_train['day'] = date_cols.dt.day

    df_train['hour'] = date_cols.dt.hour

    df_train['minute'] = date_cols.dt.minute

    df_train['second'] = date_cols.dt.second

    df_train = df_train.drop(['datetime'], axis=1)

    ##Traitement_df_test

    cols2=df_test[date]

    date_cols2=pd.to_datetime(cols2)

    df_test['year'] = date_cols2.dt.year

    df_test['month'] = date_cols2.dt.month

    df_test['day'] = date_cols2.dt.day

    df_test['hour'] = date_cols2.dt.hour

    df_test['minute'] = date_cols2.dt.minute

    df_test['second'] = date_cols2.dt.second

    df_test = df_test.drop(['datetime'], axis=1)

    return df_train, df_test

     

    

    
#Definition de X_train et X_test avec les memes columns

X_train, X_test = date_split(X_train, X_test)
#### definissons nos features numbers ( features qui sont que des nombres)

##ATTENTION: Avoir les mm colonnes avec le dataset de test

# Fonction de recuperation du meme type de colonne

def Get_cols(df, features_test=X_test.columns):

    ############################POUR X_TRAIN########################

    #Renvoyer features de X_test à  X_train

    X_train_features = df[features_test]

    return  X_train_features



#Appel de la fonction pour avoir le meme nombre de columns:

X_trainGet_cols = Get_cols(X_test)



numbers = X_trainGet_cols.select_dtypes(np.number)



numbers.head()
X_train.head()
##Definition features and target



X_train_features, y_train_target = split_dataset(X_train, features=numbers.columns)



X_train_features, y_train_target
##################Cross Validation

## random forest regressor

#Import Random Forest pour regressor

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()

##Cross Validation

score = -cross_val_score(rf, X_train_features, y_train_target, cv=5, scoring='neg_mean_squared_error')
score.mean()
###############  FIT entrainer tout le set d'entrainement

rf.fit(X_train_features, y_train_target)
## Predict sur le train

y_train_pred = rf.predict(X_train_features)
###### Predict sur le test

## Voir si on a le meme nombre de columns dans test et dans train

y_test_pred = rf.predict(X_test)

mean_train = mean_squared_error( y_train_target, y_train_pred)

#mean_test = mean_squared_error(y_test, y_test_pred)

mean_train

#, mean_test
submission = pd.read_csv("../input/sampleSubmission.csv")

submission["count"] = y_test_pred

#Convertir notre fichier en csv

submission.to_csv('submission.csv', index=False)
#!rm submission.csv
submission.head(3)