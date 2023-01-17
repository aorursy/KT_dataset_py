# Draw inline

%matplotlib inline



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import datetime



# Set figure aesthetics

sns.set_style("white", {'ytick.major.size': 10.0})

sns.set_context("poster", font_scale=1.1)
# Cargue los datos en DataFrames 

ruta =  '../input/' 

train_users = pd.read_csv (ruta +  'train_users_2.csv' )

test_users = pd.read_csv (ruta +  'test_users.csv' )

sessions = pd.read_csv (ruta +  'sessions.csv' )

countries = pd.read_csv (ruta +  'countries.csv' )

age_gender = pd.read_csv (ruta +  'age_gender_bkts.csv' )
print("Tenemos", train_users.shape[0], "registros en el set de entrenamiento y", 

      test_users.shape[0], "en el set de pruebas.")

print("En total tenemos", train_users.shape[0] + test_users.shape[0], "usuarios.")

print(sessions.shape[0], "Registros de sesión para" , sessions.user_id.nunique() , "usuarios." )

print((train_users.shape[0] + test_users.shape[0] -sessions.user_id.nunique()) , "Usuarios sin registros de sessión." )

print((countries.shape[0]) , "Registros en el Dataset de Países." )

print((age_gender.shape[0]) , "registros en el Dataset edad/genero." )
# Unimos usuarios de Pruebas y Entrenamiento

users = pd.concat((train_users, test_users), axis=0, ignore_index=True, sort=False)



# Removemos ID's

users.set_index('id',inplace=True)



users.head()
sessions.head()
countries