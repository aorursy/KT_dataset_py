import time

import numpy   as np

import pandas  as pd

import seaborn as sb

import matplotlib.pyplot as plt

import altair as alt

import missingno as msno

import unidecode

from geopy.geocoders import Nominatim
train = pd.read_csv("../input/murcia-car-challenge/train.csv",            index_col="Id")

test  = pd.read_csv("../input/murcia-car-challenge/test.csv",             index_col="Id")

sub   = pd.read_csv("../input/murcia-car-challenge/sampleSubmission.csv", index_col="Id")
def cleanString(valor):

    return unidecode.unidecode(valor.upper().strip())



def clean(data):

    

    data['Marca']  = data['Marca'].apply(cleanString)

    data['Modelo'] = data['Modelo'].apply(cleanString)



    data['Marca'].replace("MERCEDES", "MERCEDES-BENZ",     inplace=True) # Fix Mercedes

    data['Modelo'].replace("CLASE ", "CLASE_", regex=True, inplace=True) # Fix Mercedes models

    data['Modelo'].replace("SERIE ", "SERIE_", regex=True, inplace=True) # Fix BMW models

    data['Modelo'].replace("RANGE ROVER ", "", regex=True, inplace=True) # Fix Land Rover models

    

    return data



train = clean(train)

test  = clean(test)
geolocator = Nominatim(user_agent="myGeocoder")



provincias = train['Provincia'].unique().tolist()

provincias[:5]
def getLatLon():

    provincias_loc = pd.DataFrame(columns=('Provincia', 'Latitude', 'Longitude'))



    for p in provincias:

        loc = geolocator.geocode(p.replace('_', ' ') + ", España")

        print(p,"-->", loc, "-->", loc.latitude, loc.longitude)

        provincias_loc = provincias_loc.append({'Provincia':p , 'Latitude':loc.latitude, 'Longitude':loc.longitude}, ignore_index=True)

        time.sleep(.5)



    provincias_loc.to_csv("provincias.csv", index=False)



getLatLon()

provincias_loc = pd.read_csv("provincias.csv")

provincias_loc.head()
alt.Chart(provincias_loc).mark_circle().encode(

    longitude='Longitude:Q',

    latitude='Latitude:Q',

    #color='leading digit:N',

    tooltip='Provincia:N'

)
train = pd.merge(train, provincias_loc, on='Provincia', how='left')

test  = pd.merge(test,  provincias_loc, on='Provincia', how='left')
train.head(3)
train["Modelo_1st"] = train.Modelo.str.split().str.get(0)

test["Modelo_1st"]  = test.Modelo.str.split().str.get(0)
def getMotorLitros(modelo):

    if ". " in modelo:

        idx   = modelo.index(". ")

        left  = modelo[idx-1:idx]

        right = modelo[idx+2:idx+3]

        if left.isnumeric() and right.isnumeric():

            return left+"."+right

        else:

            return None        

    else:

        return None



train['Motor_litros'] = train['Modelo'].apply(getMotorLitros)

test['Motor_litros']  = test['Modelo'].apply(getMotorLitros)
train['Motor_litros'].value_counts()[:10]
train[["Marca", "Modelo", "Modelo_1st", "Motor_litros"]].sample(10)
def tranform2type(Tiempo):

    if   Tiempo.endswith('días'):  return 'dias'

    elif Tiempo.endswith('horas'): return 'horas'

    elif Tiempo.endswith('hora'):  return 'hora'

    elif Tiempo.endswith('min'):   return 'min'

    elif Tiempo.endswith('seg'):   return 'seg'

    elif Tiempo.endswith('nuevo anuncio'):  return 'nuevo anuncio'

    elif Tiempo.endswith('destacado'):      return 'destacado'



    

def tranform2dias(Tiempo):

    if   Tiempo.endswith('días'):  return int(Tiempo.replace(' días', ''))

    elif Tiempo.endswith('horas'): return int(Tiempo.replace(' horas', ''))/24

    elif Tiempo.endswith('hora'):  return int(Tiempo.replace(' hora', ''))/24

    elif Tiempo.endswith('min'):   return int(Tiempo.replace(' min', ''))/(60*24)

    elif Tiempo.endswith('seg'):   return int(Tiempo.replace(' seg', ''))/(60*60*24)

    elif Tiempo.endswith('nuevo anuncio'):  return 0

    elif Tiempo.endswith('destacado'):      return -1



train["Tiempo_días"] = train["Tiempo"].apply(tranform2dias) # Variable numerica

train["Tiempo_tipo"] = train["Tiempo"].apply(tranform2type) # Variable categorica



test["Tiempo_días"] = test["Tiempo"].apply(tranform2dias) # Variable numerica

test["Tiempo_tipo"] = test["Tiempo"].apply(tranform2type) # Variable categorica
train[["Tiempo", "Tiempo_días", "Tiempo_tipo"]].sample(5)
current_year = 2020.1

train["Años hasta Feb 2020"] = abs(train["Año"] - current_year)

train["Kms medios por año"]  = train["Kms"] / train["Años hasta Feb 2020"]



test["Años hasta Feb 2020"] = abs(test["Año"] - current_year)

test["Kms medios por año"]  = test["Kms"] / train["Años hasta Feb 2020"]
train[["Kms", "Año", "Años hasta Feb 2020", "Kms medios por año"]].sample(10)
train.rename(columns={'Año':         'Ano'},         inplace=True)

train.rename(columns={'Año_missing': 'Ano_missing'}, inplace=True)

train.rename(columns={'Tiempo_días': 'Tiempo_dias'}, inplace=True)



test.rename(columns={'Año':         'Ano'},         inplace=True)

test.rename(columns={'Año_missing': 'Ano_missing'}, inplace=True)

test.rename(columns={'Tiempo_días': 'Tiempo_dias'}, inplace=True)
train.to_csv("train_featEng.csv", index_label="Id")

test.to_csv("test_featEng.csv", index_label="Id")