import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

ds = pd.read_csv('../input/train_MV.csv')

ds.head()
# Verificar se existem linhas / colunas sem valores

ds.isnull().sum()
# Eliminar as linhas com colunas sem valores (max de 20)

ds.dropna(axis=0, inplace=True)

ds.isnull().values.any()
# Ver distribuição de valores

import seaborn as sns

import matplotlib.pyplot as plt

sns.distplot(ds['fare_amount']);

plt.title('Custo');
print("Menos de $2.5:", len(ds[ds['fare_amount'] < 2.5]))

print("Menos de $2.6:", len(ds[ds['fare_amount'] < 2.6]))

print("Mais de $80:", len(ds[ds['fare_amount'] > 80]))

print("Mais de $90:", len(ds[ds['fare_amount'] > 90]))

print("Mais de $100:", len(ds[ds['fare_amount'] > 100]))
# Remover abixo de $2 e acima de $90

ds = ds[ds['fare_amount'].between(left=2, right=90)]

ds.shape
print("Fora de NY (Origem.Latitude):", len(ds[~ds['pickup_latitude'].between(left=40, right=42)]))

print("Fora de NY (Destino.Latitude):", len(ds[~ds['dropoff_latitude'].between(left=40, right=42)]))



print("Fora de NY (Origem.Longitude):", len(ds[~ds['pickup_longitude'].between(left=-75, right=-72)]))

print("Fora de NY (Destino.Longitude):", len(ds[~ds['dropoff_longitude'].between(left=-75, right=-72)]))



print("Fora de NY:", len(ds[(~ds['dropoff_longitude'].between(left=-75, right=-72)) |

                             ~ds['pickup_latitude'].between(left=40, right=42)      |

                             ~ds['dropoff_latitude'].between(left=40, right=42)     |

                             ~ds['pickup_longitude'].between(left=-75, right=-72)

                           ]))
# Ficar apenas com os que ficam dentro do perómetro desejado

#ds = ds.loc[ds['pickup_latitude'].between(left=40, right=42)]

#ds = ds.loc[ds['dropoff_latitude'].between(left=40, right=42)]



#ds = ds.loc[ds['pickup_longitude'].between(left=-75, right=-72)]

#ds = ds.loc[ds['dropoff_longitude'].between(left=-75, right=-72)]



ds = ds.loc[(ds['dropoff_longitude'].between(left=-75, right=-72)) &

             ds['pickup_latitude'].between(left=40, right=42)      &

             ds['dropoff_latitude'].between(left=40, right=42)     &

             ds['pickup_longitude'].between(left=-75, right=-72)]



ds.shape
ds['dist_lat'] = (ds['dropoff_latitude' ] - ds['pickup_latitude' ]).abs()

ds['dist_lng'] = (ds['dropoff_longitude'] - ds['pickup_longitude']).abs()

print("Origem e destino iguais:", len(ds[(ds['dist_lat'] == 0) & (ds['dist_lng'] == 0)]))
ds = ds.loc[(ds['dist_lat'] > 0) | (ds['dist_lng'] > 0)]

ds.shape
# Já não são necessárias

ds.drop(['dist_lat','dist_lng'], axis=1)
# REMOVER a linha seguinte. é só para teste de código

ds = ds.sample(200000)

ds.shape
# Converter coluna da Data em Datetime

from datetime import datetime

ds['pickup_datetime'] = pd.to_datetime(ds['pickup_datetime'])

ds.dtypes
# vai usar outro DF para poder voltar atrás sem correr o código todo

# ds -> dataset original 'limpo'

# nds -> dataset de trabalho

nds = ds

nds.describe()
# acrescentar Dia, Mes e Hora ao Dataset

nds['mes' ] = nds['pickup_datetime'].dt.month

nds['dia' ] = nds['pickup_datetime'].dt.weekday

nds['hora'] = nds['pickup_datetime'].dt.hour

nds.head()
from math import radians, sin, cos, acos

#def distancia(row, origem, destino):

#    OLat = radians(row[origem[0]])

#    OLng = radians(row[origem[1]])

#    DLat = radians(row[destino[0]])

#    DLng = radians(row[destino[1]])

#    Distancia = 6.37101 * acos(sin(OLat)*sin(DLat) + cos(OLat)*cos(DLat)*cos(OLng - DLng))

#    return (Distancia)

#   return ( abs(np.linalg.norm(np.array(row[destino]) - np.array(row[origem]))))
#def distancia(row, origem, destino, factor=1):

#    return ((abs(row[destino[1]] - row[origem[1]]) ** factor) + (abs(row[destino[0]] - row[origem[0]])) ** factor) ** (1 / factor)
from math import radians, sin, cos, acos

#def distancia(lon1, lon2, lat1, lat2):

#    R = 6372800  # Earth radius in meters

    

#    phi1,   = math.radians(lat1)

#    phi2    = math.radians(lat2) 

#    dphi    = math.radians(lat2 - lat1)

#    dlambda = math.radians(lon2 - lon1)

    

#    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2

    

#    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))
def distancia(lat1, lon1, lat2, lon2):

    p = 0.017453292519943295 # Pi/180

    

    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2

    return 12742 * np.arcsin(np.sqrt(a))

#def distancia(x1, x2, y1, y2, p):

#    return ((abs(x2 - x1) ** p) + (abs(y2 - y1)) ** p) ** (1 / p)
nds[['pickup_latitude','pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']].dtypes
# Calcular a distância entre origem e destino

nds['distancia'] = distancia(nds['pickup_latitude'], nds['pickup_longitude'], nds['dropoff_latitude'], nds['dropoff_longitude'])

# nds['distancia'] = distancia(nds['pickup_longitude'], nds['dropoff_longitude'], nds['pickup_latitude'], nds['dropoff_latitude'])

# nds['distancia'] = nds.apply (lambda row: distancia(row, ['pickup_latitude', 'pickup_longitude'], ['dropoff_latitude', 'dropoff_longitude']), axis=1)

nds.head()
sns.distplot(ds['distancia']);

plt.title('Distancia');
print("Menos de 0,5 Km:", len(nds[nds['distancia'] < 0.5]))

print("Menos de 0,8 Km:", len(nds[nds['distancia'] < 0.8]))

print("Menos de 1 Km:", len(nds[nds['distancia'] < 1]))

print("Mais de 22 Km:", len(nds[nds['distancia'] > 22]))
#Eliminar as que têm deslocações pouco frequentes

nds = nds.loc[(nds['distancia'] > 0.5) & (nds['distancia'] < 22)]

nds.shape
# calcular valor médio por metro

nds['vmedio'] = nds['fare_amount'] / nds['distancia']

nds.head()
sns.distplot(nds['vmedio']);

plt.title('Valor médio');
nds.loc[nds['vmedio'] > 60].head(10)
print("Menos de $5 / Km:", len(nds[nds['vmedio'] < 5]))

print("Entre de $5 / Km e $20:", len(nds[(nds['vmedio'] > 5) & (nds['vmedio'] < 20)]))

print("Entre de $20 / Km e $80:", len(nds[(nds['vmedio'] > 20) & (nds['vmedio'] < 80)]))

print("Mais de $80 / Km:", len(nds[nds['vmedio'] > 80]))
nds = nds.loc[nds['vmedio'] < 80]

nds.shape
from sklearn.model_selection import train_test_split

train, test = train_test_split(nds, test_size=0.2)

train.shape
dsX = train[['passenger_count','mes','dia','hora','distancia']]

dsY = train['fare_amount']

dsX.head()
from sklearn import preprocessing

scaler = preprocessing.StandardScaler().fit(dsX.values)

dsXS = scaler.transform(dsX.values)  
from sklearn import linear_model

modelo = linear_model.LinearRegression(fit_intercept = False)

modelo.fit(dsXS, dsY)



print(modelo.intercept_)

print(modelo.coef_)
dtX = test[['passenger_count','mes','dia','hora','distancia']]

dtY = test['fare_amount']

dtX.head()
predY = modelo.predict(dtX)
predY
dtX['fare_amount'] = predY

dtX.head()
import sklearn.metrics as sklm

import math

def print_status(real, predicted, nfeatures):

    r2 = sklm.r2_score(real, predicted)

    r2_adj = r2 - (nfeatures - 1)/(real.shape[0] - nfeatures) * (1 - r2)

    

    ## Print the usual metrics and the R^2 values

    print('Mean Square Error      = ' + str(sklm.mean_squared_error(real, predicted)))

    print('Root Mean Square Error = ' + str(math.sqrt(sklm.mean_squared_error(real, predicted))))

    print('Mean Absolute Error    = ' + str(sklm.mean_absolute_error(real, predicted)))

    print('Median Absolute Error  = ' + str(sklm.median_absolute_error(real, predicted)))

    print('R^2                    = ' + str(r2))

    print('Adjusted R^2           = ' + str(r2_adj))

   

print_status(dtY, predY, dtX.count(axis=1))
# Ler ficheiro

dt = pd.read_csv('../input/test_MV.csv')

dt.head()
# Preparar Dataset de trabalho

nts = dt.drop(['key'], axis=1)

nts.head()
# Resto das colunas

nts['pickup_datetime'] = pd.to_datetime(dt['pickup_datetime'])

nts['mes' ] = nts['pickup_datetime'].dt.month

nts['dia' ] = nts['pickup_datetime'].dt.weekday

nts['hora'] = nts['pickup_datetime'].dt.hour

#nts['distancia'] = distancia(nts['pickup_longitude'], nts['dropoff_longitude'], nts['pickup_latitude'], nts['dropoff_latitude'], 1)

nts['distancia'] = distancia(nts['pickup_latitude'], nts['pickup_longitude'], nts['dropoff_latitude'], nts['dropoff_longitude'])

nts.head()

# Eliminar as colunas que não são necessárias

nts.drop(['pickup_datetime','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude'], axis=1, inplace=True)

nts.head()
# Aplicar o modelo

test_Y = modelo.predict(nts)

test_Y
# Submeter resultado

result = pd.DataFrame({'key': dt.key, 'fare_amount': test_Y})

result.to_csv('sub_teste_02.csv', index = False)