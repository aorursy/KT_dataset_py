# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
base = pd.read_csv('/kaggle/input/ebay-kleinanzeigen-car/autos.csv', encoding = 'ISO-8859-1')
base.head()
base.info()
# dados sem importancia
base = base.drop('dateCrawled', axis = 1)
base = base.drop('dateCreated', axis = 1)
base = base.drop('lastSeen', axis = 1)
base['name'].value_counts()
base = base.drop('name', axis = 1)
base['seller'].value_counts()
# dado desbalanceado
base = base.drop('seller', axis = 1)
base['offerType'].value_counts()
# dado desbalanceado
base = base.drop('offerType', axis = 1)
base['abtest'].value_counts()
base['vehicleType'].value_counts()
base['yearOfRegistration'].value_counts()
# dado desbalanceado
base = base.drop('yearOfRegistration', axis = 1)
base['gearbox'].value_counts()
base['powerPS'].value_counts()
# dado desbalanceado
base = base.drop('powerPS', axis = 1)
base['model'].value_counts()
base['odometer'].value_counts()
base['monthOfRegistration'].value_counts()
# dado desbalanceado
base = base.drop('monthOfRegistration', axis = 1)
base['notRepairedDamage'].value_counts()        
base['fuelType'].value_counts()
base['brand'].value_counts()
base['nrOfPictures'].value_counts()
# dado desbalanceado
base = base.drop('nrOfPictures', axis = 1)
base['postalCode'].value_counts()  
# dado desbalanceado
base = base.drop('postalCode', axis = 1)
base.head()
def convert_currency(val):
    new_val = val.replace(',','').replace('$', '').replace('km', '')
    return float(new_val)
base['price'] = base['price'].apply(convert_currency);
base['odometer'] = base['odometer'].apply(convert_currency);
base.head()
base.info()
inconsistentes_01 = base.loc[base.price <= 10]
inconsistentes_01.head(10)
base = base[base.price > 10]
base.head()
inconsistentes_02 = base.loc[base.price > 35000]
inconsistentes_02.head()
base = base.loc[base.price < 35000]
base.head()
base.info()
base.loc[pd.isnull(base['vehicleType'])]
base['vehicleType'].value_counts() # limousine
base.loc[pd.isnull(base['gearbox'])]
base['gearbox'].value_counts() # manuell
base.loc[pd.isnull(base['model'])]
base['model'].value_counts() # golf 
base.loc[pd.isnull(base['fuelType'])]
base['fuelType'].value_counts() # benzin 
base.loc[pd.isnull(base['notRepairedDamage'])]
base['notRepairedDamage'].value_counts() # nein 
valores = {'vehicleType': 'limousine',
          'gearbox': 'manuell',
          'model': 'golf',
          'fuelType': 'benzin',
           'notRepairedDamage': 'nein'}
base = base.fillna(value = valores)
base.head()
base.info()
previsores = base.iloc[:, 1:9].values
preco_real = base.iloc[:, 0].values
previsores
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelenconder_previsores = LabelEncoder()
previsores[:, 0] = labelenconder_previsores.fit_transform(previsores[:, 0])
previsores[:, 1] = labelenconder_previsores.fit_transform(previsores[:, 1])
previsores[:, 2] = labelenconder_previsores.fit_transform(previsores[:, 2])
previsores[:, 3] = labelenconder_previsores.fit_transform(previsores[:, 3])
previsores[:, 5] = labelenconder_previsores.fit_transform(previsores[:, 5])
previsores[:, 6] = labelenconder_previsores.fit_transform(previsores[:, 6])
previsores[:, 7] = labelenconder_previsores.fit_transform(previsores[:, 7])
previsores
previsores[0:8]
column_transform = ColumnTransformer([("encoder", 
                         OneHotEncoder(), 
                        [0,1,2,3,5,6,7])],    
                       remainder = 'passthrough')
# transformação de dados categoricos em valores numéricos
previsores = column_transform.fit_transform(previsores).toarray()
previsores
previsores.shape
from keras.models import Sequential
from keras.layers import Dense
# 307 colunas + 1 / 2
quant_neuronios = (307 + 1) / 2
quant_neuronios
regressor = Sequential()
regressor.add(Dense(units = quant_neuronios, activation = 'relu', input_dim = 307)) # primeira camada oculta e camada de entrada
regressor.add(Dense(units = quant_neuronios, activation = 'relu')) # segunda camada
regressor.add(Dense(units = 1, activation = 'linear')) # camada de saída, função linear é default
regressor.compile(loss = 'mean_absolute_error', optimizer = 'adam', metrics = ['mean_absolute_error'])
regressor.fit(previsores, preco_real, batch_size = 300, epochs = 100)
previsoes = regressor.predict(previsores)
previsoes
preco_real
preco_real.mean()
previsoes.mean()
