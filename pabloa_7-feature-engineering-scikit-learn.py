import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import preprocessing

from sklearn import impute
# importo el dataset de properati

properati = pd.read_csv('../input/datos_properati.csv', delimiter=',', parse_dates = ['created_on'])
properati.isnull().sum()
#  SimpleImputer sirve para reemplazar los valores NaN con el valor de la mediana de dicho atributo

# Opciones de Aplicación: 'mean', 'median', 'most_frequent', 'constant' 

imp = impute.SimpleImputer(missing_values= np.nan, strategy='median', fill_value=None) 
# Cuál es la mediana de esa columna?

properati['price_aprox_usd'].median()
# Aplicamos el valor de la mediana con la tecnica de Imputer en la columna "price_aprox_usd" y guardamos SOLO ESA COLUMNA

# en el elemento "properati_price_imp"

properati_price_imp = imp.fit_transform(properati[['price_aprox_usd']])
np.shape(properati[['price_aprox_usd']])
# luego revisamos si efectivamente sacamos todos los NaNs y es correcto

np.isnan(properati_price_imp).any()
# ahora reemplazamos nuestra columna con la nueva que tiene los valores reemplazados por la mediana

properati['price_aprox_usd'] = properati_price_imp
# nuevamente, observamos que esta columna ahora no tiene ningun valor nulo

properati['price_aprox_usd'].isnull().sum()
# por otro lado la mediana no ha sido afectada

properati['price_aprox_usd'].median()
# Si tengo claro como quiero imputar, puedo hacerlo de varias columnas a la vez

properati_2 = imp.fit_transform(properati.iloc[:,7:14])
# Vemos los valores que toma la variable

properati.property_type.unique()
# Label Encoder transforma mis variables categoricas en numéricas.

le_proptype = preprocessing.LabelEncoder()
# "aprendimos" un array numerico con 4 valores posibles, uno por cada categoria

le_proptype.fit_transform(properati['property_type'])
property_type_le = le_proptype.fit_transform(properati['property_type'])
le_proptype.classes_
# ahora la columna property_type del dataset properati tiene categorias numericas

properati['property_type_le'] = property_type_le
# uso one hot encoder para transformar mis categorias numericas en categorias binarias

ohenc = preprocessing.OneHotEncoder(sparse = False)
np.shape(property_type_le)[0]
# np.reshape sirve para cambiar las dimensiones de un array de numpy

property_type_le = np.reshape(property_type_le, (np.shape(property_type_le)[0],1))
np.shape(property_type_le)
property_type_le
# con one hot encoder transformo mi vector de categorias (detalladas como numeros)

# en vectores de 1 y 0s , que tienen tantas posicoines como categorias.

onehot_encoded = ohenc.fit_transform(property_type_le)
onehot_encoded
# transformo mi array de numpy "one hot encoded" a un dataframe y le asigno los nombres 

# de las columnas segun la categoria para acordarme que significa cada columna

onehot_pd = pd.DataFrame(onehot_encoded, index = properati.index , columns = le_proptype.classes_)
onehot_pd.head()