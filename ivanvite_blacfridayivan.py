# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Proceso KDD

# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 18:26:40 2018

@author: IvanVite
"""
import random
import numpy as np  
import pandas as pd 
import seaborn as sns
import os
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt5') 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
#El Dataset es una muestra de las transacciones realizadas en una tienda minorista.
#La tienda quiere conocer mejor el comportamiento de compra del cliente.
#Es un problema de regresión en el que estamos tratando de predecir la variable dependiente
# (el total de la compra) con la ayuda de la información contenida en las otras variables.
#Hay 8 variables  para analizar:
# Genero, Edad, Ocupación, Cat. Ciudad, Años en la ciudad, Estado civil,
# Categorias de productos (1,2,3) y total comprado
#
#Lectura del dataset 
blackfriday = pd.read_csv('../input/black-friday/BlackFriday.csv')
#Exploracion del dataframe
blackfriday.shape
#537577, 12
blackfriday.head()#No se muestran todas las columnas pero veo que hay valores nulos
#Ver tipos de datos


tipos = blackfriday.dtypes
print(tipos)

#Exploracion de nulos
blackfriday.isnull().sum()
print(blackfriday[['Product_Category_1','Product_Category_2','Product_Category_3']])
#Vemos que solo en product_category 2 y 3 vienen nulos
#una posible explicación es que no todos los productos tienen más de una categoria
#Sin embargo no tengo descripcion de las mismas, se reemplazara con 0's mientras tanto
blackfriday.fillna(value=0,inplace=True)

#Descrbir el dataset
blackfriday['Gender'].unique()
blackfriday['Age'].unique()
blackfriday['Occupation'].unique()
blackfriday['City_Category'].unique()
blackfriday['Stay_In_Current_City_Years'].unique()
blackfriday['Marital_Status'].unique()
blackfriday['Purchase'].describe()
#ver los valores unicos de categorias
#blackfriday['Product_Category_1'].unique()
#blackfriday['Product_Category_2'].unique()
#blackfriday['Product_Category_3'].unique()


#Transformación
 
#3 El caracter \"+\" ha de ser eliminado en columna ,Stay_In_Current_City_Years    

blackfriday['Stay_In_Current_City_Years']=(blackfriday['Stay_In_Current_City_Years'].str.strip('+').astype('float'))

#Creando nuevas columnas para obtener nuevo datos
#Creando la columna de CompraTotal por el cliente
bf=blackfriday[['Purchase', 'User_ID']].groupby('User_ID').sum()
bf.head()
bf = bf.rename(columns={'Purchase': 'totalPurchase'})
bf.head()
#Agregarlo al "datasetOriginal"
bfMerge = blackfriday.merge(bf, on='User_ID')
bfMerge.head(n=30)

#Creando la columna de CompraTotal por el cliente
bf2=blackfriday[['Purchase', 'User_ID']].groupby('User_ID').count()
bf2.head()
bf2 = bf2.rename(columns={'Purchase': 'countPurchase'})
bf2.head()
bfMerge2 = bfMerge.merge(bf2, on='User_ID')
bfMerge2.head()

#Creando la columan media de consumo por usuario
bfMerge2['meanConsumption'] =  bfMerge2['totalPurchase'].div(bfMerge2['countPurchase']).to_frame('meanConsumption')
bfMerge2[['totalPurchase','countPurchase','meanConsumption']].head()
