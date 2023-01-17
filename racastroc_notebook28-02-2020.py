# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/inf265-pucp-competencia-capstone/train.csv')

test = pd.read_csv('/kaggle/input/inf265-pucp-competencia-capstone/test.csv')

saleData = pd.read_excel('/kaggle/input/saledata/SaleData.xlsx')
train.head(10)
# Concatenar

dataToConcatenate = [train,test]

concatenado = pd.concat(dataToConcatenate, sort =True)

concatenado
print(train.shape)

print(test.shape)

if train.shape[1] - test.shape[1]:

    print("No tienen el mismo numero de columnas")
# Imprimimos la columna diferente

set(train.columns).difference(set(test.columns))
# Encontrandos los nulos

toShow = concatenado[concatenado['Target'].isna()]

print(toShow.shape[0])

toShow
concatenado.columns
concatenado = concatenado.rename(columns = lambda x: x.strip())
concatenado.columns
concatenado = concatenado.rename(columns = lambda x: x.replace(' ', '_'))
concatenado.columns
# VAMOS A LEER UN POCO CON PANDAS
#Funciones Herramientas

def pintarRaya():

    print('--------------------------------------------------')



def contarMissing(data, column):

    return data[column].isna().sum()

    

def mostrarValueCounts(data, column):

    valueCounts = data[column].value_counts()

    valueCountsNormalize = data[column].value_counts(normalize = True)

    pintarRaya()

    print(valueCounts)

    pintarRaya()

    print(valueCountsNormalize)

    pintarRaya()
mostrarValueCounts(data = concatenado, column = 'Target')
contarMissing(data = concatenado, column = 'Target')
sns.heatmap(concatenado.isna()) #DATA MUY LIMPIA
mostrarValueCounts(concatenado, 'Estado_Civil')

mostrarValueCounts(concatenado, 'Edad')
# CONDICIONALES LECTUR

concatenado[(concatenado['Estado_Civil'] == 'casado') & (concatenado['Edad'] == 36)]
# EJERCICIO BUSCAR SI UNA PERSONA TIENE EL ESTADO CIVIL SOLTERO Y UNA EDAD ENTRE LOS 37 Y 40 AÑOS
# MOSTRAR A LAS PERSONAS QUE TIENEN CELULAR, SEA SOLTERO Y A LA VEZ SEA MAYOR DE  O TENGA UNA DEUDA PERSONAL MENOR A 33950. 

# HALLAR CUANTAS PERSONAS SON
concatenado.iloc[:, 0:5]
# CREAR UNA FUNCION QUE ME MUESTRE SOLOS LOS CAMPOS ID EDAD CONTACTO DEUDA_PERSONAL Y TARGET EN UN DATAFRAME DURANTE LA LECTURA DEL TRAIN.CSV 
saleDataFrame = pd.pivot_table(saleData,index=["Manager"],values=["Sale_amt"],aggfunc=[np.mean,len])

saleDataFrame
# HACER GROUP BY PARA HALLAR LA MEDIANA, STD, Y SUMA. CON RESPECTO AL MANAGER Y LAS UNIDADES VENDIDAS

# INTENTAR HACER GROUP BY PARA MODA CON RESPECTO AL MANAGER Y LA REGION ASÍ COMO EL ITEM. SE PODRA O SE NECESITA HACER UNA FUNCION O UNA LIBRERÍA ... ? 
saleData
saleDataFrame = pd.pivot_table(saleData,index=["Manager"],values=["Item"],aggfunc= lambda x: tuple(stats.mode(x)[0]))

saleDataFrame