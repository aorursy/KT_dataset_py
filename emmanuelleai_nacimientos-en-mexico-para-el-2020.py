# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv('/kaggle/input/nacimientos-en-mxico/Natalidad_Mexico.csv')

data
data.corr()

plt.matshow(data.corr())
plt.figure(figsize=(20,10))
sns.boxplot(x='estado', y='Hombres', data=data)
data.estado.unique()
data[data.estado=='Ciudad de México']
#cdmx abreviacion para ciudad de mexico
cdmx=data[data.estado=='Ciudad de México']
cdmx.corr()
# Graficamos el total, y los sexos
plt.scatter(cdmx.year, cdmx.Total)
#plt.scatter(cdmx.year, cdmx.Hombres)
#plt.scatter(cdmx.year, cdmx.Mujeres)
#plt.scatter(cdmx.year, cdmx['No especificado'])
plt.grid()
plt.xlabel('Año')
plt.ylabel('Numero de nacimientos')
plt.scatter(cdmx.year, cdmx.Hombres)
plt.scatter(cdmx.year, cdmx.Mujeres)
plt.grid()
plt.xlabel('Año')
plt.ylabel('Numero de nacimientos')
## Metodo Calculo de la pendiente mediante las desviaciones estandar

cdmxy = cdmx.year.to_numpy().reshape(-1,1)
cdmxt = cdmx.Total.to_numpy().reshape(-1,1)

sumx = np.sum(cdmx.year) # Calculamos la suma de todos los datos de nuestra variable x (el año)
sumy = np.sum(cdmx.Total) # Calculamos la suma de todos los datos de nuestra variable y (el Total de nacimientos)

sumxx = np.sum(cdmx.year*cdmx.year) # Calculamos la suma de los cuadrados de todos los datos de nuestra variable x (el año)
sumyy = np.sum(cdmx.Total*cdmx.Total) # Calculamos la suma de los cuadrados de todos los datos de nuestra variable y (el Total de nacimientos)
sumxy = np.sum(cdmx.year*cdmx.Total) #Calculamos la suma de ada producto de nuestras variables x*y

n=len(cdmx.year) # n es el numero de muestras (en este caso el numero de años )

mediax = np.mean(cdmx.year) # Se calcula lasmedias de cada variable
mediay= np.mean(cdmx.Total)

# Se calcula la pendiente de acuerdo a la formula es la siguiente
m2 = (sumx*sumy-n*sumxy) / (sumx*sumx-n*sumxx)


# se calcula la constante b de la ecuacion
b2 = mediay - m2*mediax # Ahora ya tenemos todos los valores para calcular la ecuacion de la recta
print('La ecuacion de la recta es:')
print('y=',m2,'x+',b2)

def f1(x):
    return m2*x+b2  # La ecuacion de la recta

x = range(2010,2020)  # El rango esta determinado por el año minimo y el maximo año que queremos obtener (2020)
plt.plot(x, [f1(i) for i in x], linestyle='dotted' , color='blue', linewidth=1.5 )

# Volvemos a graficar para el total y agregamos la ecuacion de la recta que representa nuestra regresion lineal
plt.scatter(cdmx.year, cdmx.Total)
plt.grid()
plt.xlabel('Año')
plt.ylabel('Numero de nacimientos')
plt.show()
print('Para el 2020 estima que nazcan tan solo en la Ciudad de Mexico ',f1(2020), ' nuevos mexicanos')
