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
import pandas as pd

df=pd.read_csv('/kaggle/input/the-human-freedom-index/hfi_cc_2019.csv')

df
df.iloc[3].values #para ver el tipo de values 
df = df.replace("-",np.nan)       # reemplazar guiones y espacios por np.nan 

df = df.replace(" ",np.nan)
import scipy

from scipy import stats



for i in df.columns:

    df[i]  = df[i].replace(np.nan,stats.mode(df[i])[0][0])  #sustitución de np.nans por la moda; los dos zeros son para seleccionar el único valor de la moda y ninguno más  

lista =[]                                   #reemplazar por cualitativo y cuantittivo

for i in df.iloc[0].values:

    try:

        if type(float(i))==float:

            lista.append("cuantitativo")

    except:

        lista.append("cualitativo")

lista
for i,j in zip(lista,df.columns):           #cambiar cuantitaticos a float

    if i == "cuantitativo":

        df[j] = df[j].astype("float")      

    else:

        df[j]= df[j].astype("category")


#Solución de Sergio: 

for i in range(4,120): # del 0 al 3 son: año y columnas cualitativas 

    rango = np.max(df[df.columns[i]].values) - np.min(df[df.columns[i]].values)

    print(df[df.columns[i]].name,"=",rango)
##Obtén la moda o promedio dependiendo el caso para cada atributo.
# Valores cualitativos: moda 

for i in range(0,4): 

    moda=scipy.stats.mode(df[df.columns[i]].values) #poenmos el values para extraer el valor 

    print(df[df.columns[i]].values)

    print(moda)

    print('-'*70)
#Valores cuantitativos: promedio 

for i in range(4,120):

    promedio= np.mean(df[df.columns[i]].values)

 

    print('Promedio de',df[df.columns[i]].name, 'es:' ,promedio)  # .name para obtener el nombre de la columna

    print('-'*70)
#VARIANZA

for i in range(4,120):

    varianza= np.var(df[df.columns[i]].values)

    print('Varianza de', df[df.columns[i]].name, 'es:', varianza )

    print('-'*80)
#DESVIACIÓN ESTÁNDAR

for i in range(4,120):

    stdr_dev= np.std(df[df.columns[i]].values)

    print('Desviación estándar de', df[df.columns[i]].name, 'es:', stdr_dev )

    print('-'*80)
#Coef Asimetría/ Skew 

for i in range(4,120):

    Skew= scipy.stats.skew(df[df.columns[i]].values)

    print('Skew', df[df.columns[i]].name, ': ', stdr_dev)

    curtosis= scipy.stats.kurtosis(df[df.columns[i]].values)

    print('Curtosis de', df[df.columns[i]].name, ': ', curtosis)

    print('-'*50)
# stdr dev

stdr_dev= np.std(df['hf_score'])

media= np.mean(df['hf_score'])

a=media-stdr_dev #desviacion estandar por la izda --> distancia entre media y stdr_dev hacia izquierda --> 

print(a)

b=media+stdr_dev #desviacion estandar por la dcha --> distancia entre media y stdr_dev hacia derecha

print(b)
print('min:',min(df['hf_score']))

print('max:',max(df['hf_score']))

print('length (nº total datos):',len(df['hf_score']))
df[df['hf_score'].between(a,b)] # número de datos entre a y b (desviación estandar)
#porcentaje de datos dentro de la primera desviación estandar:

datos_stdr_dev= len(df[df['hf_score'].between(a,b)])/len(df['hf_score'])*(100)

print('Porcentaje de datos dentro la Desviación Estándar:', datos_stdr_dev, '%')
# SEGUNDA Stdr Dev

stdr_dev= np.std(df['hf_score'])

media= np.mean(df['hf_score'])

a2=media-(stdr_dev*2) #desviacion estandar por la izda --> distancia entre media y stdr_dev hacia izquierda --> 

print(a2)

b2=media+(stdr_dev*2) #desviacion estandar por la dcha --> distancia entre media y stdr_dev hacia derecha

print(b2)
df[df['hf_score'].between(a2,b2)] # número de datos entre a2 y b2 (SEGUNDA desviación estandar)
#SEGUNDA STDR DEVIATION

datos_stdr_dev= len(df[df['hf_score'].between(a2,b2)])/len(df['hf_score'])*(100)

print('Porcentaje de datos dentro de 2ª Desviación Estándar:',datos_stdr_dev, '% (distribución normal)')
#Distribuciones de cada atributo 



for i in range(4,120):

    media= np.mean(df[df.columns[i]].values)

    stdr_dev= np.std(df[df.columns[i]].values)

    varianza= np.var(df[df.columns[i]].values)

    skew= scipy.stats.skew(df[df.columns[i]].values)

    curtosis= scipy.stats.kurtosis(df[df.columns[i]].values)

    

    print('Media',df[df.columns[i]].name, ':', media )

    print('Stdr.Deviation',df[df.columns[i]].name, ':', stdr_dev)

    print('Varianza',df[df.columns[i]].name, ':', varianza )

    print('Skew', df[df.columns[i]].name, ': ', skew)

    print('Curtosis', df[df.columns[i]].name, ': ', curtosis)

    print('-'*50)
#Gráficas de 'hf_score','hf_rank', 'hf_quartile'
