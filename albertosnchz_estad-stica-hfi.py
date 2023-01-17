# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scipy

from scipy import stats 



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/the-human-freedom-index/hfi_cc_2019.csv")

df
df.iloc[0].values
df=df.replace("-",np.nan)

df=df.replace(" ",np.nan)
for i in df.columns:

    df[i]  = df[i].replace(np.nan,stats.mode(df[i])[0][0])
df.iloc[0].values
lista =[]

for i in df.iloc[0].values:

    try:

        if type(float(i))==float:

            lista.append("cuantitativo")

    except:

        lista.append("cualitativo")

lista
for i,j in zip(lista,df.columns):

    if i == "cuantitativo":

        df[j] = df[j].astype("float")

    else:

        df[j]= df[j].astype("category")
df.info()
#Define el rango de cada uno de los atributos (cuantitativos)



for i in range(4,120):

    rango = np.max(df[df.columns[i]].values) - np.min(df[df.columns[i]].values)

    print("Rango de",df[df.columns[i]].name,"=",rango)
#Variables cualitativas

for i in range(0,4):

    moda = scipy.stats.mode(df[df.columns[i]].values)

    print(df[df.columns[i]].name)

    print(moda)
#Variables cuantitativas

for i in range(4,120):

    media = np.mean(df[df.columns[i]].values)

    print(df[df.columns[i]].name)

    print("Media =",media)
#Obtén el rango, varianza, desviación estandar para los atributos cuantitativos

print("~"*40)

for i in range(4,120):

    rango = np.max(df[df.columns[i]].values) - np.min(df[df.columns[i]].values)

    print(df[df.columns[i]].name)

    print("rango =",rango)

    varianza = np.var(df[df.columns[i]].values)

    print("varianza =",varianza)

    desv = np.std(df[df.columns[i]].values)

    print("desviación estándar=",desv)

    print("~"*40)
for i in range(4,120):

    curtosis = df[df.columns[i]].kurtosis(axis = 0, skipna = True)

    asimetria = df[df.columns[i]].skew()

    print(df[df.columns[i]].name)

    

    print("Curtosis:",df[df.columns[i]].kurtosis(axis = 0, skipna = True))

    if curtosis > 0:

        print("La gráfica es leptocúrtida")

    elif curtosis < 0:

        print("La gráfica es paticúrtida")

    else:

        print("La gráfica es mesocúrtida")

    

    print("Coeficiente de asimetría:",asimetria)

    if asimetria > 0:

        print("Asimetría positiva")

    elif asimetria < 0:

        print("Asimetría negativa")

    else:

        print("Sigue una distribución  normal")

    print("~"*40)
for i in range(4,120):

    print(df[df.columns[i]].name)

    media = np.mean(df[df.columns[i]].values) 

    desviacion = np.sqrt(np.var(df[df.columns[i]].values))

    rango1=media-desviacion

    rango2=media+desviacion

    rango_total=(len(df[df[df.columns[i]].between(rango1,rango2)]))/len(df[df.columns[i]].values)*100

    print("El porcentaje de datos dentro de 1 desviación estándar", df[df.columns[i]].name, "es del", round(rango_total,2), "%")

    print("~"*40)
for i in range(4,120):

    print(df[df.columns[i]].name)

    media = np.mean(df[df.columns[i]].values) 

    desviacion = np.sqrt(np.var(df[df.columns[i]].values))

    rango1=media-2*desviacion

    rango2=media+2*desviacion

    rango_total=(len(df[df[df.columns[i]].between(rango1,rango2)]))/len(df[df.columns[i]].values)*100

    print("El porcentaje de datos dentro de 2 desviaciones estándar", df[df.columns[i]].name, "es del", round(rango_total,2), "%")

    print("~"*40)