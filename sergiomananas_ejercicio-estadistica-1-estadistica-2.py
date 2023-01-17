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
df = pd.read_csv('/kaggle/input/the-human-freedom-index/hfi_cc_2019.csv')

df.head()
df.dtypes
df.iloc[0].values
df = df.replace("-",np.nan)

df = df.replace(" ",np.nan)
import scipy

from scipy import stats
for i in df.columns:

    df[i]  = df[i].replace(np.nan,stats.mode(df[i])[0][0])
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

        df[j] = df[j].astype("category")
df.info()
for i in range(4,120):

    rango = np.max(df[df.columns[i]].values) - np.min(df[df.columns[i]].values)

    print(df[df.columns[i]].name,"=",rango)
#VARIABLES CUALITATIVAS

for i in range(0,4):

    moda = scipy.stats.mode(df[df.columns[i]].values)

    print(df[df.columns[i]].name)

    print(moda)

    print("*"*30)
#VARIABLES CUANTITATIVAS

for i in range(4,120):

    media = np.mean(df[df.columns[i]].values)

    print(df[df.columns[i]].name)

    print("media =",media)

    print("*"*30)
for i in range(4,120):

    rango = np.max(df[df.columns[i]].values) - np.min(df[df.columns[i]].values)

    print(df[df.columns[i]].name)

    print("rango =",rango)

    varianza = np.var(df[df.columns[i]].values)

    print("varianza =",varianza)

    desv = np.std(df[df.columns[i]].values)

    print("desviacion estandar=",desv)

    print("*"*30)
from scipy.stats import kurtosis

from scipy.stats import skew
for i in range(4,120):

    print(df[df.columns[i]].name)

    print("Coeficiente de asimetría =",skew(df[df.columns[i]].values))

    print("Curtosis =",kurtosis(df[df.columns[i]].values))

    print("*"*30)
for i in range(4,120):

    print(df[df.columns[i]].name)

    media = np.mean(df[df.columns[i]].values)

    desv = np.std(df[df.columns[i]].values)

    rango11 = media-desv

    rango12 = media+desv

    rango21 = media-(2*desv)

    rango22 = media+(2*desv)

    porcentaje1 = round((len(df[df[df.columns[i]].between(rango11,rango12)]))/len(df[df.columns[i]].values)*100,1)

    porcentaje2 = round((len(df[df[df.columns[i]].between(rango21,rango22)]))/len(df[df.columns[i]].values)*100,1)

    print("El",porcentaje1,"% de las observaciones se encuentra dentro del intervalo (",round(rango11,1),",",round(rango12,1),")")

    print("El",porcentaje2,"% de las observaciones se encuentra dentro del intervalo (",round(rango21,1),",",round(rango22,1),")")

    print("*"*30)
import seaborn as sns

import matplotlib.pyplot as plt
for i in ["hf_score","hf_rank","pf_ss_homicide"]:

    #Gráfica las distribuciones

    sns.distplot(df[i],hist=False, color="red")

    #Calcular el porcentaje de datos dentro de 1 desviación standard y 2 desviaciones standard

    media = np.mean(df[i])

    desv = np.std(df[i])

    plt.axvline(x=media, ymin=-1, ymax=1,color= "blue")

    plt.axvline(x=media-desv, ymin=-1, ymax=1,linestyle="dashed",color= "green")

    plt.axvline(x=media+desv, ymin=-1, ymax=1,linestyle="dashed",color= "green")

    plt.axvline(x=media-(2*desv), ymin=-1, ymax=1,linestyle="dashed",color= "green")

    plt.axvline(x=media+(2*desv), ymin=-1, ymax=1,linestyle="dashed",color= "green")

    #Indica si el coeficiente de asimetría y curtosis son negativos o positivos

    #Coeficiente de asimetria

    if skew(df[i]) > 0:

        titulo_asimetria = "Coeficiente de asimetría positivo"

    elif skew(df[i]) == 0:

        titulo_asimetria = "Coeficiente de asimetría indica distribución normal"

    else:

        titulo_asimetria = "Coeficiente de asimetría negativo"

    #Curtosis    

    if kurtosis(df[i]) > 0:   

        titulo_curtosis = "Leptocúrtica: Decaimiento rápido y cola ligera"

    elif kurtosis(df[i]) > 0: 

        titulo_curtosis = "Mesocúrtica: Curva normal"

    else:

        titulo_curtosis = "Paticúrtica: Decaimiento lento"

        

    plt.title(str(titulo_asimetria) + " || " + str(titulo_curtosis))

    

    plt.show()