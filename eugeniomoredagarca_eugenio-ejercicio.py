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
df = pd.read_csv("/kaggle/input/the-human-freedom-index/hfi_cc_2019.csv")
df.dtypes
df.shape

df.shape[1]

df.values[0,0]

#print(isinstance(df.values[0,1], int))

if (isinstance(df.values[0,5], float))==True:

    print("si")

else:

    print("no")
type(df.values[0,5])

df.values[0,4]
cuantitativos = []

cualitativos = []

for i in range(df.shape[1]):

    if (isinstance(df.values[0,i], int))==True:

        cuantitativos.append(i)

    else:

        cualitativos.append(i)

print(cualitativos)
df.iloc[3].values
df = df.replace("-",np.nan)

df = df.replace(" ",np.nan)
import scipy

from scipy import stats 
df.columns
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
df['ef_regulation_labor_dismissal'].unique()
for i in df.dtypes:

    print(i)
for i in df.columns:

    if df[i].dtype == "float":

        print("Rango de la columna:", i, "es:", max(df[i]) - min(df[i]))

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
from scipy.stats import skew

import scipy.stats as stats

from scipy.stats import kurtosis

for i in range(4,120):

    coeficiente=skew(df[df.columns[i]].values)

    print(df[df.columns[i]].name)

    print("Coeficiente de asimetría =",coeficiente)
for i in range(4,120):

    curtosis=kurtosis(df[df.columns[i]].values)

    print(df[df.columns[i]].name)

    print("Curtosis =",curtosis)

for i in range(4,120):

    media = np.mean(df[df.columns[i]].values)

    print(df[df.columns[i]].name)

    print("media =",media)

    print("*"*30)

for i in range(4,120):

    desv = np.std(df[df.columns[i]].values)

    print(df[df.columns[i]].name)

    print("desviacion estandar=",desv)
import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

contador=[]



for i in range(4,120):

    media = np.mean(df[df.columns[i]].values)

    print(df[df.columns[i]].name)

    print("media =",media)

    print("*"*30)

    desv = np.std(df[df.columns[i]].values)

    print("desviacion estandar=",desv)

    print("*"*30)

    rango_inf = media-desv

    rango_sup = media+desv

    k=0

    for j in df[df.columns[i]]:

        if (j>rango_inf) and (j<rango_sup):

            k+=1

    porcentaje = (k/len(df[df.columns[i]])*100)

    print("el porcenataje es:", porcentaje, "%")

    print("*"*30)
for i in df.columns:

    if df[i].dtype == "float":

        mean = np.mean(df[i])

        std = np.std(df[i])

        

        skewness = stats.skew(df[i], axis = 0, bias = True)

        kurtosis = stats.kurtosis(df[i], axis = 0, bias = True)



        sns.distplot(df[i], hist=False, color="blue", kde_kws={'bw': 0.1})

        if skewness > 0:

            if kurtosis > 0:

                    plt.title("Coeficiente de asimetría postivo, Curtosis positva.")

            else:

                    plt.title("Coeficiente de asimetría postivo, Curtosis negativa.")

        else:

            if kurtosis > 0:

                plt.title("Coeficiente de asimetría postivo, Curtosis positva.")

            else:

                plt.title("Coeficiente de asimetría postivo, Curtosis negativa.")

        plt.axvline(mean, ymin=0, ymax=1, color = "red")

        plt.axvline((mean- (2*std)), ymin=0, ymax=1, color = "green")

        plt.axvline((mean- std), ymin=0, ymax=1, color = "orange")

        plt.axvline((mean+ (2*std)), ymin=0, ymax=1, color = "green")

        plt.axvline((mean+ std), ymin=0, ymax=1, color = "orange")



    plt.show()