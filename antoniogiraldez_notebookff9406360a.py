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
df=pd.read_csv('/kaggle/input/the-human-freedom-index/hfi_cc_2019.csv')
# def is_number(n):

#        try:

#            float(n)   # Type-casting the string to `float`.

#                       # If string is not a valid `float`, 

#                       # it'll raise `ValueError` exception

#        except ValueError:

#            return False

#        return True

# df=df.replace('-',np.nan)

# df=df.replace(' ',np.nan)

# for i in df.columns:

#      if is_number(df[i][0]):

#          df[i]=df[i].astype('float')

# r=df.dtypes

# cuantitativos=[]

# cualitativos=[]

# for i in df.columns:

#     if r[i]=='float':

#         cuantitativos.append(i)

#     else:

#         cualitativos.append(i)

# cualitativos

# df.iloc[0].values
df.columns
df = df.replace("-",np.nan)

df = df.replace(" ",np.nan)

import scipy

from scipy import stats 

for i in df.columns:

    df[i]  = df[i].replace(np.nan,stats.mode(df[i])[0][0])

df.iloc[0][20]

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

df[df.columns[5]]

df.info()
rangos={}

for i,j in zip(lista,df.columns):

    if i=='cuantitativo':

        rangos[j]=np.max(df[j])-np.min(df[j])

import scipy

from scipy import stats

mod_prom={}

for i,j in zip(lista,df.columns):

    if i=='cualitativo':

        mod_prom[j]=stats.mode(df[j])

    if i=='cuantitativo':

        mod_prom[j]=np.mean(df[j])

import scipy

from scipy import stats

import matplotlib.pyplot as plt

import numpy as np

import statistics as stats

from scipy.stats import kurtosis

from scipy.stats import skew

rango={}

var={}

des_est={}

ske={}

curtosis={}

for i,j in zip(lista,df.columns):

    if i=='cuantitativo':

        des_est[j]=np.std(df[j])

        var[j]=np.var(df[j])

        rango[j]=np.max(df[j])-np.min(df[j])

        ske[j]=skew(df[j])

        curtosis[j]=kurtosis(df[j])

#Dentro de un desviacion estandar

primer_desv=[]

for i,j in zip(lista,df.columns):

    if i=='cuantitativo':

        suma=0

        c=np.std(df[j])

        p=np.mean(df[j])

        for r in range(df.shape[0]):

            if ((df[j][r]<(p+c)) &(df[j][r]>(p-c))):

                suma+=1

        primer_desv.append(suma)

segunda_desv=[]

for i,j in zip(lista,df.columns):

    if i=='cuantitativo':

        suma=0

        c=np.std(df[j])

        p=np.mean(df[j])

        for r in range(df.shape[0]):

            if ((df[j][r]<(p+2*c)) &(df[j][r]>(p-2*c))):

                suma+=1

        primer_desv.append(suma)

            
import seaborn as sns

plt.figure()

for i,j in zip(lista,df.columns):

    if i=='cuantitativo':

        if ske.get(j)<0:

            plt.subplot(221)

            sns.distplot(df[j],hist=False, color="red")

            plt.title('Skewness Negativa')

            break

for i,j in zip(lista,df.columns):

    if i=='cuantitativo':

        if ske.get(j)>0:

            plt.subplot(222)

            sns.distplot(df[j],hist=False, color="red")

            plt.title('Skewness positiva')

            break

for i,j in zip(lista,df.columns):

    if i=='cuantitativo':

        if curtosis.get(j)<0:

            plt.subplot(223)

            sns.distplot(df[j],hist=False, color="red")

            plt.title('Curtosis Negativa')

            break

for i,j in zip(lista,df.columns):

    if i=='cuantitativo':

        if curtosis.get(j)>0:

            plt.subplot(224)

            sns.distplot(df[j],hist=False, color="red")

            plt.title('Curtosis Positiva')

            break

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.9,

                    wspace=0.5)