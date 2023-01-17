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

df
df.head(1)
df.iloc[3].values
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
coeficiente_as={}

curtosis={}

for i,j in zip(lista,df.columns):

    if i == "cuantitativo":

        coeficiente_as[i]=df.skew(axis = 0)

        curtosis[i]=df.kurtosis(axis = 0)

coeficiente_as



curtosis
desS = np.std(df["hf_score"])

media = np.mean(df["hf_score"])

a = media-desS

print(a)

b=media+desS

print(b)
print(min(df["hf_score"]))

print(max(df["hf_score"]))

print(len(df["hf_score"]))
df[df["hf_score"].between(a,b)]
1024/1620*100
c = media-(desS*2)

print(c)

d = media+(desS*2)

print(d)
df[df["hf_score"].between(c,d)]
1581/1620*100
import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

sns.distplot(df["hf_score"],hist=False, color="red")

y = np.linspace(0,1,20)

x = np.mean(df["hf_score"])

d1 = np.std(df["hf_score"])

d2 = np.std(df["hf_score"])

plt.plot([x]*20,y)

plt.plot([x-d1]*20,y)

plt.plot([x+d2]*20,y)

plt.plot([x-d1*2]*20,y)

plt.plot([x+d2*2]*20,y)
 