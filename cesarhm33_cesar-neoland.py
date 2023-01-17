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
df.head(1)
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
df.skew()
for i,j in zip(lista,df.columns):

    if i == "cuantitativo":

        print("Asimetria", df.skew())

    else:

        print("No es cuantitativo")

        
for i,j in zip(lista,df.columns):

    if i == "cuantitativo":

        print("Curtosis", df.kurt())

    else:

        print("No es cuantitativo")

        
for i,j in zip(lista,df.columns):

    if i == "cuantitativo":

        print("Desviacion", df.std())

    else:

        print("No es cuantitativo")