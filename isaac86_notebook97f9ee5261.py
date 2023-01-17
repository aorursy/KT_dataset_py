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
df.head()
df.info()
df.dtypes
df.columns
df = df.replace("-",np.nan)
df = df.replace(" ",np.nan)
df
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
        df[j]= df[j].astype("category")
df['ef_regulation_labor_dismissal'].unique()
for i in df.dtypes:
    print(i)
df[df.columns[5]]
for col in df.columns:
    if df[col].dtype == "float":
        print("Rango de la columna:", col, "es:", max(df[col]) - min(df[col]))
from scipy import stats

for col in df.columns:
    if df[col].dtype == "float":
        print("La moda de la columna", col, "es:", stats.mode(df[col])[0])
    else:
        print("El promedio de la columna", col, "es:", np.mean(df[col]))
for col in df.columns:
    if df[col].dtype == "float":
        print(col)
        print("Rango:", max(df[col]) - min(df[col]))
        print("Varianza:", np.var(df[col]))
        print("Desviacion Estandar:", np.std(df[col]))
        print()
df
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
sns.distplot(df["hf_score"],hist=False, color="red")
y = np.linspace(0,0.4,20)
x= np.mean(df["hf_score"])
d1= np.std(df["hf_score"])
plt.plot([x]*20,y)
plt.plot([x-d1]*20,y)
plt.plot([x+d1]*20,y)
plt.plot([x-d1*2]*20,y)
plt.plot([x+d1*2]*20,y)
df["hf_score"].skew()
df["hf_score"].kurt()
for i in range (4,120):
    media = np.mean(df[df.columns[i]])
    print (media)
    
for i in range (4,120):
    desviacion = np.std(df[df.columns[i]])
    print (desviacion)
for i in range (4,120):
    kurt = df[df.columns[i]].kurt()
    print (kurt)
for i in range (4,120):
    Skewness = df[df.columns[i]].skew()
    print (Skewness)
