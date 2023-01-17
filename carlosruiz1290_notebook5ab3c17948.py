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
df = pd.read_csv('/kaggle/input/hfi_cc_2019.csv',na_values= ['-',' '])            

df.info()

df.dtypes
# Listas cualitativos y cuantitativos



lista=[]



for i in df.iloc[0].values:

    try:

        if type(float(i))==float:

            lista.append("cuantitativo")

    except:

        lista.append("cualitativo")

lista

# Rango



np.max(df.iloc[:,4:])- np.min(df.iloc[:,4:])
# 4 Rango



np.mean(df.iloc[:,4:])
# 5 Calcular la moda



import scipy

from scipy import stats 



scipy.stats.mode(df.iloc[:,0:4])
# 6 rango varianza y desviación estandar

#Rango



np.mean(df.iloc[:,4:])
#Varianza



np.var(df.iloc[:,4:])
# Desviacion estándar



np.sqrt(np.var(df.iloc[:,4:]))
#Coeficiente de asimetría y curtosis

for i in range(4,120):

        print('Columna', i)

        print("Coeficiente de asimetria:", df.skew()[i])

        print("Curtosis:", df.kurt()[i])

        
#Curtosis

for i in range(4,120):

    print('Curtosis: ', df.kurt()[i])