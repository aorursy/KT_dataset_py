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
df = pd.read_csv('/kaggle/input/cuiaba-metereologia/dados_83361_D_1910-12-31_2020-10-08.csv',sep=';',decimal=',')

df.head()
df['Data'] = pd.to_datetime(df['Data'],yearfirst=True)
df.rename({'Temperatura maxima':'temp_max','Temperatura Media':'temp_med','Temperatura Minima':'temp_min','Umidade Relativa':'umid','Precipitacao':'chuva','Velocidade Vento':'vento','Insolacao':'insol'},axis='columns',inplace=True)
df.head(2)
df.info()
clean = df.dropna(axis='index')
clean.info()
clean.describe()
clean.to_csv('Meteo_Cuiaba_Completo_Ready.csv')
temp = df[['Data','temp_min','temp_max','temp_med','umid']]
temp= temp.dropna()
temp.info()
temp.to_csv('Meteo_Cuiaba_Temperaturas_Ready.csv')