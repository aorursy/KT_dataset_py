# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import numpy as np
dfH = pd.read_csv('../input/home-medical-visits-healthcare/Hack2018_ES.csv', header=0)
dfAW = pd.read_csv('../input/merge-air-weather-terrassa/ficheroAW.csv', header=0)
dfH
dfAW
dfH['Fecha'] = pd.to_datetime(dfH.Fecha)
dfH['Month'] = dfH.Fecha.dt.month
dfH['Day'] = dfH.Fecha.dt.day
dfH['Year'] = dfH.Fecha.dt.year
dfH
dfH.latitude_corregida = [x.strip().replace(',','.') for x in dfH.latitude_corregida]
dfH.longitud_corregida = [x.strip().replace(',','.') for x in dfH.longitud_corregida]
dfH['longitud_corregida']=pd.to_numeric(dfH.longitud_corregida, errors='coerce').fillna(0)
dfH['latitude_corregida']=pd.to_numeric(dfH['latitude_corregida'], errors='coerce').fillna(0)
dfH_terrassa=dfH[dfH['poblacion']=='Terrassa']
# dfH_terrassa
dfH_terrassaAW=pd.merge(dfH_terrassa,dfAW,on=['Year','Month','Day'])
dfH_terrassaAW
dfH_terrassaAW.loc[dfH_terrassaAW.patologia == 'BRONQUITIS', 'BRONQUITIS_BOOLEAN'] = '1'  
dfH_terrassaAW.loc[dfH_terrassaAW.patologia != 'BRONQUITIS', 'BRONQUITIS_BOOLEAN'] = '0'


dfH_terrassaAW.describe()
dfH_terrassaAW.dtypes
dfH_terrassaAW['BRONQUITIS_BOOLEAN'] = dfH_terrassaAW.BRONQUITIS_BOOLEAN.astype(int)
dfH_terrassaAW.dtypes
dfH_terrassa2=dfH_terrassa[(dfH_terrassa.latitude_corregida >= 30) & (dfH_terrassa.longitud_corregida >=1.8)]
dfH_terrassa2.plot(kind="scatter", x="longitud_corregida", y="latitude_corregida", alpha=0.4, figsize=(15,7),color='red' )
dfH_terrassaAW.hist('NO2',color='red',figsize=(7,5))
dfH_terrassaAW.hist('BRONQUITIS_BOOLEAN',figsize=(7,5))
dfH_terrassaAW.plot(x='Fecha',y=['NO2'],alpha=0.9,figsize=(25,7))
dfH_terrassaAW.plot(x='Fecha',y=['BRONQUITIS_BOOLEAN'],alpha=0.8,figsize=(25,7), color='red')