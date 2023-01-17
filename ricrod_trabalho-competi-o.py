# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

#joaoavf   marcosvafg

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



df = pd.read_csv("../input/train.csv")

dfTeste = pd.read_csv("../input/test.csv")
df.shape

df.head()
df.select_dtypes('object').columns
y = 'nota_mat'
from sklearn.model_selection import train_test_split

train, test = train_test_split(df, random_state=42)

train.shape, test.shape




df.select_dtypes('object').columns

df.dtypes



df['codigo_mun'] = df['codigo_mun'].str.replace('ID_ID_','')

df['comissionados_por_servidor'] = df['comissionados_por_servidor'].str.replace('%','')

df['area']=df['area'].str.replace(',','')

df['ranking_igm']=df['ranking_igm'].str.replace('º','')

df['densidade_dem']=df['densidade_dem'].str.replace(',','')



dfTeste['codigo_mun'] = dfTeste['codigo_mun'].str.replace('ID_ID_','')

dfTeste['comissionados_por_servidor'] = dfTeste['comissionados_por_servidor'].str.replace('%','')

dfTeste['area']=dfTeste['area'].str.replace(',','')

dfTeste['ranking_igm']=dfTeste['ranking_igm'].str.replace('º','')

dfTeste['densidade_dem']=dfTeste['densidade_dem'].str.replace(',','')



df['codigo_mun']=df['codigo_mun'].values.astype('int64')

df['area']=df['area'].values.astype('float64')

df['densidade_dem']=df['densidade_dem'].values.astype('float64')

dfTeste['codigo_mun']=dfTeste['codigo_mun'].values.astype('int64')

dfTeste['area']=dfTeste['area'].values.astype('float64')

dfTeste['densidade_dem']=dfTeste['densidade_dem'].values.astype('float64')

df.head()

df.select_dtypes('object').columns
for c in['regiao', 'estado',  'porte']:

    df[c] = df[c].astype('category').cat.codes

    

for c in['regiao', 'estado',  'porte']:

    dfTeste[c] = dfTeste[c].astype('category').cat.codes
df.dtypes


for c in['densidade_dem', 'perc_pop_econ_ativa',  'exp_vida','exp_anos_estudo','gasto_pc_saude','hab_p_medico','gasto_pc_educacao','exp_anos_estudo','idhm']:

     df[c] = df[c].fillna((df[c].mean()))



for c in['densidade_dem', 'perc_pop_econ_ativa',  'exp_vida','exp_anos_estudo','gasto_pc_saude','hab_p_medico','gasto_pc_educacao','exp_anos_estudo','idhm']:

     dfTeste[c] = dfTeste[c].fillna((dfTeste[c].mean()))
from sklearn.model_selection import train_test_split

feats = ['exp_anos_estudo'] 
train, test = train_test_split(df, random_state=42)

train.shape, test.shape
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(random_state=42, n_jobs=-1,n_estimators=30000,min_samples_leaf=650)

y='nota_mat'







rf.fit(df[feats], df[y])
from sklearn.metrics import mean_squared_error

valid_preds = rf.predict(test[feats])

mean_squared_error(test[y], valid_preds)**(1/2)



dfTeste[y]=rf.predict(dfTeste[feats])





dfTeste[['codigo_mun', y]].to_csv('rf3.csv', index=False)
dfTeste[y]