from mpl_toolkits.mplot3d import Axes3D

#from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



import warnings

warnings.filterwarnings("ignore")



%matplotlib inline
# Creating a dataframe object from listoftuples

l=os.listdir('../input')

dflistdir = pd.DataFrame(l,columns=['files']) 

dflistdir
#input/Gas Prices in Brazil/

dt= pd.read_table('../input/2004-2019.tsv')

dt.sample(10)
desc=pd.DataFrame()

desc['field']=dt.columns

desc['type']=list(dt.dtypes)

desc

dt.describe()
#import pandas_profiling# Depreciated: pre 2.0.0 version

#pandas_profiling.ProfileReport(dt)
dt['ANO/MES']=pd.to_datetime({'day': 1,'month': dt['MÊS'],'year': dt['ANO']})
dt1=dt[['REGIÃO','ESTADO','PRODUTO','ANO','MÊS','NÚMERO DE POSTOS PESQUISADOS']].groupby(['REGIÃO','ESTADO','PRODUTO','ANO','MÊS']).sum()

dt1.pivot_table(index=['REGIÃO','ESTADO','PRODUTO'], columns=['ANO','MÊS'], values=['NÚMERO DE POSTOS PESQUISADOS'])

dt1
dt1=dt.query('ANO==2019 & MÊS==6')[['REGIÃO','NÚMERO DE POSTOS PESQUISADOS']].groupby(['REGIÃO']).sum().sort_values(by='NÚMERO DE POSTOS PESQUISADOS', ascending=False)

dt1.plot(kind='bar',figsize=(11,7));
dt1=dt.query('ANO==2019 & MÊS==6')[['ESTADO','NÚMERO DE POSTOS PESQUISADOS']].groupby(['ESTADO']).sum().sort_values(by='NÚMERO DE POSTOS PESQUISADOS', ascending=False)

#dt1

dt1.plot(kind='bar',figsize=(11,7));
dt1=dt[['REGIÃO','ESTADO','PRODUTO','ANO','MÊS','PREÇO MÉDIO REVENDA']].groupby(['REGIÃO','ESTADO','PRODUTO','ANO','MÊS']).sum()

dt1.pivot_table(index=['REGIÃO','ESTADO','PRODUTO'], columns=['ANO','MÊS'], values=['PREÇO MÉDIO REVENDA'])
dt1=dt.query("PRODUTO != 'GLP'")

dt1=dt1[['PRODUTO','ANO/MES','PREÇO MÉDIO REVENDA']].groupby(['PRODUTO','ANO/MES']).sum().sort_values(by=['PRODUTO','ANO/MES'], ascending=True)

dt2=dt1.pivot_table(index=['ANO/MES'], columns=['PRODUTO'], values=['PREÇO MÉDIO REVENDA'])

dt2.plot(kind='line',figsize=(11,7)).get_legend().set_bbox_to_anchor((1, 1)) 
dt1=dt.query("PRODUTO == 'GLP'")

dt1=dt1[['PRODUTO','ANO/MES','PREÇO MÉDIO REVENDA']].groupby(['PRODUTO','ANO/MES']).sum().sort_values(by=['PRODUTO','ANO/MES'], ascending=True)

dt2=dt1.pivot_table(index=['ANO/MES'], columns=['PRODUTO'], values=['PREÇO MÉDIO REVENDA'])

#dt2

dt2.plot(kind='line',figsize=(11,7)).get_legend().set_bbox_to_anchor((1, 1)) 
dt1=dt.query("PRODUTO == 'GASOLINA COMUM' & REGIÃO=='NORDESTE'")

dt1=dt1[['ESTADO','ANO/MES','PREÇO MÉDIO REVENDA']].groupby(['ESTADO','ANO/MES']).sum().sort_values(by=['ESTADO','ANO/MES'], ascending=True)

dt2=dt1.pivot_table(index=['ANO/MES'], columns=['ESTADO'], values=['PREÇO MÉDIO REVENDA'])

#dt2

dt2.plot(kind='line',figsize=(11,7)).get_legend().set_bbox_to_anchor((1, 1)) 
dt0=dt.query("ANO==2019 & MÊS==6")[['ESTADO','DESVIO PADRÃO DISTRIBUIÇÃO']]

dt0=dt0[pd.to_numeric(dt0['DESVIO PADRÃO DISTRIBUIÇÃO'], errors='coerce').notnull()]

dt0['DESVIO PADRÃO DISTRIBUIÇÃO']=dt0['DESVIO PADRÃO DISTRIBUIÇÃO'].astype('float64')

dt1=dt0[['ESTADO','DESVIO PADRÃO DISTRIBUIÇÃO']].groupby('ESTADO').mean().sort_values(by='DESVIO PADRÃO DISTRIBUIÇÃO', ascending=True)

dt1.plot(kind='bar',figsize=(11,7));
dt0=dt.query("ANO==2019 & MÊS==6")[['ESTADO','DESVIO PADRÃO REVENDA']]

dt0=dt0[pd.to_numeric(dt0['DESVIO PADRÃO REVENDA'], errors='coerce').notnull()]

dt0['DESVIO_PADRÃO_REVENDA']=dt0['DESVIO PADRÃO REVENDA'].astype('float64')

dt1=dt0[['ESTADO','DESVIO PADRÃO REVENDA']].groupby('ESTADO').mean().sort_values(by='DESVIO PADRÃO REVENDA', ascending=True)



listatop10=dt1.index.values



dt1.plot(kind='bar',figsize=(11,7));

dt0=dt.query("ANO==2019 & MÊS==6")[['ESTADO','DESVIO PADRÃO REVENDA']]

dt0=dt0[pd.to_numeric(dt0['DESVIO PADRÃO REVENDA'], errors='coerce').notnull()]

dt0['DESVIO_PADRÃO_REVENDA']=dt0['DESVIO PADRÃO REVENDA'].astype('float64')

dt1=dt0[['ESTADO','DESVIO PADRÃO REVENDA']].groupby('ESTADO').mean().sort_values(by='DESVIO PADRÃO REVENDA', ascending=True).head(10)

listatop10=dt1.index.values



dt1.plot(kind='bar',figsize=(11,7));

dt1=dt.query("ESTADO in "+str(list(listatop10)))

dt1=dt1[['ESTADO','PRODUTO','ANO','DESVIO PADRÃO REVENDA']].sort_values(by=['ESTADO','PRODUTO','ANO'], ascending=True)



lm=sns.lmplot(data=dt1, x='ANO', y='DESVIO PADRÃO REVENDA',col="PRODUTO", row="ESTADO", fit_reg=True,sharex=False,sharey=False, truncate=True, x_jitter=.1)



fig = lm.fig