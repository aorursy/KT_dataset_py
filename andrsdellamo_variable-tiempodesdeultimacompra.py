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
import numpy as np 
import pandas as pd
import calendar
import matplotlib.pyplot as plt
import seaborn as sns
import squarify
import gc
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import time
import datetime
from datetime import datetime
import calendar

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans

sns.set_style('white')


pd.options.display.float_format = '{:,.2f}'.format
df_sorted = pd.read_pickle('/kaggle/input/easymoney/EasyMoney_base.pkl',compression='zip')
df_sorted.info()
df_sorted['hayAlta']=0
df_sorted.loc[ (df_sorted['dif_debit_card']==1) |        
            (df_sorted['dif_em_account_p']==1) |                
            (df_sorted['dif_em_account_pp']==1) |              
            (df_sorted['dif_em_account_pp']==1) |              
            (df_sorted['dif_em_acount']==1) |                   
            (df_sorted['dif_emc_account']==1) |                 
            (df_sorted['dif_payroll']==1) |                     
            (df_sorted['dif_payroll_account']==1) |             
            (df_sorted['dif_funds']==1) |                        
            (df_sorted['dif_long_term_deposit']==1) |            
            (df_sorted['dif_mortgage']==1) |                    
            (df_sorted['dif_pension_plan']==1) |                
            (df_sorted['dif_securities']==1) |                 
            (df_sorted['dif_short_term_deposit']==1) |          
            (df_sorted['dif_loans']==1) |                      
            (df_sorted['dif_credit_card']==1),'hayAlta']=1 
df_sorted['hayAlta'].value_counts()
df_altas=df_sorted[['pk_cid','pk_partition','hayAlta']].sort_values(by=['pk_cid','pk_partition'])
df_altas
df_altas['diasLastAlta']=df_altas.groupby(['pk_cid','hayAlta'])['pk_partition'].diff()
df_altas
df_sorted = pd.read_pickle('/kaggle/input/easymoney/EasyMoney_base.pkl',compression='zip')
df_altas=df_sorted[['pk_cid','pk_partition','hayAlta']].sort_values(by=['pk_cid','pk_partition'])
df_altas['diasLastAlta']=0
len(df_altas)
j=0
for x in range(1,20):
    if (x%5==0): print(x)
j=0
fechaAlta=pd.Timedelta(days=1)
for x in df_altas.index.tolist():
    # Si es el primer paso por este punto recogemos el pk_cid en la variable cliente
    if (j==0):
        cliente=df_altas.iloc[x]['pk_cid']
    # Vemos si el cliente ha cambiao o sigue siendo el mismo en otra fecha posterior
    if (cliente == df_altas.iloc[x]['pk_cid']):
        # Si es el mismo comprobamos si hay alta ese mes. Si la hay ponemos un 0 en diasLastAlta
        # y metemos la fecha del mes en curso en fechaAlta
        if (df_altas.iloc[x]['hayAlta']==1):
            df_altas.iloc[x,3]=pd.Timedelta(days=0)
            fechaAlta=df_altas.iloc[x]['pk_partition']
        else:
            if (fechaAlta != pd.Timedelta(days=1)):
                df_altas.iloc[x,3]=df_altas.iloc[x]['pk_partition']-fechaAlta
            else:
                df_altas.iloc[x,3]=pd.Timedelta(days=1)
            #print(df_altas.iloc[x]['pk_partition']-fechaAlta)
    else:
        if (df_altas.iloc[x]['hayAlta']==1):
            df_altas.iloc[x,3]=pd.Timedelta(days=0)
            fechaAlta=df_altas.iloc[x]['pk_partition']
        else:
            df_altas.iloc[x,3]=pd.Timedelta(days=1) 
            fechaAlta=pd.Timedelta(days=1)
    cliente=df_altas.iloc[x]['pk_cid']
    j+=1
    if(j % 100 == 0): print(j,time.strftime("%d/%m/%y %H:%M:%S"))
df_altas.head(1800)
len(df_altas['pk_cid'].unique().tolist())
lista1=[]
lista2=[]
lista3=[]
lista4=[]
j=4
for x in df_altas['pk_cid'].unique().tolist():
    if (j % 4 == 0): lista1.append(x)
    if (j % 4 == 1): lista2.append(x)
    if (j % 4 == 2): lista3.append(x)
    if (j % 4 == 3): lista4.append(x)
    j+=1
len (lista1),len (lista2),len (lista3),len (lista4)
print(len (lista1)+len (lista2)+len (lista3)+len (lista4))
df_altas1=df_altas[df_altas['pk_cid'].isin(lista1)]
df_altas2=df_altas[df_altas['pk_cid'].isin(lista2)]
df_altas3=df_altas[df_altas['pk_cid'].isin(lista3)]
df_altas4=df_altas[df_altas['pk_cid'].isin(lista4)]
df_altas1.to_pickle('df_altas1.pkl',compression='zip')
df_altas2.to_pickle('df_altas2.pkl',compression='zip')
df_altas3.to_pickle('df_altas3.pkl',compression='zip')
df_altas4.to_pickle('df_altas4.pkl',compression='zip')

df_altas=pd.read_pickle('../Datos/df_altasFecha_1.pkl',compression='zip')
for x in range(2,9):
    df_=pd.read_pickle('../Datos/df_altasFecha_'+str(x)+'.pkl',compression='zip')
    df_altas=pd.concat([df_altas,df_], axis=0)
df_sorted=pd.merge(df_sorted,df_altas, how="inner",on=['pk_cid','pk_partition' ])
df_sorted = pd.read_pickle('/kaggle/input/easymoney/EasyMoney_base.pkl',compression='zip')
lista_mostrar=['pk_cid','pk_partition','isNewClient','isActive','totalAssets','totalCuentas','totalAhorro','totalFinanciacion','totalIngresos','totalBeneficio','hayAlta','diasDesdeUltimaAlta']
df_sorted[df_sorted['pk_cid']==1515194][lista_mostrar]
df_sorted['diasDesdeUltimaAlta']
df_sorted['diasDesdeUltimaAltaInt']=pd.to_timedelta(df_sorted['diasDesdeUltimaAlta']).dt.days
lista_mostrar=['pk_cid','pk_partition','isNewClient','isActive','totalAssets',
               'totalCuentas','totalAhorro','totalFinanciacion','totalIngresos',
               'totalBeneficio','hayAlta','diasDesdeUltimaAlta','diasDesdeUltimaAltaInt']
df_sorted[df_sorted['pk_cid']==1515194][lista_mostrar]
df_sorted.info()
