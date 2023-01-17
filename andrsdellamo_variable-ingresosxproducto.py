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
df_sorted .info()
lst_dif=['dif_debit_card',              
'dif_em_account_p',               
'dif_em_account_pp',              
'dif_em_acount',                  
'dif_emc_account',                
'dif_payroll',                   
'dif_payroll_account',            
'dif_funds',                      
'dif_long_term_deposit',         
'dif_mortgage',                   
'dif_pension_plan',               
'dif_securities',                
'dif_short_term_deposit',        
'dif_loans',                      
'dif_credit_card']   
df_=df_sorted.melt(id_vars=['pk_partition','pk_cid'],
              value_vars=lst_dif,
              var_name='Product',
              value_name='Count')
df_
len(df_)
df_['Count'].value_counts()
df_[df_['pk_cid']==17457]['Count'].value_counts()
df=df_[df_['Count']==1]
df.head()
ingresos={'dif_debit_card':10 ,              
'dif_em_account_p':10 ,               
'dif_em_account_pp':10 ,              
'dif_em_acount':10 ,                  
'dif_emc_account':10 ,                
'dif_payroll':10 ,                   
'dif_payroll_account':10 ,            
'dif_funds':40 ,                      
'dif_long_term_deposit':40 ,         
'dif_mortgage':40 ,                   
'dif_pension_plan':40 ,               
'dif_securities':40 ,                
'dif_short_term_deposit':40 ,        
'dif_loans':60 ,                      
'dif_credit_card':60 }
df['ingresosProducto']=df['Product'].map(ingresos)
df.head()
df['ingresos']=df['Count']*df['ingresosProducto']
df.head()
clientes_ingresos=df.groupby(['pk_partition','pk_cid']).agg(
    {'ingresosProducto':np.sum}).reset_index(drop=False)
clientes_ingresos.sort_values(by='ingresosProducto',ascending=False)
len(df_sorted),len(clientes_ingresos)
df_sorted =pd.merge(df_sorted , clientes_ingresos, 
                    how="left",on=['pk_cid','pk_partition'])
df_sorted.isnull().sum()
df_sorted['ingresosProducto'].fillna(0,inplace=True)
df_sorted.isnull().sum().sum()
df_sorted[df_sorted['ingresosProducto']>10].T
df_sorted[df_sorted['ingresosProducto']==0].head()
df_sorted[df_sorted['pk_cid']==17457].T
clientes_ingresos
clientes_ingresos.groupby('pk_cid')['ingresosProducto'].sum().sort_values(ascending=False)
df_sorted['maximoBeneficio']=df_sorted['pk_cid'].map(clientes_ingresos.groupby('pk_cid')['ingresosProducto'].sum().sort_values(ascending=False))
df_sorted[df_sorted['pk_cid']==17457].T
