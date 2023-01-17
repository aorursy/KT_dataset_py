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
cuentas_easymoney = ['debit_card','em_account_p','em_account_pp',
                                     'em_acount','emc_account','payroll','payroll_account']
df_cuentas=df_sorted.melt(id_vars=['pk_partition','pk_cid'],
              value_vars=cuentas_easymoney,
              var_name='Product',
              value_name='Count')
df_cuentas=df_cuentas.groupby(['pk_partition','pk_cid']).agg({'Count':np.sum}).reset_index(drop=False)
df_cuentas.rename(columns={'Count':'totalCuentas'}, inplace=True)
df_cuentas.head()
len(df_sorted),len(df_cuentas)
df_sorted=pd.merge(df_sorted,df_cuentas, how="inner",on=['pk_cid','pk_partition'])
del df_cuentas
gc.collect()
df_sorted['totalCuentas'].value_counts()
ahorro_easymoney = ['funds','long_term_deposit','mortgage','pension_plan',
                                     'securities','short_term_deposit']
df_ahorro=df_sorted.melt(id_vars=['pk_partition','pk_cid'],
              value_vars=ahorro_easymoney,
              var_name='Product',
              value_name='Count')
df_ahorro=df_ahorro.groupby(['pk_partition','pk_cid']).agg({'Count':np.sum}).reset_index(drop=False)
df_ahorro.rename(columns={'Count':'totalAhorro'}, inplace=True)
df_ahorro
df_sorted=pd.merge(df_sorted,df_ahorro, how="inner",on=['pk_cid','pk_partition'])
df_sorted['totalAhorro'].value_counts()
del df_ahorro
financiacion_easymoney= ['loans','credit_card']
df_financiacion=df_sorted.melt(id_vars=['pk_partition','pk_cid'],
              value_vars=financiacion_easymoney,
              var_name='Product',
              value_name='Count')
df_financiacion=df_financiacion.groupby(['pk_partition','pk_cid']).agg({'Count':np.sum}).reset_index(drop=False)
df_financiacion.rename(columns={'Count':'totalFinanciacion'}, inplace=True)
df_sorted=pd.merge(df_sorted,df_financiacion, how="inner",on=['pk_cid','pk_partition'])
del df_financiacion
df_sorted['totalFinanciacion'].value_counts()
variable_segmentacion=['totalAssets','totalCuentas','totalAhorro','totalFinanciacion']
clientes_actuales=df_sorted[(df_sorted['pk_partition']=='2019-05-28') &
          (df_sorted['isActive']==1)]
clientes_actuales[variable_segmentacion]
pipe = Pipeline(
        steps=[
            ('StandardScaler', StandardScaler()),
            ('KMeans', KMeans(n_clusters=6))
        ]
)
pipe.fit(clientes_actuales[variable_segmentacion])
clientes_actuales['Cluster'] = pipe.predict(clientes_actuales[variable_segmentacion])
clientes_actuales.head(3)
clientes_actuales.groupby('Cluster').agg({
                                         'totalAssets':np.mean,
                                         'totalCuentas':np.mean,
                                         'totalAhorro':np.mean,
                                         'totalFinanciacion':np.mean
                                        })
# para evitar error: Data must have variance to compute a kernel density estimate.
sns.distributions._has_statsmodels = False
sns.pairplot(clientes_actuales.sample(10000), vars=variable_segmentacion, hue='Cluster', aspect=1.5)
plt.show()
lista_campos=['pk_cid',
              'isNewClient',
              'isActive',
'debit_card',
'em_account_p','em_account_pp',
'em_acount','emc_account','payroll','payroll_account',
'funds',
'long_term_deposit','mortgage','pension_plan',
'securities','short_term_deposit',
'loans','credit_card',
'totalAssets','totalCuentas','totalAhorro','totalFinanciacion',
'Cluster',
'SalaryQtil']
    

clientes_actuales[clientes_actuales['Cluster']==4][lista_campos].T
clientes_actuales[clientes_actuales['Cluster']==4]['SalaryQtil'].hist()
