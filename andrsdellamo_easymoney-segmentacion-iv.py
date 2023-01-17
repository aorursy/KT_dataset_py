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
clientes_actuales = pd.read_pickle('/kaggle/input/easymoney/EasyMoney_base_segmentacion.pkl',compression='zip')
clientes_actuales[['Cluster','Cluster2']]
clientes_actuales.groupby(['Cluster']).agg({'Cluster2':[np.mean,np.min,np.max]})
clientes_actuales.groupby('Cluster').agg({
                                         'totalAssets':np.mean,
                                         'totalCuentas':np.mean,
                                         'totalAhorro':np.mean,
                                         'totalFinanciacion':np.mean,
                                         'salary':np.mean,
                                         'age':np.mean,
                                         'Cluster':len
                                        })
# Easymoney product list
productos_easymoney=['loans',
 'mortgage',
 'funds',
 'securities',
 'long_term_deposit',
 'em_account_pp',
 'credit_card',
 'payroll',
 'pension_plan',
 'payroll_account',
 'emc_account',
 'debit_card',
 'em_account_p',
 'em_acount',
 'short_term_deposit']
clientes_actuales_=clientes_actuales.pivot_table(index=['pk_partition','Cluster'], 
                                                 values=productos_easymoney,
                                                aggfunc=[np.sum])
clientes_actuales_.columns=['credit_card',
                            'debit_card','em_account_p','em_account_pp',
                            'em_acount','emc_account','funds','loans',
                            'long_term_deposit','mortgage','payroll','payroll_account',
                            'pension_plan','securities','short_term_deposit']
clientes_actuales_.reset_index(drop=False,inplace=True)
clientes_actuales_
clientes_actuales_=clientes_actuales_.melt(id_vars=['pk_partition','Cluster'],
              value_vars=productos_easymoney,
              var_name='Product',
              value_name='Count')
clientes_actuales_
evolucion_horizontal = px.bar(clientes_actuales_, 
                              x="Cluster", y="Count", color='Product', orientation='v', 
                              height=600,title='Productos por Cluster', 
                              color_discrete_sequence = px.colors.cyclical.mygbm)
evolucion_horizontal.show()
clientes_actuales.groupby('Cluster').agg({
                                         'totalAssets':np.mean,'totalCuentas':np.mean,'totalAhorro':np.mean,
                                         'totalFinanciacion':np.mean,'salary':np.mean,'age':np.mean,
                                         'Cluster':len})
clientes_actuales['recomendacion']=0
clientes_actuales_[clientes_actuales_['Cluster']==0]
evolucion_horizontal = px.bar(clientes_actuales_[clientes_actuales_['Cluster']==0], 
                              x="Cluster", y="Count", color='Product', orientation='v', 
                              height=600,title='Productos para Cluster 0', 
                              color_discrete_sequence = px.colors.cyclical.mygbm)
evolucion_horizontal.show()
clientes_actuales[ (clientes_actuales['Cluster']==0) &
                   (clientes_actuales['em_acount']==1) &
                   (clientes_actuales['debit_card']==1) ]
clientes_actuales[ (clientes_actuales['Cluster']==0) &
                   (clientes_actuales['em_acount']==1) &
                   (clientes_actuales['debit_card']==0)  ][['pk_cid','em_acount','debit_card']]
clientes_actuales_[clientes_actuales_['Cluster']==1]
evolucion_horizontal = px.bar(clientes_actuales_[clientes_actuales_['Cluster']==1], 
                              x="Cluster", y="Count", color='Product', orientation='v', 
                              height=600,title='Productos para Cluster 1', 
                              color_discrete_sequence = px.colors.cyclical.mygbm)
evolucion_horizontal.show()
clientes_actuales[ (clientes_actuales['Cluster']==1) &
                   (clientes_actuales['payroll']==1) &
                   (clientes_actuales['debit_card']==1) &
                   (clientes_actuales['pension_plan']==1) &
                   (clientes_actuales['payroll_account']==1)  ][['pk_cid','payroll_account','payroll','pension_plan','debit_card']]
len(clientes_actuales[ (clientes_actuales['Cluster']==1) &
                   (clientes_actuales['pension_plan']==0)  ])
clientes_actuales.loc[ (clientes_actuales['Cluster']==1) &
                       (clientes_actuales['pension_plan']==0)  
                      ,'recomendacion']=11
len(clientes_actuales[ (clientes_actuales['Cluster']==1) &
                   (clientes_actuales['payroll_account']==0)  ])
