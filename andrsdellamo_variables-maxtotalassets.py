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
df_sorted = pd.read_pickle('/kaggle/input/easymoney/EasyMoney_base.pkl',compression='zip')
df_sorted.head()
df_sorted.groupby('pk_cid').agg({'totalAssets':np.max,
                                 'totalCuentas':np.max,
                                 'totalAhorro':np.max,
                                 'totalFinanciacion':np.max})
df_max.columns=['maxTotalAssets','maxTotalCuentas','maxTotalAhorro','maxTotalFinanciacion']
df_max.head()
df_sorted['maxTotalAssets']=df_sorted['pk_cid'].map(df_max['maxTotalAssets'])
df_sorted['maxTotalCuentas']=df_sorted['pk_cid'].map(df_max['maxTotalCuentas'])
df_sorted['maxTotalAhorro']=df_sorted['pk_cid'].map(df_max['maxTotalAhorro'])
df_sorted['maxTotalFinanciacion']=df_sorted['pk_cid'].map(df_max['maxTotalFinanciacion'])
df_sorted
lista_mostrar=['pk_cid','pk_partition','isNewClient','isActive','totalAssets',
               'totalCuentas','totalAhorro','totalFinanciacion','totalIngresos',
               'totalBeneficio','hayAlta','diasDesdeUltimaAlta','diasDesdeUltimaAltaInt','maxTotalAssets',
               'maxTotalCuentas','maxTotalAhorro','maxTotalFinanciacion']
df_sorted[df_sorted['pk_cid']==1515194][lista_mostrar]
df_sorted[  (df_sorted['totalAssets'] < df_sorted['maxTotalAssets']) &
            (df_sorted['pk_partition']=='2019-05-28') ][lista_mostrar]
df_sorted[df_sorted['pk_cid']==29008][lista_mostrar]
