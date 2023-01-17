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

sns.set_style('white')

pd.options.display.float_format = '{:,.2f}'.format
def pintar_altas_bajas_mensuales (dataset, variable):
    fig = plt.figure(figsize = (10, 8))
    altas_=dataset[dataset['dif_'+variable] == 1].groupby(['pk_partition'])['dif_'+variable].count()
    bajas_=dataset[dataset['dif_'+variable] == -1].groupby(['pk_partition'])['dif_'+variable].count()
    # No tenemos informacion de esto parfa el primer mes
    #totales_=dataset[dataset['pk_partition'] > '2018-02-01'].groupby(['pk_partition'])['dif_'+variable].sum() 
    totales_=dataset.groupby(['pk_partition'])['dif_'+variable].sum()
    # Pintamos:
    locs, labels = plt.xticks()
    plt.setp(labels,rotation=45)
    plt.plot(altas_, color = "green", label = "Altas mensuales")
    plt.plot(bajas_, color = "red", label = "Bajas mensuales")
    plt.plot(totales_, color = "blue", label = "Total mensuales")
    plt.title("Ventas mensuales de: "+variable)
    plt.legend()
df_sorted = pd.read_pickle('/kaggle/input/easymoney/EasyMoney_base.pkl',compression='zip')
#df_sorted.fillna(0,inplace=True)
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
 'em_acount']
pintar_altas_bajas_mensuales(df_sorted,'em_acount')
pintar_altas_bajas_mensuales(df_sorted,'emc_account')
pintar_altas_bajas_mensuales(df_sorted,'em_account_p')
pintar_altas_bajas_mensuales(df_sorted,'debit_card')
pintar_altas_bajas_mensuales(df_sorted,'credit_card')
dif_productos_easymoney=['dif_loans',
 'dif_mortgage',
 'dif_funds',
 'dif_securities',
 'dif_long_term_deposit',
 'dif_em_account_pp',
 'dif_credit_card',
 'dif_payroll',
 'dif_pension_plan',
 'dif_payroll_account',
 'dif_emc_account',
 'dif_debit_card',
 'dif_em_account_p',
 'dif_em_acount']
df_altas=df_sorted.pivot_table(index='pk_partition',
                      values=dif_productos_easymoney, aggfunc=[np.sum])
df_altas.columns=['credit_card','debit_card',
                            'em_account_p',
                            'em_account_pp',
                             'em_acount',
                            'emc_account',
                            'funds',
                            'loans',
                            'long_term_deposit',
                            'mortgage',
                            'payroll',
                            'payroll_account',
                            'pension_plan',
                            'securities']
df_altas.reset_index(drop=False,inplace=True)
df_altas
df_altas=df_altas.melt(id_vars=['pk_partition'],
              value_vars=df_altas.columns[1:],
              var_name='Product',
              value_name='Count')
df_altas
df_altas.sort_values(by=['pk_partition','Product'],ascending=[True,False],inplace=True)
evolucion_horizontal = px.bar(df_altas, 
                              x="pk_partition", y="Count", color='Product', orientation='v', 
                              height=600,title='Altas por producto', 
                              color_discrete_sequence = px.colors.cyclical.mygbm)
evolucion_horizontal.show()
df_altas=df_sorted[df_sorted['isNewClient']==1].pivot_table(index='pk_partition',
                      values=dif_productos_easymoney, aggfunc=[np.sum])
df_altas.columns=['credit_card','debit_card',
                            'em_account_p',
                            'em_account_pp',
                             'em_acount',
                            'emc_account',
                            'funds',
                            'loans',
                            'long_term_deposit',
                            'mortgage',
                            'payroll',
                            'payroll_account',
                            'pension_plan',
                            'securities']
df_altas.reset_index(drop=False,inplace=True)
df_altas=df_altas.melt(id_vars=['pk_partition'],
              value_vars=df_altas.columns[1:],
              var_name='Product',
              value_name='Count')
df_altas.sort_values(by=['pk_partition','Product'],ascending=[True,False],inplace=True)
evolucion_horizontal = px.bar(df_altas, 
                              x="pk_partition", y="Count", color='Product', orientation='v', 
                              height=600,title='Altas por producto', 
                              color_discrete_sequence = px.colors.cyclical.mygbm)
evolucion_horizontal.show()
df_altas=df_sorted[df_sorted['isNewClient']==0].pivot_table(index='pk_partition',
                      values=dif_productos_easymoney, aggfunc=[np.sum])
df_altas.columns=['credit_card','debit_card',
                            'em_account_p',
                            'em_account_pp',
                             'em_acount',
                            'emc_account',
                            'funds',
                            'loans',
                            'long_term_deposit',
                            'mortgage',
                            'payroll',
                            'payroll_account',
                            'pension_plan',
                            'securities']
df_altas.reset_index(drop=False,inplace=True)
df_altas=df_altas.melt(id_vars=['pk_partition'],
              value_vars=df_altas.columns[1:],
              var_name='Product',
              value_name='Count')
df_altas.sort_values(by=['pk_partition','Product'],ascending=[True,False],inplace=True)
evolucion_horizontal = px.bar(df_altas, 
                              x="pk_partition", y="Count", color='Product', orientation='v', 
                              height=600,title='Altas por producto', 
                              color_discrete_sequence = px.colors.cyclical.mygbm)
evolucion_horizontal.show()
df_altas=df_sorted[df_sorted['isActive']==1].pivot_table(index='pk_partition',
                      values=dif_productos_easymoney, aggfunc=[np.sum])
df_altas.columns=['credit_card','debit_card',
                            'em_account_p',
                            'em_account_pp',
                             'em_acount',
                            'emc_account',
                            'funds',
                            'loans',
                            'long_term_deposit',
                            'mortgage',
                            'payroll',
                            'payroll_account',
                            'pension_plan',
                            'securities']
df_altas.reset_index(drop=False,inplace=True)
df_altas=df_altas.melt(id_vars=['pk_partition'],
              value_vars=df_altas.columns[1:],
              var_name='Product',
              value_name='Count')
df_altas.sort_values(by=['pk_partition','Product'],ascending=[True,False],inplace=True)
evolucion_horizontal = px.bar(df_altas, 
                              x="pk_partition", y="Count", color='Product', orientation='v', 
                              height=600,title='Altas por producto', 
                              color_discrete_sequence = px.colors.cyclical.mygbm)
evolucion_horizontal.show()
