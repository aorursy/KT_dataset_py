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
df_sorted = pd.read_pickle('/kaggle/input/em-1inicio/EasyMoney_Nuevo.pkl',compression='zip')
df_sorted.info()
df_sorted['debit_card_E']=df_sorted['dif_debit_card']*10
df_sorted['em_account_p_E']=df_sorted['dif_em_account_p']*10
df_sorted['em_account_pp_E']=df_sorted['dif_em_account_pp']*10
df_sorted['em_acount_E']=df_sorted['dif_em_acount']*10
df_sorted['emc_account_E']=df_sorted['dif_emc_account']*10
df_sorted['payroll_E']=df_sorted['dif_payroll']*10
df_sorted['payroll_account_E']=df_sorted['dif_payroll_account']*10
productos_easymoney_cuenta_simple = ['debit_card_E','em_account_p_E','em_account_pp_E','em_acount_E','emc_account_E','payroll_E','payroll_account_E']
DFCS = df_sorted.groupby('pk_partition')[productos_easymoney_cuenta_simple].sum()
for i in DFCS:
    DFCS[i] = np.where(DFCS[i]<=0,0,DFCS[i])
DFCS.style.background_gradient(cmap="Reds")
DFCS.groupby('pk_partition')[productos_easymoney_cuenta_simple].sum().plot()
DFCS1=DFCS.sum()
DFCS2=list(DFCS1)
plt.figure(figsize=(200,600))
x = productos_easymoney_cuenta_simple
y = DFCS2
fig, ax = plt.subplots()
ax.set_ylabel('Euros')
ax.set_xlabel('Productos')
ax.set_title('Ventas Productos Cuenta Simple (expresado en euros)')
plt.xticks(rotation=45)
plt.bar(x, y, color='red')
plt.show()
DFCS3 = pd.DataFrame(DFCS1).reset_index()
DFCS3.rename(columns={"index":"Producto"},inplace=True)
DFCS3.rename(columns={0:"Venta_Total"},inplace=True)
DFCS3
DFCS3['Ventas%'] = DFCS3['Venta_Total']/DFCS3['Venta_Total'].sum()
DFCS3
figura=px.treemap(DFCS3, path=[productos_easymoney_cuenta_simple], values="Ventas%", height=500, width=800)
figura.show()
df_sorted['funds_E']=df_sorted['dif_funds']*40
df_sorted['long_term_deposit_E']=df_sorted['dif_long_term_deposit']*40
df_sorted['mortgage_E']=df_sorted['dif_mortgage']*40
df_sorted['pension_plan_E']=df_sorted['dif_pension_plan']*40
df_sorted['securities_E']=df_sorted['dif_securities']*40
productos_easymoney_cuenta_ahorro = ['funds_E','long_term_deposit_E','mortgage_E','pension_plan_E',
                                     'securities_E']
DFCA = df_sorted.groupby('pk_partition')[productos_easymoney_cuenta_ahorro].sum()
for i in DFCA:
    DFCA[i]=np.where(DFCA[i]<=0,0,DFCA[i])
DFCA.style.background_gradient(cmap="PuBu")
DFCA.groupby('pk_partition')[productos_easymoney_cuenta_ahorro].sum().plot()
DFCA1 = DFCA.sum()
DFCA2 = list(DFCA1)
DFCA2
plt.figure(figsize=(20,6))
x = productos_easymoney_cuenta_ahorro
y = DFCA2
fig, ax = plt.subplots()
ax.set_ylabel('Euros')
ax.set_xlabel('Productos')
plt.xticks(rotation=90)
ax.set_title('Ventas Productos Cuenta Ahorro (expresado en euros)')
plt.bar(x, y)
plt.show()
DFCA3 = pd.DataFrame(DFCA1).reset_index()
DFCA3.rename(columns={"index":"Producto"},inplace=True)
DFCA3.rename(columns={0:"Venta_Total"},inplace=True)
DFCA3
DFCA3['Ventas%'] = DFCA3['Venta_Total']/DFCA3['Venta_Total'].sum()
DFCA3
figura=px.treemap(DFCA3, path=[productos_easymoney_cuenta_ahorro], values="Ventas%", height=500, width=800)
figura.show()
df_sorted['loans_E']=df_sorted['dif_loans']*60
df_sorted['credit_card_E']=df_sorted['dif_credit_card']*60
productos_easymoney_cuenta_financiamiento = ['loans_E','credit_card_E']
DFCF = df_sorted.groupby('pk_partition')[productos_easymoney_cuenta_financiamiento].sum()
for i in DFCF:
    DFCF[i]=np.where(DFCF[i]<=0,0,DFCF[i])
DFCF.style.background_gradient(cmap="Reds")
DFCF.groupby('pk_partition')[productos_easymoney_cuenta_financiamiento].sum().plot()
DFCF1 = DFCF.sum()
DFCF2 = list(DFCF1)
plt.figure(figsize=(20,6))
x = productos_easymoney_cuenta_financiamiento
y = DFCF2
fig, ax = plt.subplots()
ax.set_ylabel('Euros')
ax.set_xlabel('Productos')
ax.set_title('Ventas Productos Cuenta Financiamiento (expresado en euros)')
plt.bar(x, y)
plt.show() 
DFCF3 = pd.DataFrame(DFCF1).reset_index()
DFCF3.rename(columns={"index":"Producto"},inplace=True)
DFCF3.rename(columns={0:"Venta_Total"},inplace=True)
DFCF3
DFCF3['Ventas%'] = DFCF3['Venta_Total']/DFCF3['Venta_Total'].sum()
DFCF3
figura=px.treemap(DFCF3, path=[productos_easymoney_cuenta_financiamiento], values="Ventas%", height=500, width=800)
figura.show()
IPC = {'Cuenta_Simple':DFCS3['Venta_Total'].sum(),'Cuenta_Ahorro':DFCA3['Venta_Total'].sum(),'Cuenta_Financiamiento':DFCF3['Venta_Total'].sum()}
IPC1 = pd.DataFrame(list(IPC.items()),columns=['Cuenta','Venta_Total'])
IPC1
IPC1['Venta_Total%']=IPC1['Venta_Total']/IPC1['Venta_Total'].sum()
IPC1
IPC2 = list(IPC1['Venta_Total'])
IPC3 = list(IPC1['Cuenta'])
plt.figure(figsize=(20,6))
x = IPC3
y = IPC2
fig, ax = plt.subplots()
ax.set_ylabel('Euros')
ax.set_xlabel('Cuentas')
ax.set_title('Ventas Cuentas Total (expresado en miles euros)')
plt.bar(x, y)
plt.show() 
figura=px.treemap(IPC1, path=[IPC3], values="Venta_Total%", height=500, width=800)
figura.show()
IPC1['Venta_Total'].sum()
DFT = pd.concat([DFCS3,DFCA3,DFCF3])
DFT.drop(['Ventas%'],axis=1,inplace=True)
DFT
DFT['VentaT%']=DFT['Venta_Total']/ DFT['Venta_Total'].sum()
DFT
DFT['Venta_Total'].sum()
DFT1 = list(DFT['Venta_Total'])
DFT2 = list(DFT['Producto'])
plt.figure(figsize=(20,6))
x = DFT2
y = DFT1
fig, ax = plt.subplots()
ax.set_ylabel('Euros')
ax.set_xlabel('Productos')
plt.xticks(rotation=90)
ax.set_title('Ventas Productos Total (expresado en euros)')
plt.bar(x, y)
plt.show() 
figura=px.treemap(DFT, path=[DFT2], values="VentaT%", height=500, width=800)
figura.show()