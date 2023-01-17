import numpy as np 

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import datetime

from datetime import date

import re

import matplotlib.style as style

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Read data prepared in Task 1

data= pd.read_csv('/kaggle/input/quantium-task2/Combined-data.csv', parse_dates=['DATE'], index_col=0)
data.shape
data.head()
#Create new column with Month and Year

data['MONTH_YR']= data['DATE'].dt.strftime('%B-%Y')
store_data = pd.DataFrame(columns=['STORE_NBR','MONTH_YR','TOT_SALES','CUST_NBR'])
store_data['STORE_NBR']=data['STORE_NBR'].unique().repeat(12)
store_data['MONTH_YR']=np.tile(data['MONTH_YR'].unique(),len (data['STORE_NBR'].unique()))

store_data ['MONTH']= store_data['MONTH_YR'].map(lambda x: datetime.datetime.strptime(x.split('-', 1)[0], "%B").month )

store_data['YEAR']=store_data['MONTH_YR'].map(lambda x: int(x.split('-', 1)[1]))
store_data.head()
df =pd.Series(dtype=float)

for x in data.STORE_NBR.unique():

    for m in data['MONTH_YR'].unique():

        df = df.append( pd.Series(data[(data.STORE_NBR== x) & (data.MONTH_YR == m)]['TOT_SALES'].sum()), ignore_index=True)
store_data['TOT_SALES']=df
df =pd.Series(dtype=float)

for x in data.STORE_NBR.unique():

    for m in data['MONTH_YR'].unique():

        df = df.append( pd.Series(len(data[(data.STORE_NBR == x) & (data.MONTH_YR == m)]['LYLTY_CARD_NBR'].unique())), ignore_index=True)
store_data['CUST_NBR']=df
df =pd.Series(dtype=float)

for x in data.STORE_NBR.unique():

    for m in data['MONTH_YR'].unique():

        df = df.append( pd.Series(data[(data.STORE_NBR ==x) & (data.MONTH_YR ==m)]['TXN_ID'].value_counts().sum()), ignore_index=True)
store_data['TXN_COUNT']=df
store_data.head()
# Check for nulls

store_data.isna().sum().sum()
ptstore_data = store_data[(store_data.YEAR == 2018) | ((store_data.MONTH <2) & (store_data.YEAR == 2019))]
# Control stores

cs_data = pd.DataFrame()

cs_data['TOT_SALES'] = ptstore_data[(ptstore_data.STORE_NBR != 77 ) & (ptstore_data.STORE_NBR != 86) & (ptstore_data.STORE_NBR != 88)]['TOT_SALES'].groupby(ptstore_data['STORE_NBR']).sum()

cs_data['CUST_NBR'] = ptstore_data[(ptstore_data.STORE_NBR != 77 ) & (ptstore_data.STORE_NBR != 86) & (ptstore_data.STORE_NBR != 88)]['CUST_NBR'].groupby(ptstore_data['STORE_NBR']).sum()

cs_data['TXN_COUNT'] = ptstore_data[(ptstore_data.STORE_NBR != 77 ) & (ptstore_data.STORE_NBR != 86) & (ptstore_data.STORE_NBR != 88)]['TXN_COUNT'].groupby(ptstore_data['STORE_NBR']).sum()
cs_data
# Trial stores

ts_data = pd.DataFrame()

ts_data['TOT_SALES'] = ptstore_data['TOT_SALES'].groupby(ptstore_data['STORE_NBR']).sum()

ts_data['CUST_NBR'] = ptstore_data['CUST_NBR'].groupby(ptstore_data['STORE_NBR']).sum()

ts_data['TXN_COUNT'] = ptstore_data['TXN_COUNT'].groupby(ptstore_data['STORE_NBR']).sum()

ts_data = ts_data.loc[[77,86,88]]
ts_data
# Highly correlated control stores with trial store 77

cs_data.corrwith(ts_data.loc[77], axis=1).nlargest(5)
# difference between trial store's performance and each control store's performance

test = cs_data.loc[cs_data.corrwith(ts_data.loc[77], axis=1).nlargest(5).index]

(ts_data.loc[77] - test).sort_values(by = ['TOT_SALES'])
# Highly correlated control stores with trial store 86

cs_data.corrwith(ts_data.loc[86], axis=1).nlargest(5)
# difference between trial store's performance and each control store's performance

test = cs_data.loc[cs_data.corrwith(ts_data.loc[86], axis=1).nlargest(5).index]

(ts_data.loc[86] - test).sort_values(by = ['TOT_SALES'])
# Highly correlated control stores with trial store 88

cs_data.corrwith(ts_data.loc[88], axis=1).nlargest(10)
# difference between trial store's performance and each control store's performance

test = cs_data.loc[cs_data.corrwith(ts_data.loc[88], axis=1).nlargest(6).index]

(ts_data.loc[88] - test).sort_values(by = ['TOT_SALES'])
# Create dataframes for the store pairs

pf1 = pd.concat([cs_data[cs_data.index == 38],ts_data[ts_data.index == 77]])

pf2 = pd.concat([cs_data[cs_data.index == 105],ts_data[ts_data.index == 86]])

pf3 = pd.concat([cs_data[cs_data.index == 130],ts_data[ts_data.index == 88]])
style.use('seaborn-poster')

sns.set_style('darkgrid')
# Visualise the performance of trial stores and the control stores during pre-trial period

plt.rcParams['figure.figsize'] = (24, 9)

df = pd.concat([pf1,pf2,pf3])

df.plot(kind='bar', width=0.5)

plt.title('COMPARISION OF CONTROL STORES AND TRIAL STORES DURING PRE-TRIAL PERIOD',fontweight='bold')

plt.show()
t_data = store_data[((store_data.MONTH >1)&(store_data.YEAR == 2019)) & ((store_data.MONTH <5)&(store_data.YEAR == 2019))]
trial_data = pd.DataFrame()

trial_data['TOT_SALES'] = t_data['TOT_SALES'].groupby(t_data['STORE_NBR']).sum()

trial_data['CUST_NBR'] = t_data['CUST_NBR'].groupby(t_data['STORE_NBR']).sum()

trial_data['TXN_COUNT'] = t_data['TXN_COUNT'].groupby(t_data['STORE_NBR']).sum()
# Create dataframes for the store pairs

pf1 = trial_data.loc[[38,77]]

pf2 = trial_data.loc[[105,86]]

pf3 = trial_data.loc[[130,88]]
# Visualise the performance of trial stores and the control stores during trial period

plt.rcParams['figure.figsize'] = (24, 9)

df = pd.concat([pf1,pf2,pf3])

df.plot(kind='bar', width=0.5)

plt.title('COMPARISION OF CONTROL STORES AND TRIAL STORES DURING TRIAL PERIOD',fontweight='bold')

plt.show()