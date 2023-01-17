# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import datetime

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
!pip install linearmodels

from linearmodels import PanelOLS
print(os.listdir("../input"))
data_a=pd.read_csv("../input/retaildataset/Features data set.csv")

data_b=pd.read_csv('../input/retaildataset/sales data-set.csv')
data_a.head()
data_b.head()
data_b.isnull().sum()
store_sales=pd.DataFrame(data_b.groupby(['Store','Date'])['Weekly_Sales'].sum())

store_sales.head()
store_sales.reset_index(inplace=True)

store_sales.head()
#store_sales=store_sales.set_index(['Store', 'Date'])

#store_sales.head()
store_sales[store_sales['Store']==1].count()
store_sales.shape
data=pd.merge(store_sales,data_a,  how='left', left_on=['Store','Date'], right_on = ['Store','Date'])

data.isnull().sum()
data['Weekly_Sales']=data['Weekly_Sales']/data['CPI']

data['MarkDown1']=data['MarkDown1']/data['CPI']

data['MarkDown2']=data['MarkDown2']/data['CPI']

data['MarkDown3']=data['MarkDown3']/data['CPI']

data['MarkDown4']=data['MarkDown4']/data['CPI']

data['MarkDown5']=data['MarkDown5']/data['CPI']
data[data['Store']==1].count()
data.shape
data['Date']=pd.to_datetime(data['Date'])

data.head()    
df=data
df.sort_values(['Store','Date'],inplace=True)

df.reset_index(inplace=True)

df.drop(['index'],axis=1,inplace=True)
df['month'] = pd.to_datetime(df['Date']).dt.to_period('M')

df.head()
#df_dummy=pd.get_dummies(df['month'])

#df_dummy=df_dummy.rename(columns=lambda s:'mcode'+s)

#df=df.join(df_dummy)

#df.head()
df['IsHoliday_pre'] = df.groupby('Store')['IsHoliday'].shift(1)

df['IsHoliday_next']=df.groupby('Store')['IsHoliday'].shift(-1)

df.head()
df['sales_lag']=df.groupby('Store')['Weekly_Sales'].shift(1)

df.head()
df.isnull().sum()
df=df[df['Date'].isin(pd.date_range(start='20111111', end='20121026'))]



df.isnull().sum()
#correlation matrix

corrmat = df[['MarkDown1', 'MarkDown2','MarkDown3','MarkDown4','MarkDown5']].corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True)
df['FilledMarkdown1']=df['MarkDown1'].fillna(method='pad')

df['FilledMarkdown2']=df['MarkDown2'].fillna(method='pad')

df['FilledMarkdown3']=df['MarkDown3'].fillna(method='pad')

df['FilledMarkdown4']=df['MarkDown4'].fillna(method='pad')

df['FilledMarkdown5']=df['MarkDown5'].fillna(method='pad')


df.isnull().sum()
#df['MarkDown1'].fillna(df.groupby('Store')['MarkDown1'].shift(-1),inplace=True)#fill nan with previous values

#df['MarkDown2'].fillna(df.groupby('Store')['MarkDown2'].shift(-1),inplace=True)

#df['MarkDown3'].fillna(df.groupby('Store')['MarkDown3'].shift(-1),inplace=True)

#df['MarkDown4'].fillna(df.groupby('Store')['MarkDown4'].shift(-1),inplace=True)

#df['MarkDown5'].fillna(df.groupby('Store')['MarkDown5'].shift(-1),inplace=True)

#df.isnull().sum()
df['LogMarkdown1']=np.log(df['FilledMarkdown1'])

df['LogMarkdown2']=np.log(df['FilledMarkdown2'])

df['LogMarkdown3']=np.log(df['FilledMarkdown3'])

df['LogMarkdown4']=np.log(df['FilledMarkdown4'])

df['LogMarkdown5']=np.log(df['FilledMarkdown5'])

df['LogSales']=np.log(df['Weekly_Sales'])

df['LogSales_lag']=np.log(df['sales_lag'])

df['LogCPI']=np.log(df['CPI'])
df[df['LogMarkdown2'].isnull()][['MarkDown2','FilledMarkdown2']]

df['IsHoliday'] = df['IsHoliday'].apply(lambda x: int(x==True))

df['IsHoliday_pre'] = df['IsHoliday_pre'].apply(lambda x: int(x==True))

df['IsHoliday_next'] = df['IsHoliday_next'].apply(lambda x: int(x==True))

df['IsHoliday'].head()
df_test=df.drop(['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5'],axis=1)

df_test.isnull().sum()
df['TMarkdown']=df['FilledMarkdown1']+df['FilledMarkdown2']+df['FilledMarkdown3']+df['FilledMarkdown4']+df['FilledMarkdown5']
df_pn = df.set_index(['Store', 'Date'])

df_pn.head()
#X=[df_pn.LogCPI,df_pn.Unemployment,df_pn.IsHoliday,df_pn.IsHoliday_pre,df_pn.IsHoliday_next,df_pn.LogMarkdown1,df_pn.LogMarkdown2,df_pn.LogMarkdown3,df_pn.LogMarkdown4,df_pn.LogMarkdown5]

X=df_pn[['Unemployment','FilledMarkdown1','FilledMarkdown2','FilledMarkdown3','FilledMarkdown4','FilledMarkdown5']]

y=df_pn['Weekly_Sales']

y1=np.log(y)
X.isnull().sum()
y.rank()
mod = PanelOLS(y,X, entity_effects=True,time_effects=True)

res = mod.fit(cov_type='clustered', cluster_entity=True)

res
df_pn['month']=df_pn['month'].astype('str')
df_pn[['IsHoliday','IsHoliday_pre','IsHoliday_next','FilledMarkdown1', 'FilledMarkdown2','FilledMarkdown3','FilledMarkdown4','FilledMarkdown5']].corr()
df_pn[['IsHoliday','IsHoliday_pre','IsHoliday_next','MarkDown1', 'MarkDown2','MarkDown3','MarkDown4','MarkDown5']].corr()
formula_reg='y ~ 1 + Unemployment+FilledMarkdown1+FilledMarkdown2+FilledMarkdown3+FilledMarkdown4+FilledMarkdown5+FilledMarkdown1*IsHoliday +FilledMarkdown2*IsHoliday+FilledMarkdown3*IsHoliday+FilledMarkdown4*IsHoliday+FilledMarkdown5*IsHoliday+FilledMarkdown1*IsHoliday_pre +FilledMarkdown2*IsHoliday_pre+FilledMarkdown3*IsHoliday_pre+FilledMarkdown4*IsHoliday_pre+FilledMarkdown5*IsHoliday_pre+FilledMarkdown1*IsHoliday_next +FilledMarkdown2*IsHoliday_next+FilledMarkdown3*IsHoliday_next+FilledMarkdown4*IsHoliday_next+FilledMarkdown5*IsHoliday_next+C(month)+ EntityEffects'
formula_reg1='y ~ 1 + sales_lag+Unemployment+FilledMarkdown1+FilledMarkdown2+FilledMarkdown3+FilledMarkdown4+FilledMarkdown5+FilledMarkdown1*IsHoliday +FilledMarkdown2*IsHoliday+FilledMarkdown3*IsHoliday+FilledMarkdown4*IsHoliday+FilledMarkdown5*IsHoliday+FilledMarkdown1*IsHoliday_pre +FilledMarkdown2*IsHoliday_pre+FilledMarkdown3*IsHoliday_pre+FilledMarkdown4*IsHoliday_pre+FilledMarkdown5*IsHoliday_pre+FilledMarkdown1*IsHoliday_next +FilledMarkdown2*IsHoliday_next+FilledMarkdown3*IsHoliday_next+FilledMarkdown4*IsHoliday_next+FilledMarkdown5*IsHoliday_next+C(month)'
df_pn['IsHoliday']
mod1 = PanelOLS.from_formula(formula_reg, df_pn)

res1 = mod1.fit(cov_type='clustered', cluster_entity=True)

res1
mod2 = PanelOLS.from_formula(formula_reg1, df_pn)

res2 = mod2.fit(cov_type='clustered', cluster_entity=True)

res2