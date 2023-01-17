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
def calcula_diferencias_mensuales (dataset, variable):
    dataset[variable+'_pm']  = dataset.groupby('pk_cid')[variable].shift(1)
    dataset['dif_'+variable] = dataset[variable] - dataset[variable+'_pm']
    #dataset['dif_'+variable]  = dataset.groupby('pk_cid')[variable].diff()
    dataset.drop(variable+'_pm',axis=1,inplace=True)

products_file = '/kaggle/input/easymoney/products_df.csv'
products = pd.read_csv(products_file)
products.drop('Unnamed: 0', axis=1, inplace=True)
sd_file = '/kaggle/input/easymoney/sociodemographic_df.csv'
sociodemographic = pd.read_csv(sd_file)
sociodemographic.drop('Unnamed: 0',axis=1, inplace=True)
ca_file = '/kaggle/input/easymoney/commercial_activity_df.csv'
commercial = pd.read_csv(ca_file)
commercial.drop('Unnamed: 0',axis=1, inplace=True)
df_= pd.merge(products,commercial, how="inner",on=['pk_cid','pk_partition' ])
df=pd.merge(df_,sociodemographic, how="inner",on=['pk_cid','pk_partition'])
del products, sociodemographic, commercial,df_
gc.collect()
# we sorted the dataset by pk_cid (Client id), pk_partition (date)
df_sorted = df.sort_values(by=['pk_cid', 'pk_partition'])
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
# We calculate the diferent by months for each Easymoney product. 
for x in productos_easymoney:
    calcula_diferencias_mensuales (df_sorted, x)
# Look at this example of how "dif_" columns works.
df_sorted[df_sorted['pk_cid']==1035440][['pk_cid', 'pk_partition','em_acount','dif_em_acount']]
# we fix this dates
df_sorted.loc[ (df_sorted['entry_date']=='2019-02-29'), 
              'entry_date']='2019-02-28'
df_sorted.loc[ (df_sorted['entry_date']=='2015-02-29'), 
              'entry_date']='2015-02-28'
# we put the dates as dates
for i in ["pk_partition","entry_date"]:
    df_sorted[i]=pd.to_datetime(df_sorted[i], format='%Y-%m-%d')
# We do the subtraction of the position date with the entry date and we convert it in months
df_sorted['mesesAlta']=(df_sorted['pk_partition']-df_sorted['entry_date'])/np.timedelta64(1,'M')
#We  looking for clients with "mesesAlta" < 0
len(df_sorted[df_sorted['mesesAlta']<0])
# And they have no products
df_sorted[df_sorted['mesesAlta']<0].agg({'em_acount':np.sum,
'loans':np.sum,
 'mortgage':np.sum,
 'funds':np.sum,
 'securities':np.sum,
 'long_term_deposit':np.sum,
 'em_account_pp':np.sum,
 'credit_card':np.sum,
 'payroll':np.sum,
 'pension_plan':np.sum,
 'payroll_account':np.sum,
 'emc_account':np.sum,
 'debit_card':np.sum,
 'em_account_p':np.sum,
 'em_acount':np.sum})
# These are clients registered on the 29th, 30th and 31st of the month. 
# According to our process, they have signed in the next month
df_sorted[df_sorted['mesesAlta']<0]['mesesAlta'].value_counts()
# we create the new boolean variable isNewClient
df_sorted['isNewClient']=((df_sorted['mesesAlta'] < 1) & 
                          (df_sorted['mesesAlta'] > 0)).astype(int)
# We can se the isNewClient is in the correct month.
df_sorted[df_sorted['pk_cid']==16502][['pk_cid', 'pk_partition','mesesAlta','isNewClient','em_acount','dif_em_acount']].T
# Look at this example of how "isNewClient" column works.
df_sorted[df_sorted['pk_cid']==16502][['pk_cid', 'pk_partition','isNewClient','mesesAlta','em_acount','dif_em_acount']]
# Look at this example of how "isNewClient" column works.
df_sorted[df_sorted['pk_cid']==16502][['pk_cid', 'pk_partition','isNewClient','mesesAlta','em_acount','dif_em_acount']]
# We paint customer registrations per month
altas_=df_sorted.groupby(['pk_partition'])['isNewClient'].sum()
fig = plt.figure(figsize = (10, 8))
plt.plot(altas_, color = "green", label = "Altas mensuales")
plt.title("Altas mensuales de clientes")
plt.legend()
# When the client is a new user, all the diff fields of this month are 0. But he may have hired
# something in the same month and would not be collected in the diff field.
# For new registrations we match the dif_ columns with the product counters
df_sorted[(df_sorted['dif_em_acount'].isnull()) &
          (df_sorted['isNewClient']==1) &
          (df_sorted['em_acount']==1)].T
# look at this client as example:
df_sorted[df_sorted['pk_cid']==32560][
    ['pk_cid', 'pk_partition','isNewClient','mesesAlta','em_acount','dif_em_acount','debit_card','dif_debit_card']].T
# For new registrations we match the dif_ columns with the product counters
for x in productos_easymoney:
    df_sorted.loc[ (df_sorted['isNewClient']==1) &
                   (df_sorted['dif_'+x].isnull()==True), 
                  'dif_'+x]=df_sorted[x]
# Now the dif_variables are ok. 
df_sorted[df_sorted['pk_cid']==32560][
    ['pk_cid', 'pk_partition','isNewClient','mesesAlta','em_acount','dif_em_acount','debit_card','dif_debit_card']].T
# If the client does not have any product, he is not active
df_sorted['isActive']=((df_sorted['loans']==0) &
                        (df_sorted['mortgage']==0) &
                        (df_sorted['funds']==0) &
                        (df_sorted['securities']==0) &
                        (df_sorted['long_term_deposit']==0) &
                        (df_sorted['em_account_pp']==0) &
                        (df_sorted['credit_card']==0) &
                        (df_sorted['payroll']==0) &
                        (df_sorted['pension_plan']==0) &
                        (df_sorted['payroll_account']==0) &
                        (df_sorted['emc_account']==0) &
                        (df_sorted['debit_card']==0) &
                        (df_sorted['em_account_p']==0) &
                        (df_sorted['em_acount']==0)).astype(int)
# but it is the inverse. We have to do the negation:
df_sorted['isActive']=(df_sorted['isActive']!=1).astype(int)
# Look at this example of how "isNewClient" column works.
df_sorted[df_sorted['pk_cid']==16502][
    ['pk_cid', 'pk_partition','isNewClient','isActive','mesesAlta','em_acount','dif_em_acount']]
# We paint activecustomer registrations per month
altas_=df_sorted.groupby(['pk_partition'])['isActive'].sum()
fig = plt.figure(figsize = (10, 8))
plt.plot(altas_, color = "green", label = "Activos mensuales")
plt.title("Clientes activos mensuales")
plt.legend()
df_sorted['region_code'].isnull().sum()
# There are only 169 clients without informed region
df_sorted[df_sorted['region_code'].isnull()].groupby('pk_cid').size()
df_sorted['region_code'].fillna(-999, inplace=True)
provincias={1:'Alava',
2:'Albacete',
3:'Alicante',
4:'Almeria',
5:'Avila',
6:'Badajoz',
7:'Baleares',
8:'Barcelona',
9:'Burgos',
10:'Caceres',
11:'Cadiz',
12:'Castellon',
13:'Ciudad Real',
14:'Cordoba',
15:'La Coruna',
16:'Cuenca',
17:'Gerona',
18:'Granada',
19:'Guadalajara',
20:'Guipuzcoa',
21:'Huelva',
22:'Huesca',
23:'Jaen',
24:'Leon',
25:'Lerida',
26:'La Rioja',
27:'Lugo',
28:'Madrid',
29:'Malaga',
30:'Murcia',
31:'Navarra',
32:'Orense',
33:'Asturias',
34:'Palencia',
35:'Las Palmas',
36:'Pontevedra',
37:'Salamanca',
38:'Santa Cruz de Tenerife',
39:'Cantabria',
40:'Segovia',
41:'Sevilla',
42:'Soria',
43:'Tarragona',
44:'Teruel',
45:'Toledo',
46:'Valencia',
47:'Valladolid',
48:'Vizcaya',
49:'Zamora',
50:'Zaragoza',
51:'Ceuta',
52:'Melilla',
-999:'Desconocida'}
df_sorted['Provincia']=df_sorted['region_code'].map(provincias)
df_sorted[['region_code','Provincia']]
df_sorted['entry_date'].value_counts(dropna=False)
df_sorted['entry_date'].isnull().sum()
df_sorted['entry_date'].hist()
df_sorted['entry_channel'].value_counts(dropna=False)
df_sorted['entry_channel'].isnull().sum()
df_sorted['entry_channel'].fillna('XXX', inplace=True)
df_sorted['active_customer'].value_counts(dropna=False)
df_sorted['active_customer'].hist()
df_sorted['segment'].value_counts(dropna=False)
df_sorted['segment'].hist()
df_sorted['segment'].fillna('04 - NOINFORMADO',inplace=True)
df_sorted['country_id'].value_counts(dropna=False)
df_sorted['country_id']=pd.Categorical(df_sorted['country_id'], categories=['ES','GB','FR','DE','US','CH','OTHER'])
df_sorted['country_id'].fillna('OTHER',inplace=True)
df_sorted['country_id'].value_counts(dropna=False)
df_sorted['gender'].value_counts(dropna=False)
df_sorted['gender'].fillna('NoInformado',inplace=True)
df_sorted['age'].value_counts(dropna=False)
df_sorted['age'].hist()
df_sorted['deceased'].value_counts(dropna=False)
#df_sorted[df_sorted['deceased']=='S'].head() 
df_sorted[df_sorted['pk_cid']==81958].T
df_sorted['salary'].value_counts(dropna=False)
#df_sorted[ df_sorted['salary']<500000 ]['salary'].hist()
df_sorted['salary'].isnull().sum()
len(df_sorted['salary'])
df_sorted['salary'].describe()
df_sorted.loc[ (df_sorted['salary'] <= 61500.63), 'SalaryQtil'  ]='1Qtil'
df_sorted.loc[ (df_sorted['salary'] > 61500.63) &
               (df_sorted['salary'] <= 88654.65), 'SalaryQtil'  ]='2Qtil'
df_sorted.loc[ (df_sorted['salary'] >  88654.65) &
               (df_sorted['salary'] <= 131669.91), 'SalaryQtil'  ]='3Qtil'
df_sorted.loc[ (df_sorted['salary'] > 131669.91) , 'SalaryQtil'  ]='4Qtil'
df_sorted.loc[ (df_sorted['salary'].isnull()) , 'SalaryQtil'  ]='NOInformado'
df_sorted['salary'].fillna(-999, inplace=True)
df_sorted[['salary','SalaryQtil']]
#df_sorted[ df_sorted['salary']<500000 ]['salary'].hist()
# Easymoney dif product list
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
for x in dif_productos_easymoney:
    df_sorted[x].fillna(0,inplace=True)
df_sorted[dif_productos_easymoney].isnull().sum()
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
df_assets=df_sorted.melt(id_vars=['pk_partition','pk_cid'],
              value_vars=productos_easymoney,
              var_name='Product',
              value_name='Count')
df_assets=df_assets.groupby(['pk_partition','pk_cid']).agg({'Count':np.sum}).reset_index(drop=False)
df_assets.rename(columns={'Count':'totalAssets'}, inplace=True)
len(df_sorted),len(df_assets)
df_sorted=pd.merge(df_sorted,df_assets, how="inner",on=['pk_cid','pk_partition'])
df_sorted['totalAssets'].value_counts().to_frame()
df_sorted[df_sorted['totalAssets']==9]
df_sorted.to_pickle('EasyMoney_base.pkl',compression='zip')
df_sorted = pd.read_pickle('EasyMoney_base.pkl',compression='zip')
