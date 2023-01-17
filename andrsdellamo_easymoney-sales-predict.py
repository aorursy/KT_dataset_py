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

from sklearn import model_selection # model assesment and model selection strategies
from sklearn import metrics # model evaluation metrics
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.tree import export_graphviz
import graphviz
from sklearn.preprocessing import LabelEncoder,MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


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
#df_sorted = pd.read_pickle('/kaggle/input/easymoney/EasyMoney_base.pkl',compression='zip')
df_sorted = df.sort_values(by=['pk_cid', 'pk_partition'])
del products, sociodemographic, commercial, df
gc.collect()
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
for x in productos_easymoney:
    calcula_diferencias_mensuales (df_sorted, x)
# las ponemos a 2019-02-28
df_sorted.loc[ (df_sorted['entry_date']=='2019-02-29'), 
              'entry_date']='2019-02-28'
df_sorted.loc[ (df_sorted['entry_date']=='2015-02-29'), 
              'entry_date']='2015-02-28'
# Vamos a ponerlos como fechas
for i in ["pk_partition","entry_date"]:
    df_sorted[i]=pd.to_datetime(df_sorted[i], format='%Y-%m-%d')
# Vamos a restar las dos y lo ponemos en mesess:
df_sorted['mesesAlta']=(df_sorted['pk_partition']-df_sorted['entry_date'])/np.timedelta64(1,'M')

# Creamos el campo isNewClient
df_sorted['isNewClient']=((df_sorted['mesesAlta'] < 1) & 
                          (df_sorted['mesesAlta'] > 0)).astype(int)
# Cuando es usuario nuevo todos los campos diff de ses mes estan a 0. Pero el puede haber contratado 
# algo en ese mismo mes y no estaria recogido en el campo diff. 
# Para las altas nuevas igualamos los dif con los contadores del producto
for x in productos_easymoney:
    df_sorted.loc[ (df_sorted['isNewClient']==1) &
                   (df_sorted['dif_'+x].isnull()==True), 
                  'dif_'+x]=df_sorted[x]
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
# pero nos queda al reves, hacemos la negacion
df_sorted['isActive']=(df_sorted['isActive']!=1).astype(int)
#df_sorted.to_pickle('./dataset_base.pkl')
#df_sorted =pd.read_pickle('./dataset_base.pkl')
df_sorted.info(verbose=False)
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
del df_assets
# Leememos directamente todo lo anterior del pickle generado en el notebook: Easymoney_first_steps:
df_sorted = pd.read_pickle('/kaggle/input/easymoney/EasyMoney_base.pkl',compression='zip')
df_sorted.isnull().sum()
df_sorted['payroll'].fillna(0,inplace=True)
df_sorted['pension_plan'].fillna(0,inplace=True)
df_sorted
vars_colums=['pk_cid','pk_partition',
'entry_date',             
'entry_channel',                 
'active_customer',              
'segment',                     
'country_id',                  
'region_code',                  
'gender',                         
'age',                            
'deceased',              
'salary',  
'mesesAlta',                   
'isNewClient',                    
'isActive',
'em_acount',
'totalAssets']
#'debit_card'] 
delta_productos_easymoney=[
 'dif_em_acount']
df_altas=df_sorted.melt(id_vars=vars_colums,
              value_vars=delta_productos_easymoney,
              var_name='Product',
              value_name='Count')
df_altas.isnull().sum()
del df_sorted
gc.collect()
lista_fechas=['2018-01-28',
'2018-02-28',
'2018-03-28',
'2018-04-28',
'2018-05-28',
'2018-06-28',
'2018-07-28',
'2018-08-28',
'2018-09-28',
'2018-10-28',
'2018-11-28',
'2018-12-28',
'2019-01-28',
'2019-02-28',
'2019-03-28',
'2019-04-28',
'2019-05-28',             
'2019-06-28']
# Solo clientes activos el ultimo mes:
lista_clientes=df_altas[(df_altas['isActive']==1) & 
                        (df_altas['pk_partition']=='2019-05-28')]["pk_cid"].unique().tolist()
# Solo uno por razones de memoria del Kernel
lista_productos=['dif_em_acount']
cartesian_product = pd.MultiIndex.from_product([lista_fechas, 
                                                lista_clientes , 
                                                lista_productos], names = ["pk_partition", "pk_cid", "Product"])
len(cartesian_product)
cartesian_product
full_df = pd.DataFrame(index = cartesian_product).reset_index()
full_df.tail()
# Ponemos el campo pk_partition a tipo fecha
full_df['pk_partition']=pd.to_datetime(full_df['pk_partition'], format='%Y-%m-%d')
full_df.groupby('pk_partition')['pk_cid'].size()
full_df = pd.merge(full_df, df_altas, on = ["pk_partition", "pk_cid", "Product"], how = 'left')
del df_altas
full_df.groupby('pk_partition')['pk_cid'].size()
# Borramos todos los nulos que se generan por el producto cartesiano:
full_df.drop (full_df[ (full_df['pk_partition']!='2019-06-28') &
                       (full_df['entry_date'].isnull()) ].index, axis=0, inplace=True)
# Comprobamos el borrado:
full_df.groupby('pk_partition')['pk_cid'].size()
len(full_df)
full_df.drop(full_df[(full_df['isNewClient']==0) &
                        (full_df['isActive']==0) & 
                       (full_df['em_acount']==0) &
                       (full_df['Count']==0) ].index ,axis=0, inplace=True)
full_df.groupby('pk_partition')['pk_cid'].size()
full_df[(full_df['pk_cid']==1231342)]
full_df.info(verbose=False)
lista_actualizar=['entry_date',
 #'entry_channel',
 'active_customer',
 #'segment',
 #'country_id',
 'region_code',
 #'gender',
 'age',
 #'deceased',
 'salary']
 #'mesesAlta']
# Rellenamos entry_date y los campos anteriores para el ultimo mes 06/2019:
for x in lista_actualizar:
    print(x)
    full_df.loc[(full_df['pk_partition']=='2019-06-28'),
                x]=full_df[full_df['pk_partition']=='2019-06-28']['pk_cid'].map(full_df[['pk_cid',x]].groupby('pk_cid')[x].max())
full_df[(full_df['pk_cid']==1231342) & (full_df['pk_partition']=='2019-06-28')]
# Solo hay nulos en el ultimo mes ,el mes a predecir:
full_df.isnull().sum()
#full_df.to_pickle('./full_df_antesFe.pkl')
#full_df =pd.read_pickle('./full_df_antesFe.pkl')
gc.collect()
# Solo 5 meses hacia atras por razones de memoria del kernel
for x in [1,2,3,4,5]:
    full_df['Count_shift_'+str(x)]=full_df.groupby(['pk_cid','Product'])['Count'].shift(x)
    full_df['em_acount_shift_'+str(x)]=full_df.groupby(['pk_cid','Product'])['em_acount'].shift(x)
    full_df['isActive_shift_'+str(x)]=full_df.groupby(['pk_cid','Product'])['isActive'].shift(x)
    full_df['isNewClient_shift_'+str(x)]=full_df.groupby(['pk_cid','Product'])['isNewClient'].shift(x)
    full_df['active_customer_shift_'+str(x)]=full_df.groupby(['pk_cid','Product'])['active_customer'].shift(x)
    full_df['totalAssets_shift_'+str(x)]=full_df.groupby(['pk_cid','Product'])['totalAssets'].shift(x)
# La borro xq no aporta mucho y hacemos espacio:
full_df.drop('country_id',axis=1,inplace=True)
full_df.fillna(-999, inplace=True)
full_df["year"] = full_df["pk_partition"].dt.year
full_df["month"] = full_df["pk_partition"].dt.month
full_df["entry_date_year"] = full_df["entry_date"].dt.year
full_df["entry_date_month"] = full_df["entry_date"].dt.month
dummy_dataset = pd.get_dummies(full_df['gender'],prefix='gender')
dummy_dataset.head()
full_df = pd.concat([full_df,dummy_dataset],axis=1)
del dummy_dataset
gc.collect()
full_df.isnull().sum().sum()
#full_df.to_pickle('./full_df_antesFe.pkl')
train_index = sorted(list(full_df["pk_partition"].unique()))[6:-3]

valida_index = [sorted(list(full_df["pk_partition"].unique()))[-3]]

test_index = [sorted(list(full_df["pk_partition"].unique()))[-2]]
variables_borrar= ['Count' ,
"pk_partition",
'pk_cid',
'Product',
'segment',
'gender', 
'deceased',
'mesesAlta',
'entry_date',
#'country_id',
'entry_channel',
'em_acount',
#'debit_card',
'isNewClient',
'isActive',
'active_customer',
'totalAssets']
X_train = full_df[full_df["pk_partition"].isin(train_index)].drop(variables_borrar, axis=1)
Y_train = full_df[full_df["pk_partition"].isin(train_index)]['Count']

X_valida = full_df[full_df["pk_partition"].isin(valida_index)].drop(variables_borrar, axis=1)
Y_valida = full_df[full_df["pk_partition"].isin(valida_index)]['Count']

X_test = full_df[full_df["pk_partition"].isin(test_index)].drop(variables_borrar, axis=1)
Y_test = full_df[full_df["pk_partition"].isin(test_index)]['Count']
del full_df
gc.collect()
dt = DecisionTreeClassifier(max_depth=7,random_state=42)
dt.fit(X_train,Y_train)
score_train=dt.score(X_train, Y_train)
score_test=dt.score(X_valida, Y_valida)
print('Resultados para: Train: {} - Test: {}'.format(score_train,score_test))

y_valida_pred = pd.DataFrame(dt.predict(X_valida), index=Y_valida.index, columns=['CountPrediction'])
results_df = Y_valida.to_frame().join(y_valida_pred)
results_df[results_df['Count']!=0].sample(40)
results_df['error']=results_df['Count']-results_df['CountPrediction']
results_df['error'].hist()
results_df[results_df['Count']==0]['error'].hist()
results_df[results_df['Count']==-1]['error'].hist()
results_df[results_df['Count']==1]['error'].hist()
results_df[results_df['Count']!=0]['error'].value_counts()
top_features = pd.Series(dt.feature_importances_, index=X_train.columns).sort_values(ascending=False).head(20)
top_features
len(results_df)
results_df[results_df['Count']==0]['error'].value_counts()
results_df[results_df['Count']==-1]['error'].value_counts()
results_df[results_df['Count']==1]['error'].value_counts()
