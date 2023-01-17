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
df_sorted = pd.read_pickle('/kaggle/input/easymoney/EasyMoney_base.pkl',compression='zip')
df_sorted.isnull().sum()
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
# Solo los clientes activos en el ultimo mes:
lista_clientes=df_sorted[(df_sorted['isActive']==1) & 
                        (df_sorted['pk_partition']=='2019-05-28')]["pk_cid"].unique().tolist()
cartesian_product = pd.MultiIndex.from_product([lista_fechas, lista_clientes ], names = ["pk_partition", "pk_cid"])
len(cartesian_product)
cartesian_product
full_df = pd.DataFrame(index = cartesian_product).reset_index()
full_df.tail()
# Ponemos las fechas como fechas
full_df['pk_partition']=pd.to_datetime(full_df['pk_partition'], format='%Y-%m-%d')
full_df.groupby('pk_partition')['pk_cid'].size()
# Hacemos el Merge con los datos de EasyMoney
full_df = pd.merge(full_df,df_sorted , on = ["pk_partition", "pk_cid"], how = 'left')
full_df.groupby('pk_partition')['pk_cid'].size()
del df_sorted
gc.collect()
#full_df = pd.read_pickle('fulldf_base.pkl',compression='zip')
# Hay muchos clientes nulos debido al producto cartesiano:
full_df[full_df['entry_date'].isnull()].groupby('pk_partition')['pk_cid'].size()
full_df
# Borramos todos los nulos que se generan por el producto cartesiano
# Exceptp en el ultimo mes del que no tenemos datos y sobre el que predeciremos:
full_df.drop (full_df[ (full_df['pk_partition']!='2019-06-28') &
                       (full_df['entry_date'].isnull()) ].index, axis=0, inplace=True)
# Ya no hay los 331588 clientes por mes en todos los meses:
full_df.groupby('pk_partition')['pk_cid'].size()
# Vamos a borrar tambien los meses en que mo pasa nada.
# Asi parece que el modelo mejora su rendimiento:
full_df[(full_df['isNewClient']==0) &
                     (full_df['isActive']==0)].T
# Borramos los meses que no pasa nada para que el modelo aprenda a coger emjor los cambios.
#El rendimiento del modelo mejora con esto:
full_df.drop(full_df[(full_df['isNewClient']==0) &
                     (full_df['isActive']==0)].index ,axis=0, inplace=True)
# Quedan menos filas que antes por mes:
full_df.groupby('pk_partition')['pk_cid'].size()
# Rellenamos con informacion el ultimo mes 2019-06-28
# Actualizamos el mes a predecir:2019-06-28 con los valores de los clientes 
lista_actualizar=['entry_date',
 #'entry_channel',
 #'active_customer',
 #'segment',
 #'country_id',
 'region_code',
 #'gender',
 'age',
 #'deceased',
 'salary',
 'mesesAlta']
# Rellenamos entry_date para el ultimo mes:
for x in lista_actualizar:
    print(x)
    full_df.loc[(full_df['pk_partition']=='2019-06-28'),
            x]=full_df[full_df['pk_partition']=='2019-06-28']['pk_cid'].map(full_df[['pk_cid',x]].groupby('pk_cid')[x].max())
# Solo tenemos nulos en el mes a predecir
full_df.isnull().sum()
# full_df= pd.read_pickle('fulldf_base_nandropped.pkl',compression='zip')
full_df.tail()
delta_productos_easymoney=['dif_loans',
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
 'dif_em_acount',
 'dif_em_account_p']
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
 'em_acount',
 'em_account_p']
for y in delta_productos_easymoney:
    print(y)
    for x in [1,2,3,4]:
        print(x)
        full_df[y+'_shift_'+str(x)]=full_df.groupby(['pk_cid'])[y].shift(x)
for y in productos_easymoney:
    print(y)
    for x in [1,2,3,4]:
        print(x)
        full_df[y+'_shift_'+str(x)]=full_df.groupby(['pk_cid'])[y].shift(x)
for x in [1,2,3,4]:
    print(x)
    full_df['isActive_shift_'+str(x)]=full_df.groupby(['pk_cid'])['isActive'].shift(x)
    full_df['isNewClient_shift_'+str(x)]=full_df.groupby(['pk_cid'])['isNewClient'].shift(x)
    full_df['active_customer_shift_'+str(x)]=full_df.groupby(['pk_cid'])['active_customer'].shift(x)
    full_df['totalAssets_shift_'+str(x)]=full_df.groupby(['pk_cid'])['totalAssets'].shift(x)
full_df["year"] = full_df["pk_partition"].dt.year
full_df["month"] = full_df["pk_partition"].dt.month
full_df["entry_date_year"] = full_df["entry_date"].dt.year
full_df["entry_date_month"] = full_df["entry_date"].dt.month
# La quito porque no aporta y al ser categorica me fastidia hacer los fillna
full_df.drop('country_id',axis=1,inplace=True)
# Relleno los nulos generados en las variables LAGS
full_df.fillna(-999,inplace=True)
full_df=pd.read_pickle('/kaggle/input/easymoney/fulldf_base_nandropped_FEOk.pkl',compression='zip')
full_df.head()
full_df[full_df['pk_cid']==18704].head(10).T
gc.collect()
# Entrenaremos el modelo desde el mes 5 donde las variables LAGS no son nulas:
train_index = sorted(list(full_df["pk_partition"].unique()))[5:-3]

valida_index = [sorted(list(full_df["pk_partition"].unique()))[-3]]

test_index = [sorted(list(full_df["pk_partition"].unique()))[-2]]
gc.collect()
# Borramos todas las variables que no vamos a usar o que son autoexplicativas del modelo. 
variables_borrar=['pk_cid', 
"pk_partition",
'segment',
'gender', 
'deceased',
'entry_date',
'mesesAlta',
'entry_channel',
'isNewClient',
'isActive',
'active_customer',
'totalAssets',
'Provincia',
'SalaryQtil',
'dif_loans',
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
 'dif_em_acount',
 'dif_em_account_p',
'loans',
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
 'em_acount',
 'em_account_p']
#del X_train,Y_train, X_valida, Y_valida,X_test,Y_test
gc.collect()
X_train = full_df[full_df["pk_partition"].isin(train_index)].drop(variables_borrar, axis=1)
Y_train = full_df[full_df["pk_partition"].isin(train_index)]['dif_em_acount']

X_valida = full_df[full_df["pk_partition"].isin(valida_index)].drop(variables_borrar, axis=1)
Y_valida = full_df[full_df["pk_partition"].isin(valida_index)]['dif_em_acount']

# No definimos aqui test por limitaciones de memoria del kernel
#X_test = full_df[full_df["pk_partition"].isin(test_index)].drop(variables_borrar, axis = 1)
#Y_test = full_df[full_df["pk_partition"].isin(test_index)]['dif_em_acount']
del full_df
gc.collect()
dt = DecisionTreeClassifier(max_depth=7,random_state=42)
dt.fit(X_train,Y_train)
score_train=dt.score(X_train, Y_train)
score_test=dt.score(X_valida, Y_valida)
print('Resultados para: Train: {} - Test: {}'.format(score_train,score_test))

y_valida_pred = pd.DataFrame(dt.predict(X_valida), index=Y_valida.index, columns=['CountPrediction'])
len(y_valida_pred)
results_df = Y_valida.to_frame().join(y_valida_pred)
results_df['error']=results_df['dif_em_acount']-results_df['CountPrediction']
results_df[results_df['dif_em_acount']!=0].sample(40)
results_df[results_df['dif_em_acount']==1]['error'].hist()
results_df[results_df['dif_em_acount']==1]['error'].value_counts()
results_df[results_df['dif_em_acount']==-1]['error'].hist()
results_df[results_df['dif_em_acount']==-1]['error'].value_counts()
results_df[results_df['dif_em_acount']==0]['error'].hist()
results_df[results_df['dif_em_acount']==0]['error'].value_counts()
top_features = pd.Series(dt.feature_importances_, index=X_train.columns).sort_values(ascending=False).head(20)
top_features
