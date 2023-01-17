# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# numpy and pandas for data manipulation
import numpy as np
import pandas as pd 

# sklearn preprocessing for dealing with categorical variables
from sklearn.preprocessing import LabelEncoder

# sklearn preprocessing
from sklearn import preprocessing

# For Principal Component Analysis
from sklearn.decomposition import PCA

# File system manangement
import os

# matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings from pandas
import warnings
warnings.filterwarnings('ignore')

plt.style.use('fivethirtyeight')

# Memory management
import gc
inst_pay = pd.read_csv("../input/installments_payments.csv", header=0,  \
                    sep=',', low_memory=False, dtype={'SK_ID_PREV':int, 'SK_ID_CURR':int, 'NUM_INSTALMENT_VERSION':int, 'NUM_INSTALMENT_NUMBER':int, \
                          'DAYS_INSTALMENT':float, 'DAYS_ENTRY_PAYMENT':float, 'AMT_INSTALMENT':float,'AMT_PAYMENT':float})
#IDs con AMT_INSTALMENT nulo y AMT_PAYMENT vacio, se excluyen
null_payments = pd.DataFrame(inst_pay[(inst_pay['AMT_INSTALMENT']==0) & inst_pay['AMT_PAYMENT'].isnull()].SK_ID_PREV.drop_duplicates())
null_payments.set_index('SK_ID_PREV', drop=False)
inst_pay = inst_pay[-inst_pay['SK_ID_PREV'].isin(null_payments['SK_ID_PREV'])]
#Agrupo por SK_ID_PREV, y cuento cuantos IDs repetidos hay, para saber en cuantas cuotas o intentos de cuotas se devolvió.
#Se han encontrado casos en que hay dos lineas porque se pago en dos veces una misma cuota, o incluso se pago mas, en cuyo caso 
#la cuota se repite, el monto a pagar es la diferencia, y lo pagado es lo mismo que lo anterior (pesimo!).

repeated_payments = inst_pay[['SK_ID_PREV','NUM_INSTALMENT_NUMBER','AMT_INSTALMENT']]
repeated_payments = repeated_payments.groupby(['SK_ID_PREV','NUM_INSTALMENT_NUMBER'], as_index=False).agg('count')
repeated_payments['repeated'] = np.where(repeated_payments['AMT_INSTALMENT'] > 1, 1, 0)
repeated_payments = repeated_payments.groupby(['SK_ID_PREV'], as_index=False).agg({'repeated':'sum'})
repeated_payments.set_index('SK_ID_PREV', drop=False)
repeated_payments.head()
#Obtengo por cada SK_ID_PREV la cantidad maxima de instalments.

qty_payments = inst_pay[['SK_ID_PREV','NUM_INSTALMENT_NUMBER']].groupby(['SK_ID_PREV'], as_index=False).agg({'NUM_INSTALMENT_NUMBER':'max'})
qty_payments.set_index('SK_ID_PREV', drop=False)
qty_payments.head()
#Saco el monto a pagar por cuota. Primero agrupo para eliminar duplicados, luego sumo para agregar saldos parciales.

inst_amount = inst_pay[['SK_ID_PREV','NUM_INSTALMENT_NUMBER','AMT_INSTALMENT','AMT_PAYMENT']].groupby(['SK_ID_PREV','NUM_INSTALMENT_NUMBER','AMT_INSTALMENT'], as_index=False).agg('count')
inst_amount = inst_amount.groupby(['SK_ID_PREV','NUM_INSTALMENT_NUMBER'], as_index=False).agg({'AMT_INSTALMENT':'sum'})
inst_amount.set_index('SK_ID_PREV', drop=False)
inst_amount.head()
#Saco el monto pagado por cuota. Primero agrupo para eliminar duplicados, luego sumo para agregar saldos parciales.

payed_amount = inst_pay[['SK_ID_PREV','NUM_INSTALMENT_NUMBER','AMT_INSTALMENT','AMT_PAYMENT']].groupby(['SK_ID_PREV','NUM_INSTALMENT_NUMBER','AMT_PAYMENT'], as_index=False).agg('count')
payed_amount = payed_amount.groupby(['SK_ID_PREV','NUM_INSTALMENT_NUMBER'], as_index=False).agg({'AMT_PAYMENT':'sum'})
payed_amount.set_index('SK_ID_PREV', drop=False)
payed_amount.head()
#Ademas, hay algunos con los installments en cero pero con pagos. Habria que detectar aquellos que tengan TODOS los installments
#en cero e imputarles lo que pagaron, si asumimos que pagaron lo que correspondia. Chequeamos entonces primero que hayan pagado.

no_instalment = pd.merge(left=inst_amount[inst_amount['AMT_INSTALMENT']==0],right=payed_amount, left_on='SK_ID_PREV', right_on='SK_ID_PREV')
no_instalment.set_index('SK_ID_PREV', drop=False)
inst_amount.drop(columns=inst_amount.columns)
inst_amount = inst_pay[['SK_ID_PREV','NUM_INSTALMENT_NUMBER','AMT_INSTALMENT','AMT_PAYMENT']]
inst_amount.set_index('SK_ID_PREV', drop=False)
inst_amount = pd.merge(left=inst_amount,right=no_instalment, left_on='SK_ID_PREV', right_on='SK_ID_PREV', how='left', suffixes=('','right'))
inst_amount['AMT_INSTALMENT'] = np.where(inst_amount['AMT_INSTALMENTright'].isna() | inst_amount['AMT_INSTALMENTright'] ==0,\
                                         inst_amount['AMT_PAYMENT'],inst_amount['AMT_INSTALMENT'])

inst_amount = inst_amount.groupby(['SK_ID_PREV','NUM_INSTALMENT_NUMBER','AMT_INSTALMENT'], as_index=False).agg('count')
inst_amount = inst_amount.groupby(['SK_ID_PREV','NUM_INSTALMENT_NUMBER'], as_index=False).agg({'AMT_INSTALMENT':'sum'})
inst_amount.set_index('SK_ID_PREV', drop=False)
inst_amount.head()
#Calculo atrasos / adelantos en los pagos
delay_advance = pd.merge(left=inst_pay, right=inst_amount, how='left', on=['SK_ID_PREV','NUM_INSTALMENT_NUMBER'], \
                         suffixes=('_L', '_R'))
delay_advance['del_adv'] = np.where(delay_advance['AMT_INSTALMENT_R']==0,0,(delay_advance['DAYS_INSTALMENT']-delay_advance['DAYS_ENTRY_PAYMENT'])\
                                    *(delay_advance['AMT_INSTALMENT_L']/delay_advance['AMT_INSTALMENT_R']))
delay_advance['days_diff'] = delay_advance['DAYS_INSTALMENT'] - delay_advance['DAYS_ENTRY_PAYMENT']
delay_advance['ratio'] = delay_advance['AMT_INSTALMENT_L']/delay_advance['AMT_INSTALMENT_R'] 
delay_advance = delay_advance.groupby(['SK_ID_PREV'], as_index=False).agg({'days_diff':['sum','mean'],'del_adv':['sum','mean'], 'ratio':['mean']})
#monto del prestamo
loan_amount = inst_amount.groupby(['SK_ID_PREV'], as_index=False).agg({'AMT_INSTALMENT':'sum'})
loan_amount = loan_amount.rename(index=str, columns={"AMT_INSTALMENT":"loan"})
loan_amount.set_index('SK_ID_PREV', drop=False)
loan_amount.head()
#Uno, por cada Id, monto a pagar y pagado, y obtengo la diferencia.
inst_amount_total = inst_amount.groupby(['SK_ID_PREV'], as_index=False).agg({'AMT_INSTALMENT':'sum'})

payed_amount_total = payed_amount.groupby(['SK_ID_PREV'], as_index=False).agg({'AMT_PAYMENT':'sum'})


net_debt = pd.merge(left=inst_amount_total,right=payed_amount_total, left_on='SK_ID_PREV', right_on='SK_ID_PREV')
net_debt['balance'] = net_debt['AMT_INSTALMENT'].astype(int)-net_debt['AMT_PAYMENT'].astype(int)
net_debt = net_debt[['SK_ID_PREV','balance']]
net_debt = net_debt.groupby(['SK_ID_PREV'], as_index=False).agg({'balance':['sum','mean']})
#Para crear el Dataset final, agarro todos los IDs sin repetir
instalments = inst_pay[['SK_ID_PREV','SK_ID_CURR']].drop_duplicates()
instalments.set_index('SK_ID_PREV', drop=False)
instalments.head()
instalments = pd.merge(left=instalments,right=repeated_payments, left_on='SK_ID_PREV', right_on='SK_ID_PREV')
instalments = pd.merge(left=instalments,right=qty_payments, left_on='SK_ID_PREV', right_on='SK_ID_PREV')
instalments = pd.merge(left=instalments,right=delay_advance, left_on='SK_ID_PREV', right_on='SK_ID_PREV')
instalments = pd.merge(left=instalments,right=net_debt, left_on='SK_ID_PREV', right_on='SK_ID_PREV')

instalments.head()
instalments.columns = ["_".join(x) for x in instalments.columns.ravel()]
instalments.rename(columns={'S_K___I_D___P_R_E_V': 'SK_ID_PREV', 'S_K___I_D___C_U_R_R': 'SK_ID_CURR','r_e_p_e_a_t_e_d': 'repeated', \
                   'N_U_M___I_N_S_T_A_L_M_E_N_T___N_U_M_B_E_R': 'NUM_INSTALMENT_NUMBER'}, inplace=True)
instalments.head()
prev_app = pd.read_csv("../input/previous_application.csv", header=0, sep=',', low_memory=False)
#Tipos de variables y cantidades
prev_app.dtypes.value_counts()
#variables categoricas y cantidad de valores por cada una
prev_app.select_dtypes('object').apply(pd.Series.nunique, axis = 0)
#Armo datasets separados segun tipo de variable
categoricas = pd.concat([prev_app.SK_ID_PREV, prev_app.select_dtypes('object')], axis=1)
categoricas.set_index('SK_ID_PREV', drop=False)
enteras = prev_app.select_dtypes('int64')
enteras.set_index('SK_ID_PREV', drop=False)
reales = pd.concat([prev_app.SK_ID_PREV,prev_app.select_dtypes('float')], axis=1)
reales.set_index('SK_ID_PREV', drop=False)
reales.head()
#Normalizo y grafico
x = enteras['HOUR_APPR_PROCESS_START'].values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x = x.reshape(-1,1)
x_scaled = min_max_scaler.fit_transform(x)
enteras['HOUR_APPR_PROCESS_START_NORM'] = pd.DataFrame(x_scaled)
plt.hist(enteras['HOUR_APPR_PROCESS_START_NORM'], bins=24)
plt.show()
x = enteras['DAYS_DECISION'].values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x = x.reshape(-1,1)
x_scaled = min_max_scaler.fit_transform(x)
enteras['DAYS_DECISION_NORM'] = pd.DataFrame(x_scaled)
plt.hist(enteras['DAYS_DECISION_NORM'], bins=24)
plt.show()
#plt.hist(enteras['SELLERPLACE_AREA'], bins=50, range=(-1,100))
#plt.show()
x = enteras['SELLERPLACE_AREA'].values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x = x.reshape(-1,1)
x_scaled = min_max_scaler.fit_transform(x)
enteras['SELLERPLACE_AREA_NORM'] = pd.DataFrame(x_scaled)
#Saco logaritmo y normalizo
reales['AMT_ANNUITY_NORMAL'] = np.where((reales['AMT_ANNUITY']==0) | (reales['AMT_ANNUITY'].isnull()), 0.001, reales['AMT_ANNUITY'])
reales['AMT_ANNUITY_NORMAL'] = np.log(reales.AMT_ANNUITY_NORMAL)

x = reales['AMT_ANNUITY_NORMAL'].values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x = x.reshape(-1,1)
x_scaled = min_max_scaler.fit_transform(x)
reales['AMT_ANNUITY_NORMAL'] = pd.DataFrame(x_scaled)
#No se imputaron bien todos los nulos, quedo uno.
reales['AMT_ANNUITY_NORMAL'] = np.where((reales['AMT_ANNUITY_NORMAL']==0) | (reales['AMT_ANNUITY_NORMAL'].isnull()), 0.001, reales['AMT_ANNUITY_NORMAL'])
plt.hist(reales['AMT_ANNUITY_NORMAL'], bins=50)
reales['AMT_APPLICATION_NORMAL'] = np.where((reales['AMT_APPLICATION']==0) | (reales['AMT_APPLICATION'].isnull()), 0.001, reales['AMT_APPLICATION'])
reales['AMT_APPLICATION_NORMAL'] = np.log(reales.AMT_APPLICATION_NORMAL)

x = reales['AMT_APPLICATION_NORMAL'].values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x = x.reshape(-1,1)
x_scaled = min_max_scaler.fit_transform(x)
reales['AMT_APPLICATION_NORMAL'] = pd.DataFrame(x_scaled)
reales['AMT_APPLICATION_NORMAL'] = np.where((reales['AMT_APPLICATION_NORMAL']==0) | (reales['AMT_APPLICATION_NORMAL'].isnull()), 0.001, reales['AMT_APPLICATION_NORMAL'])

plt.hist(reales['AMT_APPLICATION_NORMAL'], bins =50)
#Saco logaritmo y normalizo
reales['AMT_CREDIT_NORMAL'] = np.where((reales['AMT_CREDIT']==0) | (reales['AMT_CREDIT'].isnull()), 0.001, reales['AMT_CREDIT'])
reales['AMT_CREDIT_NORMAL'] = np.log(reales.AMT_CREDIT_NORMAL)

x = reales['AMT_CREDIT_NORMAL'].values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x = x.reshape(-1,1)
x_scaled = min_max_scaler.fit_transform(x)
reales['AMT_CREDIT_NORMAL'] = pd.DataFrame(x_scaled)
reales['AMT_CREDIT_NORMAL'] = np.where((reales['AMT_CREDIT_NORMAL']==0) | (reales['AMT_CREDIT_NORMAL'].isnull()), 0.001, reales['AMT_CREDIT_NORMAL'])

plt.hist(reales['AMT_CREDIT_NORMAL'], bins =50)
reales['AMT_DOWN_PAYMENT'] = np.where((reales['AMT_DOWN_PAYMENT']==0) | (reales['AMT_DOWN_PAYMENT'].isnull()), 0.001, reales['AMT_DOWN_PAYMENT'])
reales['AMT_DOWN_PAYMENT_NORMAL'] = np.log(reales.AMT_DOWN_PAYMENT)
reales['AMT_DOWN_PAYMENT_NORMAL'] = np.where((reales['AMT_DOWN_PAYMENT_NORMAL']==0) | (reales['AMT_DOWN_PAYMENT_NORMAL'].isnull()), 0.001, reales['AMT_DOWN_PAYMENT_NORMAL'])

x = reales['AMT_DOWN_PAYMENT_NORMAL'].values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x = x.reshape(-1,1)
x_scaled = min_max_scaler.fit_transform(x)
reales['AMT_DOWN_PAYMENT_NORMAL'] = pd.DataFrame(x_scaled)

plt.hist(reales['AMT_DOWN_PAYMENT_NORMAL'], bins=20, range=(0,10))
reales['AMT_GOODS_PRICE'] = np.where(reales['AMT_GOODS_PRICE'].isnull(), 0.001, reales['AMT_GOODS_PRICE'])

x = reales['AMT_GOODS_PRICE'].values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x = x.reshape(-1,1)
x_scaled = min_max_scaler.fit_transform(x)
reales['AMT_GOODS_PRICE'] = pd.DataFrame(x_scaled)
reales['AMT_GOODS_PRICE'] = np.where((reales['AMT_GOODS_PRICE']==0) | (reales['AMT_GOODS_PRICE'].isnull()), 0.001, reales['AMT_GOODS_PRICE'])
#One hot encoding de variables categoricas
categoricas_dummies = pd.get_dummies(categoricas)
#Seteo hasta 10 componentes principales
pca = PCA(10)
pca.fit(categoricas_dummies.drop('SK_ID_PREV', axis=1))
plt.plot(pca.explained_variance_ratio_.cumsum())
categoricas_transformadas = pca.fit_transform(categoricas_dummies.drop('SK_ID_PREV', axis=1))
categoricas_transformadas = pd.DataFrame(categoricas_transformadas)
categoricas_transformadas = pd.concat([categoricas_dummies.SK_ID_PREV, categoricas_transformadas], axis=1)
final_prev_app = prev_app[['SK_ID_PREV','SK_ID_CURR']].drop_duplicates()
#Agrego enteras al dataset final
final_enteras = enteras[['SK_ID_PREV','HOUR_APPR_PROCESS_START_NORM','NFLAG_LAST_APPL_IN_DAY', 'SELLERPLACE_AREA_NORM','DAYS_DECISION_NORM']]

final_prev_app = pd.merge(left=final_prev_app, right=final_enteras, how='inner', left_on=['SK_ID_PREV'], \
        right_on=['SK_ID_PREV'])
final_reales = reales[['SK_ID_PREV','RATE_DOWN_PAYMENT','CNT_PAYMENT','DAYS_FIRST_DRAWING','DAYS_FIRST_DUE','DAYS_LAST_DUE_1ST_VERSION',\
                      'DAYS_LAST_DUE','DAYS_TERMINATION','NFLAG_INSURED_ON_APPROVAL','AMT_ANNUITY_NORMAL','AMT_APPLICATION_NORMAL',\
                     'AMT_CREDIT_NORMAL','AMT_DOWN_PAYMENT','AMT_GOODS_PRICE']]
final_prev_app = pd.merge(left=final_prev_app, right=final_reales, how='inner', left_on=['SK_ID_PREV'], \
        right_on=['SK_ID_PREV'])
final_prev_app = pd.merge(left=final_prev_app, right=categoricas_transformadas, how='inner', left_on=['SK_ID_PREV'], \
        right_on=['SK_ID_PREV'])
final_prev_app_and_instalments = pd.merge(left=final_prev_app, right=instalments, how='inner', left_on=['SK_ID_PREV'], \
        right_on=['SK_ID_PREV']) 
final_prev_app_and_instalments = final_prev_app_and_instalments.drop('SK_ID_CURR_y', axis=1)
final_prev_app_and_instalments.rename(columns = {'SK_ID_CURR_x':'SK_ID_CURR'}, inplace = True)
final_prev_app_and_instalments.rename(columns = {0:'pca_0',1:'pca_1',2:'pca_2',3:'pca_3',4:'pca_4',5:'pca_5',\
                                                6:'pca_6',7:'pca_7',8:'pca_8',9:'pca_9'}, inplace = True)
def agg_numeric(df, group_var, df_name):
    """Aggregates the numeric values in a dataframe. This can
    be used to create features for each instance of the grouping variable.
    
    Parameters
    --------
        df (dataframe): 
            the dataframe to calculate the statistics on
        group_var (string): 
            the variable by which to group df
        df_name (string): 
            the variable used to rename the columns
        
    Return
    --------
        agg (dataframe): 
            a dataframe with the statistics aggregated for 
            all numeric columns. Each instance of the grouping variable will have 
            the statistics (mean, min, max, sum; currently supported) calculated. 
            The columns are also renamed to keep track of features created.
    
    """
    # Remove id variables other than grouping variable
    for col in df:
        if col != group_var and 'SK_ID' in col:
            df = df.drop(columns = col)
            
    group_ids = df[group_var]
    numeric_df = df.select_dtypes('number')
    numeric_df[group_var] = group_ids

    # Group by the specified variable and calculate the statistics
    agg = numeric_df.groupby(group_var).agg(['count', 'mean', 'max', 'min', 'sum']).reset_index()

    # Need to create new column names
    columns = [group_var]

    # Iterate through the variables names
    for var in agg.columns.levels[0]:
        # Skip the grouping variable
        if var != group_var:
            # Iterate through the stat names
            for stat in agg.columns.levels[1][:-1]:
                # Make a new column name for the variable and stat
                columns.append('%s_%s_%s' % (df_name, var, stat))

    agg.columns = columns
    return agg
prevapp_and_installments = agg_numeric(final_prev_app_and_instalments,'SK_ID_CURR','prevapp_instalments')
prevapp_and_installments.head()