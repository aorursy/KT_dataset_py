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
import numpy as np
import pandas as pd 
from collections import defaultdict
from pprint import pprint
import xgboost as xgb
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
 
matplotlib.rcParams['figure.figsize'] = (8, 6)

import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
seed = 0
np.random.seed(seed)
demographic_cols = ['ncodpers','fecha_alta','ind_empleado','pais_residencia','sexo','age','ind_nuevo','antiguedad','indrel',
 'indrel_1mes','tiprel_1mes','indresi','indext','conyuemp','canal_entrada','indfall',
 'tipodom','cod_prov','ind_actividad_cliente','renta','segmento']

notuse = ["ult_fec_cli_1t","nomprov",'fecha_dato']

product_col = [
 'ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1','ind_cder_fin_ult1','ind_cno_fin_ult1','ind_ctju_fin_ult1',
 'ind_ctma_fin_ult1','ind_ctop_fin_ult1','ind_ctpp_fin_ult1','ind_deco_fin_ult1','ind_deme_fin_ult1',
 'ind_dela_fin_ult1','ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1','ind_plan_fin_ult1',
 'ind_pres_fin_ult1','ind_reca_fin_ult1','ind_tjcr_fin_ult1','ind_valo_fin_ult1','ind_viv_fin_ult1','ind_nomina_ult1',
 'ind_nom_pens_ult1','ind_recibo_ult1']
df_train = pd.read_csv('../input/datamulticlass-6-withpast2/DataMulticlass_6_withpast2.csv')
df_test = pd.read_csv('../input/testset-withpast3/TestSet_withpast3.csv')

pd.set_option('display.max_columns', None)

def filter_data(df):
    df = df[df['ind_nuevo'] == 0]
    df = df[df['antiguedad'] != -999999]
    df = df[df['indrel'] == 1]
    df = df[df['indresi'] == 'S']
    df = df[df['indfall'] == 'N']
    df = df[df['tipodom'] == 1]
    df = df[df['ind_empleado'] == 'N']
    df = df[df['pais_residencia'] == 'ES']
    df = df[df['indrel_1mes'] == 1]
    df = df[df['tiprel_1mes'] == ('A' or 'I')]
    df = df[df['indext'] == 'N']
filter_data(df_train)

drop_column = ['ind_nuevo','indrel','indresi','indfall','tipodom','ind_empleado','pais_residencia','indrel_1mes','indext','conyuemp','fecha_alta','tiprel_1mes']

df_train.drop(drop_column, axis=1, inplace = True)
df_test.drop(drop_column, axis=1, inplace = True)
df_test["renta"]   = pd.to_numeric(df_test["renta"], errors="coerce")
unique_prov = df_test[df_test.cod_prov.notnull()].cod_prov.unique()
grouped = df_test.groupby("cod_prov")["renta"].median()

def impute_renta(df):
    df["renta"]   = pd.to_numeric(df["renta"], errors="coerce")       
    for cod in unique_prov:
        df.loc[df['cod_prov']==cod,['renta']] = df.loc[df['cod_prov']==cod,['renta']].fillna({'renta':grouped[cod]}).values
    df.renta.fillna(df_test["renta"].median(), inplace=True)
    
impute_renta(df_train)
impute_renta(df_test)
def drop_na(df):
    df.dropna(axis = 0, subset = ['ind_actividad_cliente'], inplace = True)
    
drop_na(df_train)
# These column are categories feature, I'll transform them using get_dummy
dummy_col = ['sexo','canal_entrada','cod_prov','segmento']
dummy_col_select = ['canal_entrada','cod_prov']
limit = int(0.01 * len(df_train.index))
use_dummy_col = {}

for col in dummy_col_select:
    trainlist = df_train[col].value_counts()
    use_dummy_col[col] = []
    for i,item in enumerate(trainlist):
        if item > limit:
            use_dummy_col[col].append(df_train[col].value_counts().index[i])
def get_dummy(df):
    for col in dummy_col_select:
        for item in df[col].unique(): 
            if item not in use_dummy_col[col]:
                row_index = df[col] == item
                df.loc[row_index,col] = np.nan
    return pd.get_dummies(df, prefix=dummy_col, columns = dummy_col)
    
df_train = get_dummy(df_train)
df_test = get_dummy(df_test)
def clean_age(df):
    df["age"]   = pd.to_numeric(df["age"], errors="coerce")
    max_age = 80 
    log_max_age = np.log(max_age) 
    square_max_age  = np.square(max_age)
    df["age"]   = df['age'].apply(lambda x: min(x ,max_age))
    df["log_age"]   = df['age'].apply(lambda x: round(np.log10(x+1)/log_max_age, 6))
    df["square_age"]   = df['age'].apply(lambda x: round(np.square(x)/square_max_age, 6))
    df["age"]   = df['age'].apply(lambda x: round( x/max_age, 6))

def clean_renta(df):
    max_renta = 1.0e6
    log_max_renta = np.log(max_renta) 
    square_max_renta  = np.square(max_renta)
    df["renta"]   = df['renta'].apply(lambda x: min(x ,max_renta))
    df["log_renta"]   = df['renta'].apply(lambda x: round(np.log10(x+1)/log_max_renta, 6))
    df["square_renta"]   = df['renta'].apply(lambda x: round(np.square(x)/square_max_renta, 6))
    df["renta"]   = df['renta'].apply(lambda x: round( x/max_renta, 6))
    
def clean_antigue(df):
    df["antiguedad"]   = pd.to_numeric(df["antiguedad"], errors="coerce")
    df["antiguedad"] = df["antiguedad"].replace(-999999, df['antiguedad'].median())
    max_antigue = 256
    log_max_antigue = np.log(max_antigue) 
    square_max_antigue  = np.square(max_antigue)
    df["antiguedad"]   = df['antiguedad'].apply(lambda x: min(x ,max_antigue))
    df["log_antiguedad"]   = df['antiguedad'].apply(lambda x: round(np.log10(x+1)/log_max_antigue, 6))
    df["square_antiguedad"]   = df['antiguedad'].apply(lambda x: round(np.square(x)/square_max_antigue, 6))
    df["antiguedad"]   = df['antiguedad'].apply(lambda x: round( x/max_antigue, 6))
clean_age(df_train)
clean_age(df_test)

clean_renta(df_train)
clean_renta(df_test)

clean_antigue(df_train)
clean_antigue(df_test)
product_col_5 = [col for col in df_train.columns if '_ult1_5' in col]
product_col_4 = [col for col in df_train.columns if '_ult1_4' in col]
product_col_3 = [col for col in df_train.columns if '_ult1_3' in col]
product_col_2 = [col for col in df_train.columns if '_ult1_2' in col]
product_col_1 = [col for col in df_train.columns if '_ult1_1' in col]

df_train['tot5'] = df_train[product_col_5].sum(axis=1)
df_test['tot5'] = df_test[product_col_5].sum(axis=1)
#df_train['tot4'] = df_train[product_col_4].sum(axis=1)
#df_test['tot4'] = df_test[product_col_4].sum(axis=1)
#df_train['tot3'] = df_train[product_col_3].sum(axis=1)
#df_test['tot3'] = df_test[product_col_3].sum(axis=1)
#df_train['tot2'] = df_train[product_col_2].sum(axis=1)
#df_test['tot2'] = df_test[product_col_2].sum(axis=1)
#df_train['tot1'] = df_train[product_col_1].sum(axis=1)
#df_test['tot1'] = df_test[product_col_1].sum(axis=1)
for col in product_col[2:]:
    df_train[col+'_past'] = (df_train[col+'_5']+df_train[col+'_4']+df_train[col+'_3']+df_train[col+'_2']+df_train[col+'_1'])/5
    df_test[col+'_past'] = (df_test[col+'_5']+df_test[col+'_4']+df_test[col+'_3']+df_test[col+'_2']+df_test[col+'_1'])/5
for pro in product_col[2:]:
    df_train[pro+'_past'] = df_train[pro+'_past']*(1-df_train[pro+'_5'])
    df_test[pro+'_past'] = df_test[pro+'_past']*(1-df_test[pro+'_5'])
for col in product_col[2:]:
    for month in range(2,6):
        df_train[col+'_'+str(month)+'_diff'] = df_train[col+'_'+str(month)] - df_train[col+'_'+str(month-1)]
        df_test[col+'_'+str(month)+'_diff'] = df_test[col+'_'+str(month)] - df_test[col+'_'+str(month-1)]
        df_train[col+'_'+str(month)+'_add'] = df_train[col+'_'+str(month)+'_diff'].apply(lambda x: max(x,0))
        df_test[col+'_'+str(month)+'_add'] = df_test[col+'_'+str(month)+'_diff'].apply(lambda x: max(x,0))
product_col_5_diff = [col for col in df_train.columns if '5_diff' in col]
product_col_4_diff = [col for col in df_train.columns if '4_diff' in col]
product_col_3_diff = [col for col in df_train.columns if '3_diff' in col]
product_col_2_diff = [col for col in df_train.columns if '2_diff' in col]

product_col_5_add = [col for col in df_train.columns if '5_add' in col]
product_col_4_add = [col for col in df_train.columns if '4_add' in col]
product_col_3_add = [col for col in df_train.columns if '3_add' in col]
product_col_2_add = [col for col in df_train.columns if '2_add' in col]

product_col_all_diff = [col for col in df_train.columns if '_diff' in col]
product_col_all_add = [col for col in df_train.columns if '_add' in col]
df_train['tot5_add'] = df_train[product_col_5_add].sum(axis=1)
df_test['tot5_add'] = df_test[product_col_5_add].sum(axis=1)
df_train['tot4_add'] = df_train[product_col_4_add].sum(axis=1)
df_test['tot4_add'] = df_test[product_col_4_add].sum(axis=1)
df_train['tot3_add'] = df_train[product_col_3_add].sum(axis=1)
df_test['tot3_add'] = df_test[product_col_3_add].sum(axis=1)
df_train['tot2_add'] = df_train[product_col_2_add].sum(axis=1)
df_test['tot2_add'] = df_test[product_col_2_add].sum(axis=1)
cols = list(df_train.drop(['target','ncodpers']+product_col_all_diff+product_col_all_add, 1).columns.values)

id_preds = defaultdict(list)
ids = df_test['ncodpers'].values

# predict model 
y_train = pd.get_dummies(df_train['target'].astype(int))
x_train = df_train[cols]
    
# create model
model = Sequential()
model.add(Dense(150, input_dim=len(cols), init='uniform', activation='relu'))
model.add(Dense(22, init='uniform', activation='softmax'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['categorical_accuracy'])

#model.fit(x_train.as_matrix(), y_train.as_matrix(), validation_split=0.2, nb_epoch=150, batch_size=10)
model.fit(x_train.as_matrix(), y_train.as_matrix(), nb_epoch=150, batch_size=10)

x_test = df_test[cols]
x_test = x_test.fillna(0) 
        
p_test = model.predict(x_test.as_matrix())
        
for id, p in zip(ids, p_test):
    #id_preds[id] = list(p)
    id_preds[id] = [0,0] + list(p)
df_recent =  pd.read_csv('../input/df-recent/df_recent.csv')

sample = pd.read_csv('../input/sample-submission/sample_submission.csv')

# check if customer already have each product or not. 
already_active = {}
for row in df_recent.values:
    row = list(row)
    id = row.pop(0)
    active = [c[0] for c in zip(tuple(product_col), row) if c[1] > 0]
    already_active[id] = active

# add 7 products(that user don't have yet), higher probability first -> train_pred   
train_preds = {}
for id, p in id_preds.items():
    preds = [i[0] for i in sorted([i for i in zip(tuple(product_col), p) if i[0] not in already_active[id]],
                                  key=lambda i:i [1], 
                                  reverse=True)[:7]]
    train_preds[id] = preds
    
test_preds = []
for row in sample.values:
    id = row[0]
    p = train_preds[id]
    test_preds.append(' '.join(p))
sample['added_products'] = test_preds
sample.to_csv('Keras1.csv', index=False)