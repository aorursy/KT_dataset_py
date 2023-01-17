# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
from collections import defaultdict

from pprint import pprint
import xgboost as xgb
# Any results you write to the current directory are saved as output.
demographic_cols = ['fecha_dato',
 'ncodpers','ind_empleado','pais_residencia','sexo','age','fecha_alta','ind_nuevo','antiguedad','indrel',
 'indrel_1mes','tiprel_1mes','indresi','indext','conyuemp','canal_entrada','indfall',
 'tipodom','cod_prov','ind_actividad_cliente','renta','segmento']

notuse = ["ult_fec_cli_1t","nomprov"]

product_col = [
 'ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1','ind_cder_fin_ult1','ind_cno_fin_ult1','ind_ctju_fin_ult1',
 'ind_ctma_fin_ult1','ind_ctop_fin_ult1','ind_ctpp_fin_ult1','ind_deco_fin_ult1','ind_deme_fin_ult1',
 'ind_dela_fin_ult1','ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1','ind_plan_fin_ult1',
 'ind_pres_fin_ult1','ind_reca_fin_ult1','ind_tjcr_fin_ult1','ind_valo_fin_ult1','ind_viv_fin_ult1','ind_nomina_ult1',
 'ind_nom_pens_ult1','ind_recibo_ult1']

product_col_prev = []

for col in product_col:
    product_col_prev.append(col+'_prev')

#product_col = product_col[2:]

train_cols = demographic_cols + ['target'] + product_col_prev
df_train = pd.read_csv('../input/juneextramulticlass/juneExtraMulticlass.csv', usecols=train_cols)
df_test = pd.read_csv('../input/juneextra/juneExtra.csv',usecols = demographic_cols + product_col_prev)
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
limit = int(0.05 * len(df_train.index))
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
    max_age = 80
    df["age"]   = pd.to_numeric(df["age"], errors="coerce")
    df["age"]   = df['age'].apply(lambda x: round(min(x,max_age)/max_age, 4))
    
clean_age(df_train)
clean_age(df_test)

def clean_renta(df):
    max_renta = 1000000
    df["renta"]   = df['renta'].apply(lambda x: round(min(x,max_renta)/max_renta, 4))        
         
clean_renta(df_train)
clean_renta(df_test)

def clean_antigue(df):
    max_antigue = 256
    df["antiguedad"]   = pd.to_numeric(df["antiguedad"], errors="coerce")
    df["antiguedad"] = df['antiguedad'].apply(lambda x: round(min(x,max_antigue)/max_antigue, 4))   

clean_antigue(df_train)
clean_antigue(df_test)

#df_train['tot'] = df_train[product_col_prev].sum(axis=1)
#df_test['tot'] = df_test[product_col_prev].sum(axis=1)
def runXGB(train_X, train_y, colsample_bytree=0.9, max_depth= 6, eta=0.05, min_child_weight=2, subsample=0.9, num_rounds=110):
    param = {}
    param['objective'] = 'multi:softprob'
    param['seed'] = 0
    param['silent'] = 0
    param['eval_metric'] = "mlogloss"
    param['booster'] = 'gbtree'
    param['num_class'] = 24
    param['colsample_bytree'] = colsample_bytree
    param['max_depth'] = max_depth 
    param['eta'] = eta
    param['min_child_weight'] = min_child_weight
    param['subsample'] = subsample
    num_round = num_rounds

    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)
    model = xgb.train(plst, xgtrain, num_round)
    return model
cols = list(df_train.drop(['target','ncodpers',"fecha_dato"], 1).columns.values)

id_preds = defaultdict(list)
ids = df_test['ncodpers'].values

# predict model 
y_train = df_train['target']
x_train = df_train.drop(['target','ncodpers',"fecha_dato"], 1)
    
clf = runXGB(x_train, y_train)
          
x_test = df_test[cols]
x_test.fillna(0,inplace=True)
        
d_test = xgb.DMatrix(x_test)
p_test = clf.predict(d_test)
        
for id, p in zip(ids, p_test):
    id_preds[id] = list(p)
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

sample.shape
sample['added_products'] = test_preds
sample.to_csv('XGBmulticlass.csv', index=False)
