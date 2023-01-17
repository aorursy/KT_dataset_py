# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import datetime

import os
print(os.listdir("../input"))
from sklearn import tree
import seaborn as sns
import graphviz 

from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from collections import defaultdict
# Any results you write to the current directory are saved as output.
start_time = datetime.datetime.now()

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

train_cols = demographic_cols + product_col
df_train = pd.read_csv('../input/juneextra/juneExtra.csv', usecols=train_cols)
df_train.head()
df_test = pd.read_csv('../input/santander-product-recommendation/test_ver2.csv', usecols = demographic_cols)
df_test.head()
df_train.head()
df_train = df_train[df_train['ind_nuevo'] == 0]
df_train = df_train[df_train['antiguedad'] != -999999]
df_train = df_train[df_train['indrel'] == 1]
df_train = df_train[df_train['indresi'] == 'S']
df_train = df_train[df_train['indfall'] == 'N']
df_train = df_train[df_train['tipodom'] == 1]
df_train = df_train[df_train['ind_empleado'] == 'N']
df_train = df_train[df_train['pais_residencia'] == 'ES']
df_train = df_train[df_train['indrel_1mes'] == 1]
df_train = df_train[df_train['tiprel_1mes'] == ('A' or 'I')]
df_train = df_train[df_train['indext'] == 'N']
df_train.head()
df_train.isnull().sum()
df_train.tiprel_1mes.value_counts()
drop_column = ['ind_nuevo','indrel','indresi','indfall','tipodom','ind_empleado','pais_residencia','indrel_1mes','indext','conyuemp','fecha_alta','tiprel_1mes']

df_train.drop(drop_column, axis=1, inplace = True)
df_test.drop(drop_column, axis=1, inplace = True)
df_train.head()
df_test.dtypes
df_train["renta"]   = pd.to_numeric(df_train["renta"], errors="coerce")
df_test["renta"]   = pd.to_numeric(df_test["renta"], errors="coerce")
unique_prov = df_test[df_test.cod_prov.notnull()].cod_prov.unique()
unique_prov
grouped = df_test.groupby("cod_prov")["renta"].median()
df_test.head()
for cod in unique_prov:
    df_test.loc[df_test['cod_prov']==cod,['renta']] = df_test.loc[df_test['cod_prov']==cod,['renta']].fillna({'renta':grouped[cod]}).values
    df_train.loc[df_train['cod_prov']==cod,['renta']] = df_train.loc[df_train['cod_prov']==cod,['renta']].fillna({'renta':grouped[cod]}).values
df_test.head()
df_test.renta.fillna(df_test["renta"].median(), inplace=True)
df_test.head()
df_train.isnull().sum()
df_test.isnull().sum()
# not in use
replace_mapping = {
                  "sexo": {"V":0,'H':1},
                  }

df_train.replace(replace_mapping, inplace= True)
df_train.head()
# These column are categories feature, I'll transform them using get_dummy
dummy_col = ['sexo','canal_entrada','cod_prov','segmento']
limit = int(0.05 * len(df_train.index))

for col in dummy_col:
    if len(df_train[col].unique()) > 6:
        trainlist = df_train[col].value_counts()
        print(trainlist)
        use_col = []
        for i,item in enumerate(trainlist):
            if item > limit:
                use_col.append(df_train[col].value_counts().index[i])  
                print(use_col)
        for item in df_train[col].unique(): 
            if item not in use_col:
                row_index = df_train[col] == item
                print(row_index)
                df_train.loc[row_index,col] = np.nan
                print(df_train.loc[row_index,col])
        for item in df_test[col].unique(): 
            if item not in use_col:
                row_index = df_test[col] == item
                df_test.loc[row_index,col] = np.nan 
df_train = pd.get_dummies(df_train, prefix=dummy_col, columns = dummy_col)
df_test = pd.get_dummies(df_test, prefix=dummy_col, columns = dummy_col)
df_train.head()
df_train.head()
df_test.head()
df_train["age"]   = pd.to_numeric(df_train["age"], errors="coerce")
df_test["age"]   = pd.to_numeric(df_test["age"], errors="coerce")

age_group = [[0,19],[19,26],[26,36],[36,41],[41,47],[47,55],[55,60],[60,70],[70,80],[80,170]]

def create_age_group(df):  
    df['age_group'] = np.nan
    for i,age in enumerate(age_group):
        row_index = (df['age'] >= age[0]) & (df['age'] < age[1])
        df.loc[row_index,'age_group'] = i
        
create_age_group(df_train)
create_age_group(df_test)

df_train.drop('age', axis=1, inplace = True)
df_test.drop('age', axis=1, inplace = True)
df_train.head()
renta_group = [[0,50000],[50000,70000],[70000,100000],[100000,150000],[150000,200000],[200000,1000000],[1000000,29000000]]

def create_renta_group(df):  
    df['renta_group'] = np.nan
    for i,renta in enumerate(renta_group):
        row_index = (df['renta'] >= renta[0]) & (df['renta'] < renta[1])
        df.loc[row_index,'renta_group'] = i
        
create_renta_group(df_train)
create_renta_group(df_test)

df_train.drop('renta', axis=1, inplace = True)
df_test.drop('renta', axis=1, inplace = True)
df_train.head()
models = {}
model_preds = {}
id_preds = defaultdict(list)
ids = df_test['ncodpers'].values
for c in product_col:
    if c != 'ncodpers':
        print(c)
        # train model here with june 2015 data
        y_train = df_train[c]
        x_train = df_train.drop(product_col + ['ncodpers',"fecha_dato"], 1)
        
        clf = tree.DecisionTreeClassifier()

        clf.fit(x_train, y_train)
        p_train1 = clf.predict_proba(x_train)[:,1]

        # pridict model with the most recent data
        #y_train2 = df_train[c]
        x_train2 = df_test.drop(['ncodpers',"fecha_dato"], 1)
        p_train = clf.predict_proba(x_train2)[:,1]
        p_train2 = clf.predict_proba(x_train)[:,1]
        
        models[c] = clf
        model_preds[c] = p_train
        
        for id, p in zip(ids, p_train):
            id_preds[id].append(p)
            
        print(roc_auc_score(y_train, p_train2))
        
df_recent = pd.read_csv('../input/santander-product-recommendation/train_ver2.csv',usecols=['ncodpers'] + product_col)
df_recent = df_recent.drop_duplicates(['ncodpers'], keep='last')
df_recent = pd.read_csv('../input/santander-product-recommendation/train_ver2.csv',usecols=['ncodpers'] + product_col)

df_recent.head()
sample = pd.read_csv('../input/santander-product-recommendation/sample_submission.csv')
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
    # Here be dragons
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

sample.to_csv('DTree_demographic_sub.csv', index=False)
print(datetime.datetime.now()-start_time)

dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=x_train.columns,  
                         class_names=c,  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = graphviz.Source(dot_data)  
graph 
dot_data = tree.export_graphviz(clf, out_file='tree.dot', 
                         feature_names=x_train.columns,  
                         class_names=c,  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = graphviz.Source(dot_data)  
graph 
