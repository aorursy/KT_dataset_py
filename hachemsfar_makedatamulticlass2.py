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
import time
from datetime import datetime
# Any results you write to the current directory are saved as output.
cat_col = ['fecha_dato', 'ncodpers','ind_empleado','pais_residencia','sexo','age','fecha_alta','ind_nuevo','antiguedad','indrel', 'indrel_1mes','tiprel_1mes','indresi','indext','conyuemp','canal_entrada','indfall', 'tipodom','cod_prov','ind_actividad_cliente','renta','segmento']

notuse = ["ult_fec_cli_1t","nomprov"]

product_col = [
 'ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1','ind_cder_fin_ult1','ind_cno_fin_ult1','ind_ctju_fin_ult1',
 'ind_ctma_fin_ult1','ind_ctop_fin_ult1','ind_ctpp_fin_ult1','ind_deco_fin_ult1','ind_deme_fin_ult1',
 'ind_dela_fin_ult1','ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1','ind_plan_fin_ult1',
 'ind_pres_fin_ult1','ind_reca_fin_ult1','ind_tjcr_fin_ult1','ind_valo_fin_ult1','ind_viv_fin_ult1','ind_nomina_ult1',
 'ind_nom_pens_ult1','ind_recibo_ult1']

product_col = product_col[2:]

train_cols = cat_col + product_col

df_train = pd.read_csv('../input/train_ver2.csv',usecols=train_cols)
pd.set_option('display.max_columns', None)
df_train['fecha_dato'] = pd.to_datetime(df_train['fecha_dato'], format='%Y-%m-%d', errors='ignore')
month = 6

df_train_curr = df_train.loc[df_train['fecha_dato']=='2015-06-28',:]
df_train_5 = df_train.loc[df_train['fecha_dato']=='2015-05-28', product_col+['ncodpers']]
df_train_4 = df_train.loc[df_train['fecha_dato']=='2015-04-28', product_col+['ncodpers']]
df_train_3 = df_train.loc[df_train['fecha_dato']=='2015-03-28', product_col+['ncodpers']]
df_train_2 = df_train.loc[df_train['fecha_dato']=='2015-02-28', product_col+['ncodpers']]
df_train_1 = df_train.loc[df_train['fecha_dato']=='2015-01-28', product_col+['ncodpers']]
dfm = pd.merge(df_train_curr,df_train_5, how='left', on=['ncodpers'], suffixes=('', '_5'))
dfm = pd.merge(dfm,df_train_4, how='left', on=['ncodpers'], suffixes=('', '_4'))
dfm = pd.merge(dfm,df_train_3, how='left', on=['ncodpers'], suffixes=('', '_3'))
dfm = pd.merge(dfm,df_train_2, how='left', on=['ncodpers'], suffixes=('', '_2'))
dfm = pd.merge(dfm,df_train_1, how='left', on=['ncodpers'], suffixes=('', '_1'))
dfm.head()
prevcols = [col for col in dfm.columns if '_ult1_'+str(month-1) in col]
currcols = [col for col in dfm.columns if '_ult1' == col[-5:]]
all_product_col = [col for col in dfm.columns if '_ult1' in col]

for col in all_product_col:
    dfm[col].fillna(0, inplace=True)
for col in currcols:
    dfm[col] = dfm[col] - dfm[col+'_'+str(month-1)]
    dfm[col] = dfm[col].apply(lambda x: max(x,0))
dfm = dfm[dfm[currcols].sum(axis=1) >0]
print(dfm[currcols].sum(axis = 1).value_counts())
dfm = dfm.reset_index(drop=True) 
print(dfm.shape)
data = []

for index, row in dfm.iterrows():
    if index%1000 == 0:
        print(index)
        print('finish: ', time.strftime('%a %H:%M:%S'))
    if row[currcols].sum() > 0:
        for i,col in enumerate(currcols):
            if row[col] == 1:
                row['target'] = float(currcols.index(col))
                data.append(list(row))
                
df_new = pd.DataFrame(data, columns = list(dfm.columns.values) + ['target'])
df_new.drop(currcols+['fecha_dato'], axis=1, inplace=True)
df_new.shape
print(df_new['target'].value_counts())
df_new.to_csv('DataMulticlass_'+ str(month) + '_withpast2.csv', index=False)