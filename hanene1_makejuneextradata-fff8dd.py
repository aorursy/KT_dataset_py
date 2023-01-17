# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/"))

# Any results you write to the current directory are saved as output.
train_cols = ['fecha_dato',
 'ncodpers','ind_empleado','pais_residencia','sexo','age','fecha_alta','ind_nuevo','antiguedad','indrel',
 'indrel_1mes','tiprel_1mes','indresi','indext','conyuemp','canal_entrada','indfall',
 'tipodom','cod_prov','ind_actividad_cliente','renta','segmento',
 'ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1','ind_cder_fin_ult1','ind_cno_fin_ult1','ind_ctju_fin_ult1',
 'ind_ctma_fin_ult1','ind_ctop_fin_ult1','ind_ctpp_fin_ult1','ind_deco_fin_ult1','ind_deme_fin_ult1',
 'ind_dela_fin_ult1','ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1','ind_plan_fin_ult1',
 'ind_pres_fin_ult1','ind_reca_fin_ult1','ind_tjcr_fin_ult1','ind_valo_fin_ult1','ind_viv_fin_ult1','ind_nomina_ult1',
 'ind_nom_pens_ult1','ind_recibo_ult1']

notuse = ["ult_fec_cli_1t","nomprov"]

product_col = [
 'ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1','ind_cder_fin_ult1','ind_cno_fin_ult1','ind_ctju_fin_ult1',
 'ind_ctma_fin_ult1','ind_ctop_fin_ult1','ind_ctpp_fin_ult1','ind_deco_fin_ult1','ind_deme_fin_ult1',
 'ind_dela_fin_ult1','ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1','ind_plan_fin_ult1',
 'ind_pres_fin_ult1','ind_reca_fin_ult1','ind_tjcr_fin_ult1','ind_valo_fin_ult1','ind_viv_fin_ult1','ind_nomina_ult1',
 'ind_nom_pens_ult1','ind_recibo_ult1']

df_train = pd.read_csv('../input/train_ver2.csv',usecols=train_cols)
df_recent = df_train[['ncodpers'] + product_col]
df_recent = df_recent.drop_duplicates(['ncodpers'], keep='last')

df_recent.to_csv('df_recent.csv')
pd.set_option('display.max_columns', None)
df_train_may = df_train[df_train['fecha_dato']=='2015-05-28']
df_train_june = df_train[df_train['fecha_dato']=='2015-06-28']

df_train.drop('fecha_dato', axis=1, inplace=True)
dfm = pd.merge(df_train_june,df_train_may, how='inner', on=['ncodpers'], suffixes=('', '_prev'))
prevcols = [col for col in dfm.columns if '_ult1_prev' in col]
currcols = [col for col in dfm.columns if '_ult1' in col and '_ult1_prev' not in col]
dfm.head()
def test(x):
    for col in currcols:
        if x[col+'_prev'] < x[col]:
            return True
        else:
            pass
    return False

dfm['test'] = dfm.apply(test, axis=1)
dfm.head()
dfm[dfm['test']==True].shape
dfm[dfm['test']==False].shape
dfm = dfm[dfm['test']==True]
dfm.drop(prevcols, axis=1, inplace=True)
prevcols2 = [col for col in dfm.columns if '_prev' in col]
dfm.drop(prevcols2, axis=1, inplace=True)

df_train1 = dfm
df_train1.head()
df_train1.drop('test', axis=1, inplace=True)
df_train1.head()
df_train1.to_csv('juneExtra.csv')
