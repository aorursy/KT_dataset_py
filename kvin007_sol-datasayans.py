dataDir = '/kaggle/input/datathon-belcorp-prueba/'
import numpy as np

import math

import pandas as pd
df_prod = pd.read_csv(dataDir + 'maestro_producto.csv')
df_consultora = pd.read_csv(dataDir +  'maestro_consultora.csv')
df_camp_consultora = pd.read_csv(dataDir +  'campana_consultora.csv')
df_ventas = pd.read_csv(dataDir + 'dtt_fvta_cl.csv')
df_consultora.drop('Unnamed: 0',axis=1,inplace=True)

df_camp_consultora.drop('Unnamed: 0',axis=1,inplace=True)

df_prod.drop('Unnamed: 0',axis=1,inplace=True)
df_historico_camp = df_camp_consultora[['campana','IdConsultora','Flagpasopedido']]
pivot_historico = pd.pivot_table(df_historico_camp, values='Flagpasopedido', index='campana',columns=['IdConsultora'])
dictConsultCamp = pivot_historico.to_dict()
def getTarget(idConsultora,campana):



  if (campana == 201906):

    return np.nan

  

  year = str(campana)[:-2]

  cap = str(campana)[-2:]



  if (cap=='18'):

    nextCampaign = int(str(int(year)+1) + '01')

  else:

    nextCampaign = campana +1



  ## If the following campaign exists

  if not math.isnan(dictConsultCamp[idConsultora][nextCampaign]):

    return dictConsultCamp[idConsultora][nextCampaign]

  else:

    return np.nan
df_camp_consultora['target'] = df_camp_consultora['IdConsultora'].astype(object).combine(df_camp_consultora['campana'],func=getTarget)
## Mergeo la info de la consultora

df_info_consult = df_camp_consultora.merge(df_consultora,how='left',on='IdConsultora')
## Mergeo la info relacionado a ventas

df_info_vt = df_ventas.merge(df_prod,how='left',on='idproducto')
df_info_vt['campanaConsultora'] = df_info_vt['campana'].astype(str) + '_' + df_info_vt['idconsultora'].astype(str)
df_info_vt_summary = df_info_vt.groupby('campanaConsultora').agg({

                                             'codigotipooferta': lambda x: x.mode().iat[0],

                                             'descuento': 'sum',

                                             'ahorro': 'sum',

                                             'preciocatalogo': 'sum',

                                             'realanulmnneto': 'sum',

                                             'realdevmnneto': 'sum',

                                             'realuuanuladas': 'sum',

                                             'realuudevueltas': 'sum',

                                             'realuufaltantes': 'sum',

                                             'realuuvendidas': 'sum',

                                             'realvtamnfaltneto': 'sum',

                                             'realvtamnneto': 'sum',

                                             'realvtamncatalogo': 'sum',

                                             'realvtamnfaltcatalogo': 'sum'

                                             })
df_info_vt_summary.reset_index(inplace=True)
df_info_consult['campanaConsultora']= df_info_consult['campana'].astype(str)  + '_' + df_info_consult['IdConsultora'].astype(str)
df_info_vt_summary.head(1)
## Mergeo todo

df_total = df_info_consult.merge(df_info_vt_summary,on='campanaConsultora',how='left')
# missingData = df_total.isnull().sum()

# percentageMissing = missingData / df_total.shape[0]
# percentageMissing.to_frame().reset_index().to_csv('percentageMissing.csv')
# percentageMissing = pd.read_csv('percentageMissing.csv')
# percentageMissing.drop(columns='Unnamed: 0',axis=1,inplace=True)

# percentageMissing.rename(columns={'0':'porc'},inplace=True)
## To drop (high missing values)

droppable = ['codigocanalorigen','flagcorreovalidad','codigofactura']
# percentageMissing.sort_values(by='porc',ascending=False)[:25]
df_total[df_total['campana']==201906]['IdConsultora'].unique().shape
df_total.drop(labels=droppable,axis=1,inplace=True)
# df_total = df_total[~df_total['IdConsultora'].isnull()]
cat_vars = [c for c in df_total if not pd.api.types.is_numeric_dtype(df_total[c])]
cat_vars
# 1. Convertir las columnas a tipo "category", ignorar la variable dependiente

# cat_vars = [c for c in df_total if not pd.api.types.is_numeric_dtype(df_total[c])]



cat_vars = ['evaluacion_nuevas',

 'segmentacion',

 'geografia',

 'estadocivil']



for n,col in df_total.items():

    if n in cat_vars:

      df_total[n] = df_total[n].astype('category')



# df_total.dtypes
for n,col in df_total.items():

    if pd.api.types.is_categorical_dtype(col):

        df_total[n] = col.cat.codes+1



df_total.head(3)
imputation_columns = [

                      'codigotipooferta',

                      'campanaultimopedido',

                      'flagsupervisor',

                      'campanaingreso',

'preciocatalogo',

'realvtamnfaltcatalogo',

'flagdigital',

'ahorro',

'realdevmnneto',

'cantidadlogueos',

'descuento',

'realanulmnneto',

'estadocivil',

'realuuanuladas',

'campanaprimerpedido',

'realuudevueltas',

'flagcelularvalidado',

'realuufaltantes',

'edad',

'realuuvendidas',

'flagconsultoradigital',

'realvtamnfaltneto',

'realvtamnneto',

'realvtamncatalogo']

for n,col in df_total.items():

    if n in imputation_columns:

      df_total[n].fillna((df_total[n].median()), inplace=True)
percentageMissing = df_total.isnull().mean()
# percentageMissing.sort_values(ascending=False)
df_predict = pd.read_csv(dataDir + 'predict_submission.csv')
df_total[df_total['target'].isnull()]['IdConsultora'].value_counts().sort_values(ascending=False)
df_predict['idconsultora'].unique().shape
df_total['campanaprimerpedido']
df_total.head(1)
## Hacef funcion para restar mejor

df_total['campanaprimerpedido'] = df_total['campana'] - df_total['campanaprimerpedido']

df_total['campanaingreso'] = df_total['campana'] - df_total['campanaingreso']

df_total['campanaultimopedido'] = df_total['campana'] - df_total['campanaultimopedido']

df_to_predict = df_total[df_total['target'].isnull()]
df_to_predict.shape
df_to_predict = df_to_predict.sort_values('campana', ascending=False).drop_duplicates('IdConsultora').sort_index()
df_predict.rename(columns={'idconsultora':'IdConsultora'},inplace=True)
df_entry_model = df_predict.merge(df_to_predict,how='left',on='IdConsultora')
df_entry_model
df = df_total[~df_total['target'].isnull()]
dropToTrain = ['campana','IdConsultora','campanaConsultora','Flagpasopedido']
df.drop(columns=dropToTrain,inplace=True)
Y = df['target']

X = df.drop('target',axis=1)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
import lightgbm as lgb

train_data = lgb.Dataset(X_train, label=y_train)

test_data = lgb.Dataset(X_test, label=y_test)
parameters = {

    'application': 'binary',

    'objective': 'binary',

    'metric': 'auc',

    'is_unbalance': 'true',

    'boosting': 'gbdt',

    'num_leaves': 31,

    'feature_fraction': 0.5,

    'bagging_fraction': 0.5,

    'bagging_freq': 20,

    'learning_rate': 0.05,

    'verbose': 0

}



model = lgb.train(parameters,

                       train_data,

                       valid_sets=test_data,

                       num_boost_round=5000,

                       early_stopping_rounds=100)
X = df.drop('target',axis=1)
ids = df_entry_model['IdConsultora'].values
dropToTrain = ['campana','campanaConsultora','Flagpasopedido']
df_entry_model.drop(columns=dropToTrain,inplace=True)
df_entry_model.drop('IdConsultora', inplace=True, axis=1)
df_entry_model.drop('flagpasopedido',inplace=True,axis=1)
df_entry_model.drop('target',inplace=True,axis=1)
x = df_entry_model
df_entry_model.columns
X.columns
y = model.predict(x)
y
output = pd.DataFrame({'idconsultora': ids, 'flagpasopedido': y})

output.to_csv("submission.csv", index=False)