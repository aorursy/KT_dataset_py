import pandas as pd

import numpy as np

import shutil

import datetime

from joblib import dump, load

import pyodbc

import seaborn as sns

from sklearn import tree

from sklearn.model_selection import cross_validate

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics

from sklearn.metrics import f1_score

from sklearn.metrics import roc_auc_score

from sklearn.metrics import make_scorer

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from scipy import stats

import lightgbm as lgb

import warnings

from sklearn.model_selection import train_test_split

import xgboost as xgb

from joblib import dump, load
def Columnas_vacias(df):

        mis_val = df.isnull().sum()

        mis_val_percent = 100 * df.isnull().sum() / len(df)

        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

        mis_val_table_ren_columns = mis_val_table.rename(

        columns = {0 : 'Missing Values', 1 : '% of Total Values'})

        mis_val_table_ren_columns = mis_val_table_ren_columns[

            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(

        '% of Total Values', ascending=False).round(1)

        print ("La tabla seleccionada con " + str(df.shape[1]) + " columnas.\n"      

            "Tiene " + str(mis_val_table_ren_columns.shape[0]) +

              " columnas con valores nulos.")

        return mis_val_percent.sort_values(ascending=False).head(mis_val_table_ren_columns.shape[0])
X_test = pd.read_csv("/kaggle/input/interbank-internacional-2019/ib_base_inicial_test/ib_base_inicial_test.csv")

sunat = pd.read_csv("/kaggle/input/interbank-internacional-2019/ib_base_sunat/ib_base_sunat.csv")

train = pd.read_csv("/kaggle/input/interbank-internacional-2019/ib_base_inicial_train/ib_base_inicial_train.csv")

rcc = pd.read_csv("/kaggle/input/interbank-internacional-2019/ib_base_rcc/ib_base_rcc.csv")

reniec = pd.read_csv("/kaggle/input/interbank-internacional-2019/ib_base_reniec/ib_base_reniec.csv")

digital = pd.read_csv("/kaggle/input/interbank-internacional-2019/ib_base_digital/ib_base_digital.csv")

vehicular = pd.read_csv("/kaggle/input/interbank-internacional-2019/ib_base_vehicular/ib_base_vehicular.csv")

campanias = pd.read_csv("/kaggle/input/interbank-internacional-2019/ib_base_campanias/ib_base_campanias.csv")
train=train[train['margen']>=-5]

train=train[train['margen']<=1000]
y_train = train[['codmes', 'id_persona', 'margen']].copy()

y_train["prediction_id"] = y_train["id_persona"].astype(str) + "_" + y_train["codmes"].astype(str)

# y_train["target"] = y_train["margen"].astype("float32")

y_train = y_train.set_index("prediction_id")

X_train = train.drop(["codtarget", "margen"], axis=1)

X_train["prediction_id"] = X_train["id_persona"].astype(str) + "_" + X_train["codmes"].astype(str)

del train
X_train["Rt_linea_ingreso"] = X_train["linea_ofrecida"] / X_train["ingreso_neto"]

X_test["Rt_linea_ingreso"] = X_test["linea_ofrecida"] / X_test["ingreso_neto"]



X_train["Rt_cem_ingreso"] = X_train["cem"] / X_train["ingreso_neto"]

X_test["Rt_cem_ingreso"] = X_test["cem"] / X_test["ingreso_neto"]



X_train["Rt_cem_linea"] = X_train["cem"] / X_train["linea_ofrecida"]

X_test["Rt_cem_linea"] = X_test["cem"] / X_test["linea_ofrecida"]



X_train["Df_linea_ingreso"] = X_train["linea_ofrecida"]- X_train["ingreso_neto"]

X_test["Df_linea_ingreso"] = X_test["linea_ofrecida"] - X_test["ingreso_neto"]



X_train["Ahorro_neto"] = X_train["linea_ofrecida"]- X_train["ingreso_neto"]-X_train["cem"]

X_test["Ahorro_neto"] = X_test["linea_ofrecida"] - X_test["ingreso_neto"]-X_train["cem"]
rcc.clasif.fillna(-9999, inplace=True)

rcc.rango_mora.fillna(-9999, inplace=True)
rcc['Rango_mora_final']=0

rcc.loc[rcc['rango_mora']==-9999,'Rango_mora_final']=-9999

rcc.loc[rcc['rango_mora']==1,'Rango_mora_final']=1

rcc=rcc.drop(['rango_mora'],axis=1)
rcc.producto.unique()
rcc['Producto_final']='Otros'

rcc.loc[rcc['producto']=='PRESTAMOS COMERCIALES','Producto_final']='PRESTAMOS COMERCIALES'

rcc.loc[rcc['producto']=='TARJETAS COMPRAS','Producto_final']='TARJETAS COMPRAS'

rcc.loc[rcc['producto']=='TARJETAS EFECTIVO','Producto_final']='TARJETAS EFECTIVO'

rcc.loc[rcc['producto']=='TARJETAS OTROS CONCEPTOS','Producto_final']='TARJETAS OTROS CONCEPTOS'

rcc.loc[rcc['producto']=='LINEA TOTAL TC','Producto_final']='LINEA TOTAL TC'

rcc.loc[rcc['producto']=='PRESTAMO PERSONAL','Producto_final']='PRESTAMO PERSONAL'

rcc=rcc.drop(['producto'],axis=1)
rcc['Cod_banco_final']='Otros'

rcc.loc[rcc['cod_banco']==40,'Cod_banco_final']='COD_40'

rcc.loc[rcc['cod_banco']==16,'Cod_banco_final']='COD_16'

rcc.loc[rcc['cod_banco']==66,'Cod_banco_final']='COD_66'

rcc.loc[rcc['cod_banco']==20,'Cod_banco_final']='COD_20'

rcc.loc[rcc['cod_banco']==28,'Cod_banco_final']='COD_28'

rcc.loc[rcc['cod_banco']==36,'Cod_banco_final']='COD_36'

rcc.loc[rcc['cod_banco']==7,'Cod_banco_final']='COD_7'

rcc.loc[rcc['cod_banco']==43,'Cod_banco_final']='COD_43'

rcc.loc[rcc['cod_banco']==52,'Cod_banco_final']='COD_52'

rcc.loc[rcc['cod_banco']==14,'Cod_banco_final']='COD_14'

rcc=rcc.drop(['cod_banco'],axis=1)
rcc_clasif = rcc.groupby(["codmes", "id_persona"]).clasif.max().reset_index().set_index("codmes").sort_index().astype("int32")

rcc_mora = rcc.groupby(["codmes", "id_persona", "Rango_mora_final"]).mto_saldo.sum().unstack(level=2, fill_value=0).reset_index().set_index("codmes").sort_index().astype("int32")

rcc_producto = rcc.groupby(["codmes", "id_persona", "Producto_final"]).mto_saldo.sum().unstack(level=2, fill_value=0).reset_index().set_index("codmes").sort_index().astype("int32")

rcc_banco = rcc.groupby(["codmes", "id_persona", "Cod_banco_final"]).mto_saldo.sum().unstack(level=2, fill_value=0).reset_index().set_index("codmes").sort_index().astype("int32")
del rcc
rcc_mora.columns = ["mora_" + str(c) if c != "id_persona" else c for c in rcc_mora.columns ]

rcc_producto.columns = ["producto_" + str(c) if c != "id_persona" else c for c in rcc_producto.columns]

rcc_banco.columns = ["banco_" + str(c) if c != "id_persona" else c for c in rcc_banco.columns]
campanias['producto']=campanias['producto'].str.upper()

campanias['canal_asignado']=campanias['canal_asignado'].str.upper()
Columnas_vacias(campanias)
campanias['canal_asignado']=campanias['canal_asignado'].fillna('RED DE TIENDAS')

campanias['producto']=campanias['producto'].fillna('ADQUISICIÓN TC')

Columnas_vacias(campanias)
campanias['FLG_TELEVENTAS']=0

campanias['FLG_RED_TIENDAS']=0

campanias['FLG_RED_OTROS']=0



campanias.loc[campanias['canal_asignado']=='TELEVENTAS','FLG_TELEVENTAS']=1

campanias.loc[campanias['canal_asignado']=='RED DE TIENDAS','FLG_RED_TIENDAS']=1

campanias.loc[campanias['canal_asignado'].isin(['TELEVENTAS','RED DE TIENDAS'])==False,'FLG_RED_OTROS']=1

campanias=campanias.drop(['canal_asignado'],axis=1)
campanias['PRODUCTO_NUEVO']='NINGUNO'

campanias.loc[campanias['producto'].isin(['SEGURO ACCIDENTES REMARK','SEGURO ASISTENCIA COMPLETA','SEGURO BLINDADO DE TC','SEGURO CONTRA ACCIDENTES','SEGURO DENTAL','SEGURO ONCOSALUD','SEGURO RENTA HOSPITALARIA','SEGURO SALUD','SEGURO VEHICULAR','SEGURO VIAJES','SEGURO VIDA RETORNO','SEGUROS'])==True,'PRODUCTO_NUEVO']='SEGUROS'   

campanias.loc[campanias['producto'].isin(['PRÉSTAMO EXPRESS','PRÉSTAMOS PERSONALES','PRÉSTAMOS REENGANCHE'])==True,'PRODUCTO_NUEVO']='PRESTAMOS'   

campanias.loc[campanias['producto'].isin(['RETENCION TC','RETENCIÓN'])==True,'PRODUCTO_NUEVO']='RETENCION'  

campanias.loc[campanias['producto'].isin(['DEPÓSITO A PLAZO JUBILACION','DÉPOSITO A PLAZO RENOVACION','DEPÓSITO A PLAZO RENOVACION','DEPÓSITO A PLAZO'])==True,'PRODUCTO_NUEVO']='DEPOSITO A PLAZO'  

campanias.loc[campanias['producto'].isin(['CUENTA MILLONARIA SUPERTASA','CUENTA MILLONARIA'])==True,'PRODUCTO_NUEVO']='CUENTA MILLONARIA'  

campanias.loc[campanias['producto'].isin(['CUENTA SUELDO INDEPENDIENTE','CUENTA SUELDO'])==True,'PRODUCTO_NUEVO']='CUENTA SUELDO'  

campanias.loc[campanias['producto'].isin(['CUENTA SIMPLE'])==True,'PRODUCTO_NUEVO']='CUENTA SIMPLE' 

campanias.loc[campanias['producto'].isin(['COMBOS CUENTA+APP','COMBOS TC + PA','COMBOS TC+CUENTA+APP','COMBOS TC+PA','COMBOS'])==True,'PRODUCTO_NUEVO']='COMBOS' 

campanias.loc[campanias['producto'].isin(['EXTRACASH ATAQUE','EXTRACASH'])==True,'PRODUCTO_NUEVO']='EXTRACASH' 

campanias.loc[campanias['producto'].isin(['MEMBRESIA','MEMBRESÍA'])==True,'PRODUCTO_NUEVO']='MEMBRESIA' 

campanias.loc[campanias['producto'].isin(['CRÉDITO HIPOTECARIO','CRÉDITO VEHICULAR', 'HIPOTECARIO'])==True,'PRODUCTO_NUEVO']='CREDITOS' 

campanias.loc[campanias['producto'].isin(['CONVENIOS AMPLIACIONES','CONVENIOS COMPRA DEUDA CLIENTE','CONVENIOS COMPRA DEUDA NO CLIENTE'])==True,'PRODUCTO_NUEVO']='CONVENIOS' 

campanias.loc[campanias['producto'].isin(['UPGRADE TC','UPGRADE'])==True,'PRODUCTO_NUEVO']='UPGRADE' 

campanias.loc[campanias['producto'].isin(['CD PRÉSTAMOS','CD-DEFENSA','CD-ATAQUE'])==True,'PRODUCTO_NUEVO']='COMPRA DEUDA' 

campanias.loc[campanias['producto'].isin(['ADQUISICIÓN CONVENIOS','ADQUISICIÓN TC'])==True,'PRODUCTO_NUEVO']='ADQUISICION' 

campanias.loc[campanias['producto'].isin([ 'ADELANTO DE SUELDO'])==True,'PRODUCTO_NUEVO']='ADELANTO' 

campanias.loc[campanias['producto'].isin(['ALCANCÍA'])==True,'PRODUCTO_NUEVO']='ALCANCÍA'

campanias.loc[campanias['producto'].isin(['CARTERA ABP'])==True,'PRODUCTO_NUEVO']='CARTERA ABP'

campanias.loc[campanias['producto'].isin(['CERTIFICADO BANCARIO'])==True,'PRODUCTO_NUEVO']= 'BANCARIO'

campanias.loc[campanias['producto'].isin(['CTS'])==True,'PRODUCTO_NUEVO']='CTS'

campanias.loc[campanias['producto'].isin([ 'INCREMENTO LINEA'])==True,'PRODUCTO_NUEVO']= 'LINEA'

campanias.loc[campanias['producto'].isin([ 'PAGO AUTOMATICO'])==True,'PRODUCTO_NUEVO']= 'AUTOMATICO'

campanias.loc[campanias['producto'].isin([ 'PLAZO'])==True,'PRODUCTO_NUEVO']='PLAZO'

campanias.loc[campanias['producto'].isin([ 'TELEVENTAS'])==True,'PRODUCTO_NUEVO']='TELEVENTAS'

campanias.loc[campanias['producto'].isin([  'TRADING'])==True,'PRODUCTO_NUEVO']='TRADING'

campanias=campanias.drop(['producto'],axis=1)
campanias['FLG_PRODUCTO_ADQUISION']=0

campanias['FLG_PRODUCTO_CTS']=0

campanias['FLG_PRODUCTO_COMPRA_DEUDA']=0

campanias['FLG_PRODUCTO_COMBOS']=0

campanias['FLG_PRODUCTO_CUENTA_SUELDO']=0

campanias['FLG_PRODUCTO_CREDITOS']=0

campanias['FLG_PRODUCTO_PRESTAMOS']=0

campanias['FLG_PRODUCTO_OTROS']=0



campanias.loc[campanias['PRODUCTO_NUEVO']=='ADQUISICION','FLG_PRODUCTO_ADQUISION']=1

campanias.loc[campanias['PRODUCTO_NUEVO']=='CTS','FLG_PRODUCTO_CTS'   ]=1

campanias.loc[campanias['PRODUCTO_NUEVO']=='COMPRA DEUDA','FLG_PRODUCTO_COMPRA_DEUDA'   ]=1

campanias.loc[campanias['PRODUCTO_NUEVO']=='COMBOS','FLG_PRODUCTO_COMBOS'   ]=1

campanias.loc[campanias['PRODUCTO_NUEVO']=='CUENTA SUELDO','FLG_PRODUCTO_CUENTA_SUELDO'   ]=1

campanias.loc[campanias['PRODUCTO_NUEVO']=='CREDITOS','FLG_PRODUCTO_CREDITOS'   ]=1

campanias.loc[campanias['PRODUCTO_NUEVO']=='PRESTAMOS','FLG_PRODUCTO_PRESTAMOS'   ]=1

campanias.loc[campanias['PRODUCTO_NUEVO'].isin(['ADQUISICION','CTS','COMPRA DEUDA','COMBOS','CUENTA SUELDO','CREDITOS','PRESTAMOS'])==False,'FLG_PRODUCTO_OTROS'   ]=1

campanias=campanias.drop(['PRODUCTO_NUEVO'],axis=1)
campanias=campanias.set_index("codmes").sort_index().astype("int32")

campanias.head()
digital.head()
digital["codmes"] = digital.codday.astype(str).str[:-2].astype(int)

digital = digital.drop("codday", axis=1).fillna(0)
digital['Redes']=digital['facebook']+digital['youtb']+digital['goog']+digital['email']

digital['Busquedas']=digital['busqvisa']+digital['busqamex']+digital['busqmc']+digital['busqcsimp']+digital['busqmill']++digital['busqcsld']+digital['busqtc']+digital['busq']

digital['Times']=digital['time_mllst']+digital['time_ctasld']+digital['time_tc']+digital['time_tc']+digital['time_ctasimple']+digital['time_mllp']

digital=digital.drop(['time_ctasimple', 'time_mllp',

       'time_mllst', 'time_ctasld', 'time_tc','busqtc',

       'busqvisa', 'busqamex', 'busqmc', 'busqcsimp', 'busqmill', 'busqcsld',

       'busq','email', 'facebook', 'goog',

       'youtb'],axis=1)

digital.head(5)
digital_sumas=digital.groupby(["codmes", "id_persona"]).sum().reset_index().set_index("codmes").sort_index().astype("int32")

digital_sumas.columns = ["SUMAS_" + str(c) if c != "id_persona" else c for c in digital_sumas.columns]

digital_sumas.head(5)
del digital
Columnas_vacias(sunat)
sunat['activ_econo_final']='Otros'

sunat.loc[sunat['activ_econo']=='Grupo_11','activ_econo_final']='Grupo_11'

sunat.loc[sunat['activ_econo']=='Grupo_15','activ_econo_final']='Grupo_15'

sunat.loc[sunat['activ_econo']=='Grupo_07','activ_econo_final']='Grupo_07'

sunat.loc[sunat['activ_econo']=='Grupo_13','activ_econo_final']='Grupo_13'

sunat.loc[sunat['activ_econo']=='Grupo_09','activ_econo_final']='Grupo_09'

sunat.loc[sunat['activ_econo']=='Grupo_08','activ_econo_final']='Grupo_08'

sunat.loc[sunat['activ_econo']=='Grupo_14','activ_econo_final']='Grupo_14'

sunat.loc[sunat['activ_econo']=='Grupo_12','activ_econo_final']='Grupo_12'

sunat.loc[sunat['activ_econo']=='Grupo_06','activ_econo_final']='Grupo_06'

sunat=sunat.drop(['activ_econo'],axis=1)
sunat.head()
sunat = sunat.groupby(["id_persona", "activ_econo_final"]).meses_alta.sum().unstack(level=1, fill_value=0).astype("int32")
vehicular['Marca_final']='Otros'

vehicular.loc[vehicular['marca']=='TOYOTA','Marca_final']='TOYOTA'

vehicular.loc[vehicular['marca']=='NISSAN','Marca_final']='NISSAN'

vehicular.loc[vehicular['marca']=='HYUNDAI','Marca_final']='HYUNDAI'

vehicular.loc[vehicular['marca']=='KIA','Marca_final']='KIA'

vehicular.loc[vehicular['marca']=='VOLKSWAGEN','Marca_final']='VOLKSWAGEN'

vehicular.loc[vehicular['marca']=='SUZUKI','Marca_final']='SUZUKI'

vehicular.loc[vehicular['marca']=='HONDA','Marca_final']='HONDA'

vehicular.loc[vehicular['marca']=='CHEVROLET','Marca_final']='CHEVROLET'

vehicular.loc[vehicular['marca']=='BAJAJ','Marca_final']='BAJAJ'

vehicular.loc[vehicular['marca']=='MITSUBISHI','Marca_final']='MITSUBISHI'

vehicular.loc[vehicular['marca']=='DAEWOO','Marca_final']='DAEWOO'

vehicular.loc[vehicular['marca']=='MAZDA','Marca_final']='MAZDA'

vehicular.loc[vehicular['marca']=='FORD','Marca_final']='FORD'

vehicular.loc[vehicular['marca']=='SUBARU','Marca_final']='SUBARU'

vehicular.loc[vehicular['marca']=='PEUGEOT','Marca_final']='PEUGEOT'

vehicular=vehicular.drop(['marca'],axis=1)
vehicular1 = vehicular.groupby(["id_persona", "Marca_final"]).veh_var1.sum().unstack(level=1, fill_value=0).astype("float32")

vehicular2 = vehicular.groupby(["id_persona", "Marca_final"]).veh_var2.sum().unstack(level=1, fill_value=0).astype("float32")

vehicular1.columns = [c + "_v1" for c in vehicular1.columns]

vehicular2.columns = [c + "_v2" for c in vehicular2.columns]

del vehicular
reniec = reniec.set_index("id_persona").astype("float32")

reniec['soc_var6']=reniec['soc_var6'].fillna(-99999)
Columnas_vacias(X_test)
X_test=X_test.fillna(-9999)

X_train=X_train.fillna(-9999)
X_train.head()
X_train = X_train.set_index("prediction_id").astype("int32").reset_index().set_index("id_persona").join(vehicular1).join(vehicular2).join(reniec).join(sunat)

X_test = X_test.set_index("prediction_id").astype("int32").reset_index().set_index("id_persona").join(vehicular1).join(vehicular2).join(reniec).join(sunat)

del vehicular1, vehicular2, reniec, sunat
import gc
meses = {

    201901: slice(201808, 201810),

    201902: slice(201809, 201811),

    201903: slice(201810, 201812),

    201904: slice(201811, 201901),

    201905: slice(201812, 201902),

    201906: slice(201901, 201903),

    201907: slice(201902, 201904)

        }



meses_train = X_train.codmes.unique()

meses_test = X_test.codmes.unique()

complementos = []

for mes in meses.keys():

    print("*"*10, mes, "*"*10)

    res = pd.concat([

        rcc_clasif.loc[meses[mes]].groupby("id_persona").sum(),

        rcc_mora.loc[meses[mes]].groupby("id_persona").sum(),

        rcc_producto.loc[meses[mes]].groupby("id_persona").sum(),

        rcc_banco.loc[meses[mes]].groupby("id_persona").sum(),

        campanias.loc[meses[mes]].groupby("id_persona").sum(),

        digital_sumas.loc[meses[mes]].groupby("id_persona").sum()

        

    ], axis=1)

    res["codmes"] = mes

    res = res.reset_index().set_index(["id_persona", "codmes"]).astype("float32")

    complementos.append(res)



gc.collect()

print("concatenando complementos")

complementos = pd.concat(complementos)

gc.collect()

print("X_train join")

X_train = X_train.reset_index().join(complementos, on=["id_persona", "codmes"]).set_index("prediction_id")

gc.collect()

print("X_test join")

X_test = X_test.reset_index().join(complementos, on=["id_persona", "codmes"]).set_index("prediction_id")

gc.collect()



del rcc_clasif, rcc_mora, rcc_producto, rcc_banco,campanias, digital_sumas, complementos,res

gc.collect()
for i, c in enumerate(X_train.columns[[not all(ord(c) < 128 for c in s) for s in X_train.columns]]):

    X_train["non_ascii_" + str(i)] = X_train[c]

    X_train = X_train.drop(c, axis= 1)

    X_test["non_ascii_" + str(i)] = X_test[c]

    X_test = X_test.drop(c, axis= 1)
from lightgbm import LGBMRegressor

gc.collect()
drop_cols = ["codmes"]

test_preds = []

train_preds = []

y_train["target"] = y_train["margen"].astype("float32")

for mes in X_train.codmes.unique():

    print("*"*10, mes, "*"*10)

    Xt = X_train[X_train.codmes != mes]

    yt = y_train.loc[Xt.index, "target"]

    Xt = Xt.drop(drop_cols, axis=1)



    Xv = X_train[X_train.codmes == mes]

    yv = y_train.loc[Xv.index, "target"]

    

    learner = LGBMRegressor(n_estimators=80)

    learner.fit(Xt, yt,  early_stopping_rounds=100, eval_metric="mae",

                eval_set=[(Xt, yt), (Xv.drop(drop_cols, axis=1), yv)], verbose=50)

    gc.collect()

    test_preds.append(pd.Series(learner.predict(X_test.drop(drop_cols, axis=1)),

                                index=X_test.index, name="fold_" + str(mes)))

    train_preds.append(pd.Series(learner.predict(Xv.drop(drop_cols, axis=1)),

                                index=Xv.index, name="probs"))

    gc.collect()



test_preds = pd.concat(test_preds, axis=1).mean(axis=1)

train_preds = pd.concat(train_preds)
from lightgbm import LGBMClassifier

gc.collect()
drop_cols = ["codmes"]

fi = []

test_probs = []

train_probs = []

y_train["target"] = (y_train["margen"] > 0).astype("int32")

for mes in X_train.codmes.unique():

    print("*"*10, mes, "*"*10)

    Xt = X_train[X_train.codmes != mes]

    yt = y_train.loc[Xt.index, "target"]

    Xt = Xt.drop(drop_cols, axis=1)



    Xv = X_train[X_train.codmes == mes]

    yv = y_train.loc[Xv.index, "target"]

    

    learner = LGBMClassifier(n_estimators=80)

    learner.fit(Xt, yt,  early_stopping_rounds=100, eval_metric="mae",

                eval_set=[(Xt, yt), (Xv.drop(drop_cols, axis=1), yv)], verbose=50)

    gc.collect()

    test_probs.append(pd.Series(learner.predict_proba(X_test.drop(drop_cols, axis=1))[:, -1],

                                index=X_test.index, name="fold_" + str(mes)))

    train_probs.append(pd.Series(learner.predict_proba(Xv.drop(drop_cols, axis=1))[:, -1],

                                index=Xv.index, name="probs"))

    gc.collect()



test_probs = pd.concat(test_probs, axis=1).mean(axis=1)

train_probs = pd.concat(train_probs)
test = pd.concat([test_probs.rename("probs"), test_preds.rename("preds")], axis=1)

train = pd.concat([train_probs.rename("probs"), train_preds.rename("preds")], axis=1)
from scipy.optimize import differential_evolution



def clasificar(res, c):

    return ((res.probs > c[0]) | (res.preds > c[1])) * c[2] + ((res.probs > c[3]) & (res.preds > c[4])) * c[5] > c[6]



def cost(res, coefs):

    return -((clasificar(res, coefs) * res.margen) / res.margen.sum()).sum()



res = y_train.join(train)

optimization = differential_evolution(lambda x: cost(res, x), [(-100, 100), (0, 1), (0, 1),

                                                               (-100, 100), (0, 1), (0, 1),

                                                               (0, 2)])

optimization
test_preds = clasificar(test, optimization["x"]).astype(int)

test_preds.index.name="prediction_id"

test_preds.name="class"

test_preds.to_csv("benchmark3.csv", header=True)