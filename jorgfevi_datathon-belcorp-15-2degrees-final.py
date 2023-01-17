#En el caso de usar google colab

#from google.colab import drive

#drive.mount('/content/drive')
#cd "/content/drive/My Drive/Competencias/Datathon/Belcorp/Fase_Online"
ls
#!unzip -q "/content/drive/My Drive/Competencias/Datathon/Belcorp/Fase_Online/datathon-belcorp-prueba.zip"
# Basic

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



# Common Tools

from sklearn.preprocessing import LabelEncoder

from collections import Counter



#Algorithms

from sklearn import ensemble, tree, svm, naive_bayes, neighbors, linear_model, gaussian_process, neural_network

import xgboost as xgb

from xgboost.sklearn import XGBClassifier

from sklearn import metrics



# Model

from sklearn.metrics import accuracy_score, f1_score, auc, roc_curve, roc_auc_score

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score

#from sklearn.ensemble import VotingClassifier



#Configure Defaults

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
def Columnas_con_missing(df):

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
predecir = pd.read_csv("../input/datathon-belcorp-prueba/predict_submission.csv")

predecir['campana']=201907

predecir = predecir[['campana','idconsultora']]
predecir.head()
predecir.shape
maestro_consultora = pd.read_csv("../input/datathon-belcorp-prueba/maestro_consultora.csv")

maestro_consultora = maestro_consultora.drop('Unnamed: 0',axis=1)

maestro_consultora = maestro_consultora.rename({'IdConsultora': 'idconsultora'}, axis=1)

maestro_consultora.head()
maestro_consultora.shape
campana_consultora=pd.read_csv("../input/datathon-belcorp-prueba/campana_consultora.csv")
campana_consultora = campana_consultora.drop('Unnamed: 0',axis=1)

#campana_consultora = campana_consultora.sort_values(['campana', 'IdConsultora'], ascending=[True, True]).reset_index(drop=True)

#campana_consultora["llave"] = campana_consultora[['campana', 'IdConsultora']].apply(lambda x : '{}_{}'.format(x[0],x[1]), axis=1)

campana_consultora = campana_consultora.rename({'IdConsultora': 'idconsultora','Flagpasopedido': 'flagpasopedido'}, axis=1)

campana_consultora.head(3)
campana_consultora.shape
campana_consultora.groupby(['campana']).size()
#campana_consultora.dtypes #train.info()
Columnas_con_missing(campana_consultora)
campana_consultora = campana_consultora.drop("codigocanalorigen",axis=1)
#campana_consultora['FLG_codigofactura']=1

#campana_consultora.loc[pd.isna(campana_consultora['codigofactura'])==True, 'FLG_codigofactura']=0



#campana_consultora['FLG_cantidadlogueos']=1

#campana_consultora.loc[pd.isna(campana_consultora['cantidadlogueos'])==True, 'FLG_cantidadlogueos']=0
campana_consultora.campana.min(),campana_consultora.campana.max()
pd.value_counts(campana_consultora.flagpasopedido)/campana_consultora.shape[0]
#pd.value_counts(campana_consultora.idconsultora).describe()
#Columnas_con_missing(campana_consultora)
historico_consultora = campana_consultora.pivot(index="idconsultora",columns="campana",values="flagpasopedido")

historico_consultora = pd.DataFrame(historico_consultora.to_records())
historico_consultora=historico_consultora.fillna(0)

historico_consultora.head(3)
historico_consultora.shape
historico_predecir = pd.merge(predecir,historico_consultora,on='idconsultora',how='left')

historico_predecir.shape
historico_predecir.head(3)
train = historico_consultora[["idconsultora","201906"]]

train["campana"]=201906

train.columns=["idconsultora","flagpasopedido","campana"]

train = train[["campana","idconsultora","flagpasopedido"]]
train.head(3)
train.shape
test = predecir
for i in range(15):

  n=historico_consultora.shape[1]

  auxi = pd.DataFrame(historico_consultora.iloc[:,(n-i-2):(n-1)])

  train["pasopedido"+ str(i+1)+"UM"] = auxi.sum(axis=1)
for i in range(15):

  n=historico_predecir.shape[1]

  auxi = pd.DataFrame(historico_predecir.iloc[:,(n-i-1):(n)])

  test["pasopedido"+ str(i+1)+"UM"] = auxi.sum(axis=1)
historico_consultora = campana_consultora.pivot(index="idconsultora",columns="campana",values="flagactiva")

historico_consultora = pd.DataFrame(historico_consultora.to_records())
historico_consultora=historico_consultora.fillna(0)

historico_consultora.head(5)
predecir = pd.read_csv("../input/datathon-belcorp-prueba/predict_submission.csv")

predecir['campana']=201907

predecir = predecir[['campana','idconsultora']]
historico_predecir = pd.merge(predecir,historico_consultora,on='idconsultora',how='left')

historico_predecir.shape
historico_predecir.head(3)
for i in range(15):

  n=historico_consultora.shape[1]

  auxi = pd.DataFrame(historico_consultora.iloc[:,(n-i-1):(n)])

  train["activa"+ str(i+1)+"UM"] = auxi.sum(axis=1)
for i in range(15):

  n=historico_predecir.shape[1]

  auxi = pd.DataFrame(historico_predecir.iloc[:,(n-i-1):(n)])

  test["activa"+ str(i+1)+"UM"] = auxi.sum(axis=1)
historico_consultora = campana_consultora.pivot(index="idconsultora",columns="campana",values="flagpasopedidocuidadopersonal")

historico_consultora = pd.DataFrame(historico_consultora.to_records())
historico_consultora=historico_consultora.fillna(0)

historico_consultora.head(3)
predecir = pd.read_csv("../input/datathon-belcorp-prueba/predict_submission.csv")

predecir['campana']=201907

predecir = predecir[['campana','idconsultora']]
historico_predecir = pd.merge(predecir,historico_consultora,on='idconsultora',how='left')

historico_predecir.shape
historico_predecir.head(3)
for i in range(15):

  n=historico_consultora.shape[1]

  auxi = pd.DataFrame(historico_consultora.iloc[:,(n-i-2):(n-1)])

  train["cuidadopersonal"+ str(i+1)+"UM"] = auxi.sum(axis=1)
for i in range(15):

  n=historico_predecir.shape[1]

  auxi = pd.DataFrame(historico_predecir.iloc[:,(n-i-1):(n)])

  test["cuidadopersonal"+ str(i+1)+"UM"] = auxi.sum(axis=1)
historico_consultora = campana_consultora.pivot(index="idconsultora",columns="campana",values="flagpasopedidomaquillaje")

historico_consultora = pd.DataFrame(historico_consultora.to_records())
historico_consultora=historico_consultora.fillna(0)

historico_consultora.head(3)
historico_predecir = pd.merge(predecir,historico_consultora,on='idconsultora',how='left')

historico_predecir.shape
historico_predecir.head(3)
for i in range(15):

  n=historico_consultora.shape[1]

  auxi = pd.DataFrame(historico_consultora.iloc[:,(n-i-2):(n-1)])

  train["maquillaje"+ str(i+1)+"UM"] = auxi.sum(axis=1)
for i in range(15):

  n=historico_predecir.shape[1]

  auxi = pd.DataFrame(historico_predecir.iloc[:,(n-i-1):(n)])

  test["maquillaje"+ str(i+1)+"UM"] = auxi.sum(axis=1)
historico_consultora = campana_consultora.pivot(index="idconsultora",columns="campana",values="flagpasopedidotratamientocorporal")

historico_consultora = pd.DataFrame(historico_consultora.to_records())
historico_consultora=historico_consultora.fillna(0)

historico_consultora.head(3)
historico_predecir = pd.merge(predecir,historico_consultora,on='idconsultora',how='left')

historico_predecir.shape
historico_predecir.head(3)
for i in range(15):

  n=historico_consultora.shape[1]

  auxi = pd.DataFrame(historico_consultora.iloc[:,(n-i-2):(n-1)])

  train["tratamientocorporal"+ str(i+1)+"UM"] = auxi.sum(axis=1)



for i in range(15):

  n=historico_predecir.shape[1]

  auxi = pd.DataFrame(historico_predecir.iloc[:,(n-i-1):(n)])

  test["tratamientocorporal"+ str(i+1)+"UM"] = auxi.sum(axis=1)

historico_consultora = campana_consultora.pivot(index="idconsultora",columns="campana",values="flagpasopedidotratamientofacial")

historico_consultora = pd.DataFrame(historico_consultora.to_records())

historico_consultora = historico_consultora.fillna(0)
historico_predecir = pd.merge(predecir,historico_consultora,on='idconsultora',how='left')

historico_predecir.shape
for i in range(15):

  n=historico_consultora.shape[1]

  auxi = pd.DataFrame(historico_consultora.iloc[:,(n-i-2):(n-1)])

  train["tratamientofacial"+ str(i+1)+"UM"] = auxi.sum(axis=1)



for i in range(15):

  n=historico_predecir.shape[1]

  auxi = pd.DataFrame(historico_predecir.iloc[:,(n-i-1):(n)])

  test["tratamientofacial"+ str(i+1)+"UM"] = auxi.sum(axis=1)
historico_consultora = campana_consultora.pivot(index="idconsultora",columns="campana",values="flagpedidoanulado")

historico_consultora = pd.DataFrame(historico_consultora.to_records())

historico_consultora = historico_consultora.fillna(0)

historico_consultora.head(3)
historico_predecir = pd.merge(predecir,historico_consultora,on='idconsultora',how='left')

historico_predecir.head(3)
train["anuladoUM"] = historico_consultora["201905"]

train["anulado3UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])

train["anulado6UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])+pd.to_numeric(historico_consultora["201902"])+pd.to_numeric(historico_consultora["201901"])+pd.to_numeric(historico_consultora["201818"])

train["anulado9UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])+pd.to_numeric(historico_consultora["201902"])+pd.to_numeric(historico_consultora["201901"])+pd.to_numeric(historico_consultora["201818"])+pd.to_numeric(historico_consultora["201817"])+pd.to_numeric(historico_consultora["201816"])+pd.to_numeric(historico_consultora["201815"])

train["anulado12UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])+pd.to_numeric(historico_consultora["201902"])+pd.to_numeric(historico_consultora["201901"])+pd.to_numeric(historico_consultora["201818"])+pd.to_numeric(historico_consultora["201817"])+pd.to_numeric(historico_consultora["201816"])+pd.to_numeric(historico_consultora["201815"])+pd.to_numeric(historico_consultora["201814"])+pd.to_numeric(historico_consultora["201813"])+pd.to_numeric(historico_consultora["201812"])



test["anuladoUM"] = historico_predecir["201906"]

test["anulado3UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])

test["anulado6UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])+pd.to_numeric(historico_predecir["201903"])+pd.to_numeric(historico_predecir["201902"])+pd.to_numeric(historico_predecir["201901"])

test["anulado9UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])+pd.to_numeric(historico_predecir["201903"])+pd.to_numeric(historico_predecir["201902"])+pd.to_numeric(historico_predecir["201901"])+pd.to_numeric(historico_predecir["201818"])+pd.to_numeric(historico_predecir["201817"])+pd.to_numeric(historico_predecir["201816"])

test["anulado12UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])+pd.to_numeric(historico_predecir["201903"])+pd.to_numeric(historico_predecir["201902"])+pd.to_numeric(historico_predecir["201901"])+pd.to_numeric(historico_predecir["201818"])+pd.to_numeric(historico_predecir["201817"])+pd.to_numeric(historico_predecir["201816"])+pd.to_numeric(historico_predecir["201815"])+pd.to_numeric(historico_predecir["201814"])+pd.to_numeric(historico_predecir["201813"])
historico_consultora = campana_consultora.pivot(index="idconsultora",columns="campana",values="flagpasopedidofragancias")

historico_consultora = pd.DataFrame(historico_consultora.to_records())

historico_consultora = historico_consultora.fillna(0)
historico_predecir = pd.merge(predecir,historico_consultora,on='idconsultora',how='left')
historico_consultora.shape[1],historico_predecir.shape[1]
for i in range(15):

  n=historico_consultora.shape[1]

  auxi = pd.DataFrame(historico_consultora.iloc[:,(n-i-2):(n-1)])

  train["fragancias"+ str(i+1)+"UM"] = auxi.sum(axis=1)



for i in range(15):

  n=historico_predecir.shape[1]

  auxi = pd.DataFrame(historico_predecir.iloc[:,(n-i-1):(n)])

  test["fragancias"+ str(i+1)+"UM"] = auxi.sum(axis=1)
historico_consultora = campana_consultora.pivot(index="idconsultora",columns="campana",values="flagpasopedidoweb")

historico_consultora = pd.DataFrame(historico_consultora.to_records())

historico_consultora = historico_consultora.fillna(0)



historico_predecir = pd.merge(predecir,historico_consultora,on='idconsultora',how='left')
historico_consultora.shape, historico_predecir.shape
for i in range(15):

  n=historico_consultora.shape[1]

  auxi = pd.DataFrame(historico_consultora.iloc[:,(n-i-2):(n-1)])

  train["pedidoweb"+ str(i+1)+"UM"] = auxi.sum(axis=1)



for i in range(15):

  n=historico_predecir.shape[1]

  auxi = pd.DataFrame(historico_predecir.iloc[:,(n-i-1):(n)])

  test["pedidoweb"+ str(i+1)+"UM"] = auxi.sum(axis=1)
historico_consultora = campana_consultora.pivot(index="idconsultora",columns="campana",values="cantidadlogueos")

historico_consultora = pd.DataFrame(historico_consultora.to_records())

historico_consultora = historico_consultora.fillna(0)



historico_predecir = pd.merge(predecir,historico_consultora,on='idconsultora',how='left')
historico_consultora.shape, historico_predecir.shape
for i in range(15):

  n=historico_consultora.shape[1]

  auxi = pd.DataFrame(historico_consultora.iloc[:,(n-i-2):(n-1)])

  train["logueos"+ str(i+1)+"UM"] = auxi.sum(axis=1)



for i in range(15):

  n=historico_predecir.shape[1]

  auxi = pd.DataFrame(historico_predecir.iloc[:,(n-i-1):(n)])

  test["logueos"+ str(i+1)+"UM"] = auxi.sum(axis=1)
historico_consultora = campana_consultora.pivot(index="idconsultora",columns="campana",values="flagdispositivo")

historico_consultora = pd.DataFrame(historico_consultora.to_records())

historico_consultora = historico_consultora.fillna(0)



historico_predecir = pd.merge(predecir,historico_consultora,on='idconsultora',how='left')
train["flgdispositivoUM"] = historico_consultora["201905"]

train["flgdispositivo2UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])

train["flgdispositivo3UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])

train["flgdispositivo6UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])+pd.to_numeric(historico_consultora["201902"])+pd.to_numeric(historico_consultora["201901"])+pd.to_numeric(historico_consultora["201818"])

train["flgdispositivo9UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])+pd.to_numeric(historico_consultora["201902"])+pd.to_numeric(historico_consultora["201901"])+pd.to_numeric(historico_consultora["201818"])+pd.to_numeric(historico_consultora["201817"])+pd.to_numeric(historico_consultora["201816"])+pd.to_numeric(historico_consultora["201815"])

train["flgdispositivo12UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])+pd.to_numeric(historico_consultora["201902"])+pd.to_numeric(historico_consultora["201901"])+pd.to_numeric(historico_consultora["201818"])+pd.to_numeric(historico_consultora["201817"])+pd.to_numeric(historico_consultora["201816"])+pd.to_numeric(historico_consultora["201815"])+pd.to_numeric(historico_consultora["201814"])+pd.to_numeric(historico_consultora["201813"])+pd.to_numeric(historico_consultora["201812"])



test["flgdispositivoUM"] = historico_predecir["201906"]

test["flgdispositivo2UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])

test["flgdispositivo3UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])

test["flgdispositivo6UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])+pd.to_numeric(historico_predecir["201903"])+pd.to_numeric(historico_predecir["201902"])+pd.to_numeric(historico_predecir["201901"])

test["flgdispositivo9UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])+pd.to_numeric(historico_predecir["201903"])+pd.to_numeric(historico_predecir["201902"])+pd.to_numeric(historico_predecir["201901"])+pd.to_numeric(historico_predecir["201818"])+pd.to_numeric(historico_predecir["201817"])+pd.to_numeric(historico_predecir["201816"])

test["flgdispositivo12UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])+pd.to_numeric(historico_predecir["201903"])+pd.to_numeric(historico_predecir["201902"])+pd.to_numeric(historico_predecir["201901"])+pd.to_numeric(historico_predecir["201818"])+pd.to_numeric(historico_predecir["201817"])+pd.to_numeric(historico_predecir["201816"])+pd.to_numeric(historico_predecir["201815"])+pd.to_numeric(historico_predecir["201814"])+pd.to_numeric(historico_predecir["201813"])
historico_consultora = campana_consultora.pivot(index="idconsultora",columns="campana",values="flagdigital")

historico_consultora = pd.DataFrame(historico_consultora.to_records())

historico_consultora = historico_consultora.fillna(0)



historico_predecir = pd.merge(predecir,historico_consultora,on='idconsultora',how='left')
train["flgdigitalUM"] = historico_consultora["201905"]

train["flgdigital2UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])

train["flgdigital3UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])

train["flgdigital6UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])+pd.to_numeric(historico_consultora["201902"])+pd.to_numeric(historico_consultora["201901"])+pd.to_numeric(historico_consultora["201818"])

train["flgdigital9UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])+pd.to_numeric(historico_consultora["201902"])+pd.to_numeric(historico_consultora["201901"])+pd.to_numeric(historico_consultora["201818"])+pd.to_numeric(historico_consultora["201817"])+pd.to_numeric(historico_consultora["201816"])+pd.to_numeric(historico_consultora["201815"])

train["flgdigital12UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])+pd.to_numeric(historico_consultora["201902"])+pd.to_numeric(historico_consultora["201901"])+pd.to_numeric(historico_consultora["201818"])+pd.to_numeric(historico_consultora["201817"])+pd.to_numeric(historico_consultora["201816"])+pd.to_numeric(historico_consultora["201815"])+pd.to_numeric(historico_consultora["201814"])+pd.to_numeric(historico_consultora["201813"])+pd.to_numeric(historico_consultora["201812"])



test["flgdigitalUM"] = historico_predecir["201906"]

test["flgdigital2UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])

test["flgdigital3UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])

test["flgdigital6UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])+pd.to_numeric(historico_predecir["201903"])+pd.to_numeric(historico_predecir["201902"])+pd.to_numeric(historico_predecir["201901"])

test["flgdigital9UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])+pd.to_numeric(historico_predecir["201903"])+pd.to_numeric(historico_predecir["201902"])+pd.to_numeric(historico_predecir["201901"])+pd.to_numeric(historico_predecir["201818"])+pd.to_numeric(historico_predecir["201817"])+pd.to_numeric(historico_predecir["201816"])

test["flgdigital12UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])+pd.to_numeric(historico_predecir["201903"])+pd.to_numeric(historico_predecir["201902"])+pd.to_numeric(historico_predecir["201901"])+pd.to_numeric(historico_predecir["201818"])+pd.to_numeric(historico_predecir["201817"])+pd.to_numeric(historico_predecir["201816"])+pd.to_numeric(historico_predecir["201815"])+pd.to_numeric(historico_predecir["201814"])+pd.to_numeric(historico_predecir["201813"])
historico_consultora = campana_consultora.pivot(index="idconsultora",columns="campana",values="flagofertadigital")

historico_consultora = pd.DataFrame(historico_consultora.to_records())

historico_consultora = historico_consultora.fillna(0)



historico_predecir = pd.merge(predecir,historico_consultora,on='idconsultora',how='left')
historico_consultora.shape[1],historico_predecir.shape[1]
for i in range(15):

  n=historico_consultora.shape[1]

  auxi = pd.DataFrame(historico_consultora.iloc[:,(n-i-2):(n-1)])

  train["flgofertadigital"+ str(i+1)+"UM"] = auxi.sum(axis=1)



for i in range(15):

  n=historico_predecir.shape[1]

  auxi = pd.DataFrame(historico_predecir.iloc[:,(n-i-1):(n)])

  test["flgofertadigital"+ str(i+1)+"UM"] = auxi.sum(axis=1)
historico_consultora = campana_consultora.pivot(index="idconsultora",columns="campana",values="flagsuscripcion")

historico_consultora = pd.DataFrame(historico_consultora.to_records())

historico_consultora = historico_consultora.fillna(0)



historico_predecir = pd.merge(predecir,historico_consultora,on='idconsultora',how='left')
historico_consultora.shape[1],historico_predecir.shape[1]
for i in range(15):

  n=historico_consultora.shape[1]

  auxi = pd.DataFrame(historico_consultora.iloc[:,(n-i-2):(n-1)])

  train["suscripcion"+ str(i+1)+"UM"] = auxi.sum(axis=1)



for i in range(15):

  n=historico_predecir.shape[1]

  auxi = pd.DataFrame(historico_predecir.iloc[:,(n-i-1):(n)])

  test["suscripcion"+ str(i+1)+"UM"] = auxi.sum(axis=1)
train.shape,test.shape
sns.catplot(y="codigofactura", kind="count",palette="pastel", edgecolor=".6",data=campana_consultora,)
campana_consultora["codigofactura_web"]=0

campana_consultora["codigofactura_web"].loc[campana_consultora["codigofactura"]=="WEB"]=1



campana_consultora["codigofactura_app"]=0

campana_consultora["codigofactura_app"].loc[campana_consultora["codigofactura"]=="APP"]=1



campana_consultora["codigofactura_apw"]=0

campana_consultora["codigofactura_apw"].loc[campana_consultora["codigofactura"]=="APW"]=1



campana_consultora["codigofactura_otros"]=0

campana_consultora["codigofactura_otros"].loc[~campana_consultora["codigofactura"].isin(["WEB","APP","APW"])]=1
historico_consultora = campana_consultora.pivot(index="idconsultora",columns="campana",values="codigofactura_web")

historico_consultora = pd.DataFrame(historico_consultora.to_records())



historico_predecir = pd.merge(predecir,historico_consultora,on='idconsultora',how='left')
historico_consultora.shape[1],historico_predecir.shape[1]
for i in range(15):

  n=historico_consultora.shape[1]

  auxi = pd.DataFrame(historico_consultora.iloc[:,(n-i-2):(n-1)])

  train["factura_web"+ str(i+1)+"UM"] = auxi.sum(axis=1)



for i in range(15):

  n=historico_predecir.shape[1]

  auxi = pd.DataFrame(historico_predecir.iloc[:,(n-i-1):(n)])

  test["factura_web"+ str(i+1)+"UM"] = auxi.sum(axis=1)
historico_consultora = campana_consultora.pivot(index="idconsultora",columns="campana",values="codigofactura_app")

historico_consultora = pd.DataFrame(historico_consultora.to_records())

historico_predecir = pd.merge(predecir,historico_consultora,on='idconsultora',how='left')
for i in range(15):

  n=historico_consultora.shape[1]

  auxi = pd.DataFrame(historico_consultora.iloc[:,(n-i-2):(n-1)])

  train["factura_app"+ str(i+1)+"UM"] = auxi.sum(axis=1)



for i in range(15):

  n=historico_predecir.shape[1]

  auxi = pd.DataFrame(historico_predecir.iloc[:,(n-i-1):(n)])

  test["factura_app"+ str(i+1)+"UM"] = auxi.sum(axis=1)
historico_consultora = campana_consultora.pivot(index="idconsultora",columns="campana",values="codigofactura_apw")

historico_consultora = pd.DataFrame(historico_consultora.to_records())

historico_predecir = pd.merge(predecir,historico_consultora,on='idconsultora',how='left')
for i in range(15):

  n=historico_consultora.shape[1]

  auxi = pd.DataFrame(historico_consultora.iloc[:,(n-i-2):(n-1)])

  train["factura_apw"+ str(i+1)+"UM"] = auxi.sum(axis=1)



for i in range(15):

  n=historico_predecir.shape[1]

  auxi = pd.DataFrame(historico_predecir.iloc[:,(n-i-1):(n)])

  test["factura_apw"+ str(i+1)+"UM"] = auxi.sum(axis=1)
historico_consultora = campana_consultora.pivot(index="idconsultora",columns="campana",values="codigofactura_otros")

historico_consultora = pd.DataFrame(historico_consultora.to_records())

historico_predecir = pd.merge(predecir,historico_consultora,on='idconsultora',how='left')
for i in range(15):

  n=historico_consultora.shape[1]

  auxi = pd.DataFrame(historico_consultora.iloc[:,(n-i-2):(n-1)])

  train["factura_otros"+ str(i+1)+"UM"] = auxi.sum(axis=1)



for i in range(15):

  n=historico_predecir.shape[1]

  auxi = pd.DataFrame(historico_predecir.iloc[:,(n-i-1):(n)])

  test["factura_otros"+ str(i+1)+"UM"] = auxi.sum(axis=1)
sns.catplot(y="evaluacion_nuevas", kind="count",palette="pastel", edgecolor=".6",data=campana_consultora,)
campana_consultora["evaluacion_nuevas_EST"]=0

campana_consultora["evaluacion_nuevas_EST"].loc[campana_consultora["evaluacion_nuevas"]=="Est"]=1



campana_consultora["evaluacion_nuevas_C_1d1"]=0

campana_consultora["evaluacion_nuevas_C_1d1"].loc[campana_consultora["evaluacion_nuevas"]=="C_1d1"]=1



campana_consultora["evaluacion_nuevas_C_2d2"]=0

campana_consultora["evaluacion_nuevas_C_2d2"].loc[campana_consultora["evaluacion_nuevas"]=="C_2d2"]=1



campana_consultora["evaluacion_nuevas_C_3d3"]=0

campana_consultora["evaluacion_nuevas_C_3d3"].loc[campana_consultora["evaluacion_nuevas"]=="C_3d3"]=1



campana_consultora["evaluacion_nuevas_otros"]=0

campana_consultora["evaluacion_nuevas_otros"].loc[~campana_consultora["evaluacion_nuevas"].isin(["Est","C_1d1","C_2d2","C_3d3"])]=1
historico_consultora = campana_consultora.pivot(index="idconsultora",columns="campana",values="evaluacion_nuevas_EST")

historico_consultora = pd.DataFrame(historico_consultora.to_records())



historico_predecir = pd.merge(predecir,historico_consultora,on='idconsultora',how='left')
for i in range(15):

  n=historico_consultora.shape[1]

  auxi = pd.DataFrame(historico_consultora.iloc[:,(n-i-2):(n-1)])

  train["evaluacion_nuevas_EST"+ str(i+1)+"UM"] = auxi.sum(axis=1)



for i in range(15):

  n=historico_predecir.shape[1]

  auxi = pd.DataFrame(historico_predecir.iloc[:,(n-i-1):(n)])

  test["evaluacion_nuevas_EST"+ str(i+1)+"UM"] = auxi.sum(axis=1)
historico_consultora = campana_consultora.pivot(index="idconsultora",columns="campana",values="evaluacion_nuevas_C_1d1")

historico_consultora = pd.DataFrame(historico_consultora.to_records())



historico_predecir = pd.merge(predecir,historico_consultora,on='idconsultora',how='left')
for i in range(15):

  n=historico_consultora.shape[1]

  auxi = pd.DataFrame(historico_consultora.iloc[:,(n-i-2):(n-1)])

  train["evaluacion_nuevas_C_1d1"+ str(i+1)+"UM"] = auxi.sum(axis=1)



for i in range(15):

  n=historico_predecir.shape[1]

  auxi = pd.DataFrame(historico_predecir.iloc[:,(n-i-1):(n)])

  test["evaluacion_nuevas_C_1d1"+ str(i+1)+"UM"] = auxi.sum(axis=1)
historico_consultora = campana_consultora.pivot(index="idconsultora",columns="campana",values="evaluacion_nuevas_C_2d2")

historico_consultora = pd.DataFrame(historico_consultora.to_records())



historico_predecir = pd.merge(predecir,historico_consultora,on='idconsultora',how='left')



for i in range(15):

  n=historico_consultora.shape[1]

  auxi = pd.DataFrame(historico_consultora.iloc[:,(n-i-2):(n-1)])

  train["evaluacion_nuevas_C_2d2"+ str(i+1)+"UM"] = auxi.sum(axis=1)



for i in range(15):

  n=historico_predecir.shape[1]

  auxi = pd.DataFrame(historico_predecir.iloc[:,(n-i-1):(n)])

  test["evaluacion_nuevas_C_2d2"+ str(i+1)+"UM"] = auxi.sum(axis=1)
historico_consultora = campana_consultora.pivot(index="idconsultora",columns="campana",values="evaluacion_nuevas_C_3d3")

historico_consultora = pd.DataFrame(historico_consultora.to_records())



historico_predecir = pd.merge(predecir,historico_consultora,on='idconsultora',how='left')



for i in range(15):

  n=historico_consultora.shape[1]

  auxi = pd.DataFrame(historico_consultora.iloc[:,(n-i-2):(n-1)])

  train["evaluacion_nuevas_C_3d3"+ str(i+1)+"UM"] = auxi.sum(axis=1)



for i in range(15):

  n=historico_predecir.shape[1]

  auxi = pd.DataFrame(historico_predecir.iloc[:,(n-i-1):(n)])

  test["evaluacion_nuevas_C_3d3"+ str(i+1)+"UM"] = auxi.sum(axis=1)
historico_consultora = campana_consultora.pivot(index="idconsultora",columns="campana",values="evaluacion_nuevas_otros")

historico_consultora = pd.DataFrame(historico_consultora.to_records())



historico_predecir = pd.merge(predecir,historico_consultora,on='idconsultora',how='left')



for i in range(15):

  n=historico_consultora.shape[1]

  auxi = pd.DataFrame(historico_consultora.iloc[:,(n-i-2):(n-1)])

  train["evaluacion_nuevas_otros"+ str(i+1)+"UM"] = auxi.sum(axis=1)



for i in range(15):

  n=historico_predecir.shape[1]

  auxi = pd.DataFrame(historico_predecir.iloc[:,(n-i-1):(n)])

  test["evaluacion_nuevas_otros"+ str(i+1)+"UM"] = auxi.sum(axis=1)
sns.catplot(y="segmentacion", kind="count",palette="pastel", edgecolor=".6",data=campana_consultora,)
campana_consultora.pivot(index="idconsultora",columns="campana",values="segmentacion").head(3)
mapping = {"Nuevas": 0, "Nivel7": 1, "Nivel6": 2, "Nivel5": 3, "Nivel4": 4, "Nivel3": 5, "Nivel2": 6, "Nivel1": 7, "Tops": 8}

campana_consultora["segmentacion"] = campana_consultora["segmentacion"].map(mapping)
campana_consultora.pivot(index="idconsultora",columns="campana",values="segmentacion").head(3)
historico_consultora = campana_consultora.pivot(index="idconsultora",columns="campana",values="segmentacion")

historico_consultora = pd.DataFrame(historico_consultora.to_records())



historico_predecir = pd.merge(predecir,historico_consultora,on='idconsultora',how='left')
train["segmentacionUM"] = historico_consultora["201905"]

train["segmentacion2UM"] = np.max(historico_consultora[["201905","201904"]],axis=1)

train["segmentacion3UM"] = np.max(historico_consultora[["201905","201904","201903"]],axis=1)

train["segmentacion6UM"] = np.max(historico_consultora[["201905","201904","201903","201902","201901","201818"]],axis=1)

train["segmentacion9UM"] = np.max(historico_consultora[["201905","201904","201903","201902","201901","201818","201817","201816","201815"]],axis=1)

train["segmentacion12UM"] = np.max(historico_consultora[["201905","201904","201903","201902","201901","201818","201817","201816","201815","201814","201814","201813"]],axis=1)



test["segmentacionUM"] = historico_predecir["201905"]

test["segmentacion2UM"] = np.max(historico_predecir[["201905","201904"]],axis=1)

test["segmentacion3UM"] = np.max(historico_predecir[["201905","201904","201903"]],axis=1)

test["segmentacion6UM"] = np.max(historico_predecir[["201905","201904","201903","201902","201901","201818"]],axis=1)

test["segmentacion9UM"] = np.max(historico_predecir[["201905","201904","201903","201902","201901","201818","201817","201816","201815"]],axis=1)

test["segmentacion12UM"] = np.max(historico_predecir[["201905","201904","201903","201902","201901","201818","201817","201816","201815","201814","201814","201813"]],axis=1)
campana_consultora.pivot(index="idconsultora",columns="campana",values="geografia").head(3)
sns.catplot(y="geografia", kind="count",palette="pastel", edgecolor=".6",data=campana_consultora,)
historico_consultora = campana_consultora.pivot(index="idconsultora",columns="campana",values="geografia")

historico_consultora = pd.DataFrame(historico_consultora.to_records())



historico_predecir = pd.merge(predecir,historico_consultora,on='idconsultora',how='left')
train["geografiaUM"] = historico_consultora["201818"]

test["geografiaUM"] = historico_predecir["201818"]
train = pd.merge(train,maestro_consultora,on='idconsultora',how='left')
train.shape
test = pd.merge(test,maestro_consultora,on='idconsultora',how='left')
test.shape
train["ncapanaingreso"] = train["campana"]-train["campanaingreso"]

test["ncapanaingreso"] = test["campana"]-test["campanaingreso"]
train["ncapanaingreso2"] = train["campanaultimopedido"]-train["campanaingreso"]

test["ncapanaingreso2"] = test["campanaultimopedido"]-test["campanaingreso"]
train["isnueva"]=0

train["isnueva"].loc[train["campanaprimerpedido"]==201906]=1



test["isnueva"]=0

test["isnueva"].loc[test["campanaprimerpedido"]==201907]=1
train["numcampanaultimopedido"] = train["campana"]-train["campanaultimopedido"]

test["numcampanaultimopedido"] = test["campana"]-test["campanaultimopedido"]



train["isultpedido"]=0

train["isultpedido"].loc[train["campanaultimopedido"]==201906]=1



test["isultpedido"]=0

test["isultpedido"].loc[test["campanaultimopedido"]==201907]=1
train["correovalidad"]=1

train["correovalidad"].loc[pd.isna(train["flagcorreovalidad"])==True]=0



test["correovalidad"]=1

test["correovalidad"].loc[pd.isna(test["flagcorreovalidad"])==True]=0
train = train.drop(["campanaingreso","campanaprimerpedido","campanaultimopedido","flagcorreovalidad"],axis=1)

test = test.drop(["campanaingreso","campanaprimerpedido","campanaultimopedido","flagcorreovalidad"],axis=1)
venta = pd.read_csv("../input/datathon-belcorp-prueba/dtt_fvta_cl.csv")
venta.shape
historico_consultora = venta.pivot_table(index="idconsultora",columns="campana",values="descuento", aggfunc="mean")

historico_consultora = pd.DataFrame(historico_consultora.to_records())

historico_consultora=historico_consultora.fillna(0)



historico_predecir = pd.merge(predecir,historico_consultora,on='idconsultora',how='left')
historico_consultora.shape,historico_predecir.shape
train["descuentomeanUM"] = historico_consultora["201905"]

train["descuentomean2UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])

train["descuentomean3UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])

train["descuentomean4UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])+pd.to_numeric(historico_consultora["201902"])

train["descuentomean6UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])+pd.to_numeric(historico_consultora["201902"])+pd.to_numeric(historico_consultora["201901"])+pd.to_numeric(historico_consultora["201818"])

train["descuentomean9UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])+pd.to_numeric(historico_consultora["201902"])+pd.to_numeric(historico_consultora["201901"])+pd.to_numeric(historico_consultora["201818"])+pd.to_numeric(historico_consultora["201817"])+pd.to_numeric(historico_consultora["201816"])+pd.to_numeric(historico_consultora["201815"])

train["descuentomean12UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])+pd.to_numeric(historico_consultora["201902"])+pd.to_numeric(historico_consultora["201901"])+pd.to_numeric(historico_consultora["201818"])+pd.to_numeric(historico_consultora["201817"])+pd.to_numeric(historico_consultora["201816"])+pd.to_numeric(historico_consultora["201815"])+pd.to_numeric(historico_consultora["201814"])+pd.to_numeric(historico_consultora["201813"])+pd.to_numeric(historico_consultora["201812"])



test["descuentomeanUM"] = historico_predecir["201906"]

test["descuentomean2UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])

test["descuentomean3UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])

test["descuentomean4UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])+pd.to_numeric(historico_predecir["201903"])

test["descuentomean6UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])+pd.to_numeric(historico_predecir["201903"])+pd.to_numeric(historico_predecir["201902"])+pd.to_numeric(historico_predecir["201901"])

test["descuentomean9UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])+pd.to_numeric(historico_predecir["201903"])+pd.to_numeric(historico_predecir["201902"])+pd.to_numeric(historico_predecir["201901"])+pd.to_numeric(historico_predecir["201818"])+pd.to_numeric(historico_predecir["201817"])+pd.to_numeric(historico_predecir["201816"])

test["descuentomean12UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])+pd.to_numeric(historico_predecir["201903"])+pd.to_numeric(historico_predecir["201902"])+pd.to_numeric(historico_predecir["201901"])+pd.to_numeric(historico_predecir["201818"])+pd.to_numeric(historico_predecir["201817"])+pd.to_numeric(historico_predecir["201816"])+pd.to_numeric(historico_predecir["201815"])+pd.to_numeric(historico_predecir["201814"])+pd.to_numeric(historico_predecir["201813"])
historico_consultora = venta.pivot_table(index="idconsultora",columns="campana",values="descuento", aggfunc="min")

historico_consultora = pd.DataFrame(historico_consultora.to_records())

historico_consultora=historico_consultora.fillna(0)



historico_predecir = pd.merge(predecir,historico_consultora,on='idconsultora',how='left')
train["descuentominUM"] = historico_consultora["201905"]

train["descuentomin3UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])

train["descuentomin6UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])+pd.to_numeric(historico_consultora["201902"])+pd.to_numeric(historico_consultora["201901"])+pd.to_numeric(historico_consultora["201818"])

train["descuentomin9UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])+pd.to_numeric(historico_consultora["201902"])+pd.to_numeric(historico_consultora["201901"])+pd.to_numeric(historico_consultora["201818"])+pd.to_numeric(historico_consultora["201817"])+pd.to_numeric(historico_consultora["201816"])+pd.to_numeric(historico_consultora["201815"])

train["descuentomin12UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])+pd.to_numeric(historico_consultora["201902"])+pd.to_numeric(historico_consultora["201901"])+pd.to_numeric(historico_consultora["201818"])+pd.to_numeric(historico_consultora["201817"])+pd.to_numeric(historico_consultora["201816"])+pd.to_numeric(historico_consultora["201815"])+pd.to_numeric(historico_consultora["201814"])+pd.to_numeric(historico_consultora["201813"])+pd.to_numeric(historico_consultora["201812"])



test["descuentominUM"] = historico_predecir["201906"]

test["descuentomin3UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])

test["descuentomin6UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])+pd.to_numeric(historico_predecir["201903"])+pd.to_numeric(historico_predecir["201902"])+pd.to_numeric(historico_predecir["201901"])

test["descuentomin9UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])+pd.to_numeric(historico_predecir["201903"])+pd.to_numeric(historico_predecir["201902"])+pd.to_numeric(historico_predecir["201901"])+pd.to_numeric(historico_predecir["201818"])+pd.to_numeric(historico_predecir["201817"])+pd.to_numeric(historico_predecir["201816"])

test["descuentomin12UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])+pd.to_numeric(historico_predecir["201903"])+pd.to_numeric(historico_predecir["201902"])+pd.to_numeric(historico_predecir["201901"])+pd.to_numeric(historico_predecir["201818"])+pd.to_numeric(historico_predecir["201817"])+pd.to_numeric(historico_predecir["201816"])+pd.to_numeric(historico_predecir["201815"])+pd.to_numeric(historico_predecir["201814"])+pd.to_numeric(historico_predecir["201813"])
historico_consultora = venta.pivot_table(index="idconsultora",columns="campana",values="descuento", aggfunc="max")

historico_consultora = pd.DataFrame(historico_consultora.to_records())

historico_consultora=historico_consultora.fillna(0)



historico_predecir = pd.merge(predecir,historico_consultora,on='idconsultora',how='left')



train["descuentomaxUM"] = historico_consultora["201905"]

train["descuentomax3UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])

train["descuentomax6UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])+pd.to_numeric(historico_consultora["201902"])+pd.to_numeric(historico_consultora["201901"])+pd.to_numeric(historico_consultora["201818"])

train["descuentomax9UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])+pd.to_numeric(historico_consultora["201902"])+pd.to_numeric(historico_consultora["201901"])+pd.to_numeric(historico_consultora["201818"])+pd.to_numeric(historico_consultora["201817"])+pd.to_numeric(historico_consultora["201816"])+pd.to_numeric(historico_consultora["201815"])

train["descuentomax12UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])+pd.to_numeric(historico_consultora["201902"])+pd.to_numeric(historico_consultora["201901"])+pd.to_numeric(historico_consultora["201818"])+pd.to_numeric(historico_consultora["201817"])+pd.to_numeric(historico_consultora["201816"])+pd.to_numeric(historico_consultora["201815"])+pd.to_numeric(historico_consultora["201814"])+pd.to_numeric(historico_consultora["201813"])+pd.to_numeric(historico_consultora["201812"])



test["descuentomaxUM"] = historico_predecir["201906"]

test["descuentomax3UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])

test["descuentomax6UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])+pd.to_numeric(historico_predecir["201903"])+pd.to_numeric(historico_predecir["201902"])+pd.to_numeric(historico_predecir["201901"])

test["descuentomax9UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])+pd.to_numeric(historico_predecir["201903"])+pd.to_numeric(historico_predecir["201902"])+pd.to_numeric(historico_predecir["201901"])+pd.to_numeric(historico_predecir["201818"])+pd.to_numeric(historico_predecir["201817"])+pd.to_numeric(historico_predecir["201816"])

test["descuentomax12UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])+pd.to_numeric(historico_predecir["201903"])+pd.to_numeric(historico_predecir["201902"])+pd.to_numeric(historico_predecir["201901"])+pd.to_numeric(historico_predecir["201818"])+pd.to_numeric(historico_predecir["201817"])+pd.to_numeric(historico_predecir["201816"])+pd.to_numeric(historico_predecir["201815"])+pd.to_numeric(historico_predecir["201814"])+pd.to_numeric(historico_predecir["201813"])
historico_consultora = venta.pivot_table(index="idconsultora",columns="campana",values="ahorro", aggfunc="mean")

historico_consultora = pd.DataFrame(historico_consultora.to_records())

historico_consultora=historico_consultora.fillna(0)



historico_predecir = pd.merge(predecir,historico_consultora,on='idconsultora',how='left')



train["ahorromeanUM"] = historico_consultora["201905"]

train["ahorromean2UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])

train["ahorromean3UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])

train["ahorromean6UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])+pd.to_numeric(historico_consultora["201902"])+pd.to_numeric(historico_consultora["201901"])+pd.to_numeric(historico_consultora["201818"])

train["ahorromean9UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])+pd.to_numeric(historico_consultora["201902"])+pd.to_numeric(historico_consultora["201901"])+pd.to_numeric(historico_consultora["201818"])+pd.to_numeric(historico_consultora["201817"])+pd.to_numeric(historico_consultora["201816"])+pd.to_numeric(historico_consultora["201815"])

train["ahorromean12UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])+pd.to_numeric(historico_consultora["201902"])+pd.to_numeric(historico_consultora["201901"])+pd.to_numeric(historico_consultora["201818"])+pd.to_numeric(historico_consultora["201817"])+pd.to_numeric(historico_consultora["201816"])+pd.to_numeric(historico_consultora["201815"])+pd.to_numeric(historico_consultora["201814"])+pd.to_numeric(historico_consultora["201813"])+pd.to_numeric(historico_consultora["201812"])



test["ahorromeanUM"] = historico_predecir["201906"]

test["ahorromean2UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])

test["ahorromean3UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])

test["ahorromean6UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])+pd.to_numeric(historico_predecir["201903"])+pd.to_numeric(historico_predecir["201902"])+pd.to_numeric(historico_predecir["201901"])

test["ahorromean9UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])+pd.to_numeric(historico_predecir["201903"])+pd.to_numeric(historico_predecir["201902"])+pd.to_numeric(historico_predecir["201901"])+pd.to_numeric(historico_predecir["201818"])+pd.to_numeric(historico_predecir["201817"])+pd.to_numeric(historico_predecir["201816"])

test["ahorromean12UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])+pd.to_numeric(historico_predecir["201903"])+pd.to_numeric(historico_predecir["201902"])+pd.to_numeric(historico_predecir["201901"])+pd.to_numeric(historico_predecir["201818"])+pd.to_numeric(historico_predecir["201817"])+pd.to_numeric(historico_predecir["201816"])+pd.to_numeric(historico_predecir["201815"])+pd.to_numeric(historico_predecir["201814"])+pd.to_numeric(historico_predecir["201813"])
historico_consultora = venta.pivot_table(index="idconsultora",columns="campana",values="ahorro", aggfunc="min")

historico_consultora = pd.DataFrame(historico_consultora.to_records())

historico_consultora=historico_consultora.fillna(0)



historico_predecir = pd.merge(predecir,historico_consultora,on='idconsultora',how='left')



train["ahorrominUM"] = historico_consultora["201905"]

train["ahorromin3UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])

train["ahorromin6UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])+pd.to_numeric(historico_consultora["201902"])+pd.to_numeric(historico_consultora["201901"])+pd.to_numeric(historico_consultora["201818"])

train["ahorromin9UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])+pd.to_numeric(historico_consultora["201902"])+pd.to_numeric(historico_consultora["201901"])+pd.to_numeric(historico_consultora["201818"])+pd.to_numeric(historico_consultora["201817"])+pd.to_numeric(historico_consultora["201816"])+pd.to_numeric(historico_consultora["201815"])

train["ahorromin12UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])+pd.to_numeric(historico_consultora["201902"])+pd.to_numeric(historico_consultora["201901"])+pd.to_numeric(historico_consultora["201818"])+pd.to_numeric(historico_consultora["201817"])+pd.to_numeric(historico_consultora["201816"])+pd.to_numeric(historico_consultora["201815"])+pd.to_numeric(historico_consultora["201814"])+pd.to_numeric(historico_consultora["201813"])+pd.to_numeric(historico_consultora["201812"])



test["ahorrominUM"] = historico_predecir["201906"]

test["ahorromin3UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])

test["ahorromin6UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])+pd.to_numeric(historico_predecir["201903"])+pd.to_numeric(historico_predecir["201902"])+pd.to_numeric(historico_predecir["201901"])

test["ahorromin9UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])+pd.to_numeric(historico_predecir["201903"])+pd.to_numeric(historico_predecir["201902"])+pd.to_numeric(historico_predecir["201901"])+pd.to_numeric(historico_predecir["201818"])+pd.to_numeric(historico_predecir["201817"])+pd.to_numeric(historico_predecir["201816"])

test["ahorromin12UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])+pd.to_numeric(historico_predecir["201903"])+pd.to_numeric(historico_predecir["201902"])+pd.to_numeric(historico_predecir["201901"])+pd.to_numeric(historico_predecir["201818"])+pd.to_numeric(historico_predecir["201817"])+pd.to_numeric(historico_predecir["201816"])+pd.to_numeric(historico_predecir["201815"])+pd.to_numeric(historico_predecir["201814"])+pd.to_numeric(historico_predecir["201813"])
historico_consultora = venta.pivot_table(index="idconsultora",columns="campana",values="ahorro", aggfunc="max")

historico_consultora = pd.DataFrame(historico_consultora.to_records())

historico_consultora=historico_consultora.fillna(0)



historico_predecir = pd.merge(predecir,historico_consultora,on='idconsultora',how='left')



train["ahorromaxUM"] = historico_consultora["201905"]

train["ahorromax3UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])

train["ahorromax6UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])+pd.to_numeric(historico_consultora["201902"])+pd.to_numeric(historico_consultora["201901"])+pd.to_numeric(historico_consultora["201818"])

train["ahorromax9UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])+pd.to_numeric(historico_consultora["201902"])+pd.to_numeric(historico_consultora["201901"])+pd.to_numeric(historico_consultora["201818"])+pd.to_numeric(historico_consultora["201817"])+pd.to_numeric(historico_consultora["201816"])+pd.to_numeric(historico_consultora["201815"])

train["ahorromax12UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])+pd.to_numeric(historico_consultora["201902"])+pd.to_numeric(historico_consultora["201901"])+pd.to_numeric(historico_consultora["201818"])+pd.to_numeric(historico_consultora["201817"])+pd.to_numeric(historico_consultora["201816"])+pd.to_numeric(historico_consultora["201815"])+pd.to_numeric(historico_consultora["201814"])+pd.to_numeric(historico_consultora["201813"])+pd.to_numeric(historico_consultora["201812"])



test["ahorromaxUM"] = historico_predecir["201906"]

test["ahorromax3UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])

test["ahorromax6UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])+pd.to_numeric(historico_predecir["201903"])+pd.to_numeric(historico_predecir["201902"])+pd.to_numeric(historico_predecir["201901"])

test["ahorromax9UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])+pd.to_numeric(historico_predecir["201903"])+pd.to_numeric(historico_predecir["201902"])+pd.to_numeric(historico_predecir["201901"])+pd.to_numeric(historico_predecir["201818"])+pd.to_numeric(historico_predecir["201817"])+pd.to_numeric(historico_predecir["201816"])

test["ahorromax12UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])+pd.to_numeric(historico_predecir["201903"])+pd.to_numeric(historico_predecir["201902"])+pd.to_numeric(historico_predecir["201901"])+pd.to_numeric(historico_predecir["201818"])+pd.to_numeric(historico_predecir["201817"])+pd.to_numeric(historico_predecir["201816"])+pd.to_numeric(historico_predecir["201815"])+pd.to_numeric(historico_predecir["201814"])+pd.to_numeric(historico_predecir["201813"])
historico_consultora = venta.pivot_table(index="idconsultora",columns="campana",values="preciocatalogo", aggfunc="mean")

historico_consultora = pd.DataFrame(historico_consultora.to_records())

historico_consultora=historico_consultora.fillna(0)



historico_predecir = pd.merge(predecir,historico_consultora,on='idconsultora',how='left')



train["preciocatalogomeanUM"] = historico_consultora["201905"]

train["preciocatalogomean2UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])

train["preciocatalogomean3UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])

train["preciocatalogomean6UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])+pd.to_numeric(historico_consultora["201902"])+pd.to_numeric(historico_consultora["201901"])+pd.to_numeric(historico_consultora["201818"])

train["preciocatalogomean9UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])+pd.to_numeric(historico_consultora["201902"])+pd.to_numeric(historico_consultora["201901"])+pd.to_numeric(historico_consultora["201818"])+pd.to_numeric(historico_consultora["201817"])+pd.to_numeric(historico_consultora["201816"])+pd.to_numeric(historico_consultora["201815"])

train["preciocatalogomean12UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])+pd.to_numeric(historico_consultora["201902"])+pd.to_numeric(historico_consultora["201901"])+pd.to_numeric(historico_consultora["201818"])+pd.to_numeric(historico_consultora["201817"])+pd.to_numeric(historico_consultora["201816"])+pd.to_numeric(historico_consultora["201815"])+pd.to_numeric(historico_consultora["201814"])+pd.to_numeric(historico_consultora["201813"])+pd.to_numeric(historico_consultora["201812"])



test["preciocatalogomeanUM"] = historico_predecir["201906"]

test["preciocatalogomean2UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])

test["preciocatalogomean3UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])

test["preciocatalogomean6UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])+pd.to_numeric(historico_predecir["201903"])+pd.to_numeric(historico_predecir["201902"])+pd.to_numeric(historico_predecir["201901"])

test["preciocatalogomean9UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])+pd.to_numeric(historico_predecir["201903"])+pd.to_numeric(historico_predecir["201902"])+pd.to_numeric(historico_predecir["201901"])+pd.to_numeric(historico_predecir["201818"])+pd.to_numeric(historico_predecir["201817"])+pd.to_numeric(historico_predecir["201816"])

test["preciocatalogomean12UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])+pd.to_numeric(historico_predecir["201903"])+pd.to_numeric(historico_predecir["201902"])+pd.to_numeric(historico_predecir["201901"])+pd.to_numeric(historico_predecir["201818"])+pd.to_numeric(historico_predecir["201817"])+pd.to_numeric(historico_predecir["201816"])+pd.to_numeric(historico_predecir["201815"])+pd.to_numeric(historico_predecir["201814"])+pd.to_numeric(historico_predecir["201813"])
historico_consultora = venta.pivot_table(index="idconsultora",columns="campana",values="preciocatalogo", aggfunc="min")

historico_consultora = pd.DataFrame(historico_consultora.to_records())

historico_consultora=historico_consultora.fillna(0)



historico_predecir = pd.merge(predecir,historico_consultora,on='idconsultora',how='left')



train["preciocatalogominUM"] = historico_consultora["201905"]

train["preciocatalogomin3UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])

train["preciocatalogomin6UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])+pd.to_numeric(historico_consultora["201902"])+pd.to_numeric(historico_consultora["201901"])+pd.to_numeric(historico_consultora["201818"])

train["preciocatalogomin9UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])+pd.to_numeric(historico_consultora["201902"])+pd.to_numeric(historico_consultora["201901"])+pd.to_numeric(historico_consultora["201818"])+pd.to_numeric(historico_consultora["201817"])+pd.to_numeric(historico_consultora["201816"])+pd.to_numeric(historico_consultora["201815"])

train["preciocatalogomin12UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])+pd.to_numeric(historico_consultora["201902"])+pd.to_numeric(historico_consultora["201901"])+pd.to_numeric(historico_consultora["201818"])+pd.to_numeric(historico_consultora["201817"])+pd.to_numeric(historico_consultora["201816"])+pd.to_numeric(historico_consultora["201815"])+pd.to_numeric(historico_consultora["201814"])+pd.to_numeric(historico_consultora["201813"])+pd.to_numeric(historico_consultora["201812"])



test["preciocatalogominUM"] = historico_predecir["201906"]

test["preciocatalogomin3UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])

test["preciocatalogomin6UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])+pd.to_numeric(historico_predecir["201903"])+pd.to_numeric(historico_predecir["201902"])+pd.to_numeric(historico_predecir["201901"])

test["preciocatalogomin9UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])+pd.to_numeric(historico_predecir["201903"])+pd.to_numeric(historico_predecir["201902"])+pd.to_numeric(historico_predecir["201901"])+pd.to_numeric(historico_predecir["201818"])+pd.to_numeric(historico_predecir["201817"])+pd.to_numeric(historico_predecir["201816"])

test["preciocatalogomin12UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])+pd.to_numeric(historico_predecir["201903"])+pd.to_numeric(historico_predecir["201902"])+pd.to_numeric(historico_predecir["201901"])+pd.to_numeric(historico_predecir["201818"])+pd.to_numeric(historico_predecir["201817"])+pd.to_numeric(historico_predecir["201816"])+pd.to_numeric(historico_predecir["201815"])+pd.to_numeric(historico_predecir["201814"])+pd.to_numeric(historico_predecir["201813"])
historico_consultora = venta.pivot_table(index="idconsultora",columns="campana",values="preciocatalogo", aggfunc="max")

historico_consultora = pd.DataFrame(historico_consultora.to_records())

historico_consultora=historico_consultora.fillna(0)



historico_predecir = pd.merge(predecir,historico_consultora,on='idconsultora',how='left')



train["preciocatalogomaxUM"] = historico_consultora["201905"]

train["preciocatalogomax3UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])

train["preciocatalogomax6UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])+pd.to_numeric(historico_consultora["201902"])+pd.to_numeric(historico_consultora["201901"])+pd.to_numeric(historico_consultora["201818"])

train["preciocatalogomax9UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])+pd.to_numeric(historico_consultora["201902"])+pd.to_numeric(historico_consultora["201901"])+pd.to_numeric(historico_consultora["201818"])+pd.to_numeric(historico_consultora["201817"])+pd.to_numeric(historico_consultora["201816"])+pd.to_numeric(historico_consultora["201815"])

train["preciocatalogomax12UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])+pd.to_numeric(historico_consultora["201902"])+pd.to_numeric(historico_consultora["201901"])+pd.to_numeric(historico_consultora["201818"])+pd.to_numeric(historico_consultora["201817"])+pd.to_numeric(historico_consultora["201816"])+pd.to_numeric(historico_consultora["201815"])+pd.to_numeric(historico_consultora["201814"])+pd.to_numeric(historico_consultora["201813"])+pd.to_numeric(historico_consultora["201812"])



test["preciocatalogomaxUM"] = historico_predecir["201906"]

test["preciocatalogomax3UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])

test["preciocatalogomax6UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])+pd.to_numeric(historico_predecir["201903"])+pd.to_numeric(historico_predecir["201902"])+pd.to_numeric(historico_predecir["201901"])

test["preciocatalogomax9UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])+pd.to_numeric(historico_predecir["201903"])+pd.to_numeric(historico_predecir["201902"])+pd.to_numeric(historico_predecir["201901"])+pd.to_numeric(historico_predecir["201818"])+pd.to_numeric(historico_predecir["201817"])+pd.to_numeric(historico_predecir["201816"])

test["preciocatalogomax12UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])+pd.to_numeric(historico_predecir["201903"])+pd.to_numeric(historico_predecir["201902"])+pd.to_numeric(historico_predecir["201901"])+pd.to_numeric(historico_predecir["201818"])+pd.to_numeric(historico_predecir["201817"])+pd.to_numeric(historico_predecir["201816"])+pd.to_numeric(historico_predecir["201815"])+pd.to_numeric(historico_predecir["201814"])+pd.to_numeric(historico_predecir["201813"])
historico_consultora = venta.pivot_table(index="idconsultora",columns="campana",values="realvtamnneto", aggfunc="mean")

historico_consultora = pd.DataFrame(historico_consultora.to_records())

historico_consultora=historico_consultora.fillna(0)



historico_predecir = pd.merge(predecir,historico_consultora,on='idconsultora',how='left')



train["realvtamnnetomeanUM"] = historico_consultora["201905"]

train["realvtamnnetomean2UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])

train["realvtamnnetomean3UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])

train["realvtamnnetomean6UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])+pd.to_numeric(historico_consultora["201902"])+pd.to_numeric(historico_consultora["201901"])+pd.to_numeric(historico_consultora["201818"])

train["realvtamnnetomean9UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])+pd.to_numeric(historico_consultora["201902"])+pd.to_numeric(historico_consultora["201901"])+pd.to_numeric(historico_consultora["201818"])+pd.to_numeric(historico_consultora["201817"])+pd.to_numeric(historico_consultora["201816"])+pd.to_numeric(historico_consultora["201815"])

train["realvtamnnetomean12UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])+pd.to_numeric(historico_consultora["201902"])+pd.to_numeric(historico_consultora["201901"])+pd.to_numeric(historico_consultora["201818"])+pd.to_numeric(historico_consultora["201817"])+pd.to_numeric(historico_consultora["201816"])+pd.to_numeric(historico_consultora["201815"])+pd.to_numeric(historico_consultora["201814"])+pd.to_numeric(historico_consultora["201813"])+pd.to_numeric(historico_consultora["201812"])



test["realvtamnnetomeanUM"] = historico_predecir["201906"]

test["realvtamnnetomean2UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])

test["realvtamnnetomean3UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])

test["realvtamnnetomean6UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])+pd.to_numeric(historico_predecir["201903"])+pd.to_numeric(historico_predecir["201902"])+pd.to_numeric(historico_predecir["201901"])

test["realvtamnnetomean9UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])+pd.to_numeric(historico_predecir["201903"])+pd.to_numeric(historico_predecir["201902"])+pd.to_numeric(historico_predecir["201901"])+pd.to_numeric(historico_predecir["201818"])+pd.to_numeric(historico_predecir["201817"])+pd.to_numeric(historico_predecir["201816"])

test["realvtamnnetomean12UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])+pd.to_numeric(historico_predecir["201903"])+pd.to_numeric(historico_predecir["201902"])+pd.to_numeric(historico_predecir["201901"])+pd.to_numeric(historico_predecir["201818"])+pd.to_numeric(historico_predecir["201817"])+pd.to_numeric(historico_predecir["201816"])+pd.to_numeric(historico_predecir["201815"])+pd.to_numeric(historico_predecir["201814"])+pd.to_numeric(historico_predecir["201813"])
historico_consultora = venta.pivot_table(index="idconsultora",columns="campana",values="realvtamnneto", aggfunc="min")

historico_consultora = pd.DataFrame(historico_consultora.to_records())

historico_consultora=historico_consultora.fillna(0)



historico_predecir = pd.merge(predecir,historico_consultora,on='idconsultora',how='left')



train["realvtamnnetominUM"] = historico_consultora["201905"]

train["realvtamnnetomin3UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])

train["realvtamnnetomin6UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])+pd.to_numeric(historico_consultora["201902"])+pd.to_numeric(historico_consultora["201901"])+pd.to_numeric(historico_consultora["201818"])

train["realvtamnnetomin9UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])+pd.to_numeric(historico_consultora["201902"])+pd.to_numeric(historico_consultora["201901"])+pd.to_numeric(historico_consultora["201818"])+pd.to_numeric(historico_consultora["201817"])+pd.to_numeric(historico_consultora["201816"])+pd.to_numeric(historico_consultora["201815"])

train["realvtamnnetomin12UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])+pd.to_numeric(historico_consultora["201902"])+pd.to_numeric(historico_consultora["201901"])+pd.to_numeric(historico_consultora["201818"])+pd.to_numeric(historico_consultora["201817"])+pd.to_numeric(historico_consultora["201816"])+pd.to_numeric(historico_consultora["201815"])+pd.to_numeric(historico_consultora["201814"])+pd.to_numeric(historico_consultora["201813"])+pd.to_numeric(historico_consultora["201812"])



test["realvtamnnetominUM"] = historico_predecir["201906"]

test["realvtamnnetomin3UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])

test["realvtamnnetomin6UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])+pd.to_numeric(historico_predecir["201903"])+pd.to_numeric(historico_predecir["201902"])+pd.to_numeric(historico_predecir["201901"])

test["realvtamnnetomin9UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])+pd.to_numeric(historico_predecir["201903"])+pd.to_numeric(historico_predecir["201902"])+pd.to_numeric(historico_predecir["201901"])+pd.to_numeric(historico_predecir["201818"])+pd.to_numeric(historico_predecir["201817"])+pd.to_numeric(historico_predecir["201816"])

test["realvtamnnetomin12UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])+pd.to_numeric(historico_predecir["201903"])+pd.to_numeric(historico_predecir["201902"])+pd.to_numeric(historico_predecir["201901"])+pd.to_numeric(historico_predecir["201818"])+pd.to_numeric(historico_predecir["201817"])+pd.to_numeric(historico_predecir["201816"])+pd.to_numeric(historico_predecir["201815"])+pd.to_numeric(historico_predecir["201814"])+pd.to_numeric(historico_predecir["201813"])
historico_consultora = venta.pivot_table(index="idconsultora",columns="campana",values="realvtamnneto", aggfunc="max")

historico_consultora = pd.DataFrame(historico_consultora.to_records())

historico_consultora=historico_consultora.fillna(0)



historico_predecir = pd.merge(predecir,historico_consultora,on='idconsultora',how='left')



train["realvtamnnetomaxUM"] = historico_consultora["201905"]

train["realvtamnnetomax3UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])

train["realvtamnnetomax6UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])+pd.to_numeric(historico_consultora["201902"])+pd.to_numeric(historico_consultora["201901"])+pd.to_numeric(historico_consultora["201818"])

train["realvtamnnetomax9UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])+pd.to_numeric(historico_consultora["201902"])+pd.to_numeric(historico_consultora["201901"])+pd.to_numeric(historico_consultora["201818"])+pd.to_numeric(historico_consultora["201817"])+pd.to_numeric(historico_consultora["201816"])+pd.to_numeric(historico_consultora["201815"])

train["realvtamnnetomax12UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])+pd.to_numeric(historico_consultora["201902"])+pd.to_numeric(historico_consultora["201901"])+pd.to_numeric(historico_consultora["201818"])+pd.to_numeric(historico_consultora["201817"])+pd.to_numeric(historico_consultora["201816"])+pd.to_numeric(historico_consultora["201815"])+pd.to_numeric(historico_consultora["201814"])+pd.to_numeric(historico_consultora["201813"])+pd.to_numeric(historico_consultora["201812"])



test["realvtamnnetomaxUM"] = historico_predecir["201906"]

test["realvtamnnetomax3UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])

test["realvtamnnetomax6UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])+pd.to_numeric(historico_predecir["201903"])+pd.to_numeric(historico_predecir["201902"])+pd.to_numeric(historico_predecir["201901"])

test["realvtamnnetomax9UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])+pd.to_numeric(historico_predecir["201903"])+pd.to_numeric(historico_predecir["201902"])+pd.to_numeric(historico_predecir["201901"])+pd.to_numeric(historico_predecir["201818"])+pd.to_numeric(historico_predecir["201817"])+pd.to_numeric(historico_predecir["201816"])

test["realvtamnnetomax12UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])+pd.to_numeric(historico_predecir["201903"])+pd.to_numeric(historico_predecir["201902"])+pd.to_numeric(historico_predecir["201901"])+pd.to_numeric(historico_predecir["201818"])+pd.to_numeric(historico_predecir["201817"])+pd.to_numeric(historico_predecir["201816"])+pd.to_numeric(historico_predecir["201815"])+pd.to_numeric(historico_predecir["201814"])+pd.to_numeric(historico_predecir["201813"])
historico_consultora = venta.pivot_table(index="idconsultora",columns="campana",values="realuuvendidas", aggfunc="sum")

historico_consultora = pd.DataFrame(historico_consultora.to_records())

historico_consultora=historico_consultora.fillna(0)



historico_predecir = pd.merge(predecir,historico_consultora,on='idconsultora',how='left')



train["realuuvendidassumUM"] = historico_consultora["201905"]

train["realuuvendidassum2UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])

train["realuuvendidassum3UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])

train["realuuvendidassum6UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])+pd.to_numeric(historico_consultora["201902"])+pd.to_numeric(historico_consultora["201901"])+pd.to_numeric(historico_consultora["201818"])

train["realuuvendidassum9UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])+pd.to_numeric(historico_consultora["201902"])+pd.to_numeric(historico_consultora["201901"])+pd.to_numeric(historico_consultora["201818"])+pd.to_numeric(historico_consultora["201817"])+pd.to_numeric(historico_consultora["201816"])+pd.to_numeric(historico_consultora["201815"])

train["realuuvendidassum12UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])+pd.to_numeric(historico_consultora["201902"])+pd.to_numeric(historico_consultora["201901"])+pd.to_numeric(historico_consultora["201818"])+pd.to_numeric(historico_consultora["201817"])+pd.to_numeric(historico_consultora["201816"])+pd.to_numeric(historico_consultora["201815"])+pd.to_numeric(historico_consultora["201814"])+pd.to_numeric(historico_consultora["201813"])+pd.to_numeric(historico_consultora["201812"])



test["realuuvendidassumUM"] = historico_predecir["201906"]

test["realuuvendidassum2UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])

test["realuuvendidassum3UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])

test["realuuvendidassum6UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])+pd.to_numeric(historico_predecir["201903"])+pd.to_numeric(historico_predecir["201902"])+pd.to_numeric(historico_predecir["201901"])

test["realuuvendidassum9UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])+pd.to_numeric(historico_predecir["201903"])+pd.to_numeric(historico_predecir["201902"])+pd.to_numeric(historico_predecir["201901"])+pd.to_numeric(historico_predecir["201818"])+pd.to_numeric(historico_predecir["201817"])+pd.to_numeric(historico_predecir["201816"])

test["realuuvendidassum12UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])+pd.to_numeric(historico_predecir["201903"])+pd.to_numeric(historico_predecir["201902"])+pd.to_numeric(historico_predecir["201901"])+pd.to_numeric(historico_predecir["201818"])+pd.to_numeric(historico_predecir["201817"])+pd.to_numeric(historico_predecir["201816"])+pd.to_numeric(historico_predecir["201815"])+pd.to_numeric(historico_predecir["201814"])+pd.to_numeric(historico_predecir["201813"])
historico_consultora = venta.pivot_table(index="idconsultora",columns="campana",values="realuufaltantes", aggfunc="sum")

historico_consultora = pd.DataFrame(historico_consultora.to_records())

historico_consultora=historico_consultora.fillna(0)



historico_predecir = pd.merge(predecir,historico_consultora,on='idconsultora',how='left')



train["realuufaltantessumUM"] = historico_consultora["201905"]

train["realuufaltantessum2UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])

train["realuufaltantessum3UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])

train["realuufaltantessum6UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])+pd.to_numeric(historico_consultora["201902"])+pd.to_numeric(historico_consultora["201901"])+pd.to_numeric(historico_consultora["201818"])

train["realuufaltantessum9UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])+pd.to_numeric(historico_consultora["201902"])+pd.to_numeric(historico_consultora["201901"])+pd.to_numeric(historico_consultora["201818"])+pd.to_numeric(historico_consultora["201817"])+pd.to_numeric(historico_consultora["201816"])+pd.to_numeric(historico_consultora["201815"])

train["realuufaltantessum12UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])+pd.to_numeric(historico_consultora["201902"])+pd.to_numeric(historico_consultora["201901"])+pd.to_numeric(historico_consultora["201818"])+pd.to_numeric(historico_consultora["201817"])+pd.to_numeric(historico_consultora["201816"])+pd.to_numeric(historico_consultora["201815"])+pd.to_numeric(historico_consultora["201814"])+pd.to_numeric(historico_consultora["201813"])+pd.to_numeric(historico_consultora["201812"])



test["realuufaltantessumUM"] = historico_predecir["201906"]

test["realuufaltantessum2UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])

test["realuufaltantessum3UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])

test["realuufaltantessum6UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])+pd.to_numeric(historico_predecir["201903"])+pd.to_numeric(historico_predecir["201902"])+pd.to_numeric(historico_predecir["201901"])

test["realuufaltantessum9UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])+pd.to_numeric(historico_predecir["201903"])+pd.to_numeric(historico_predecir["201902"])+pd.to_numeric(historico_predecir["201901"])+pd.to_numeric(historico_predecir["201818"])+pd.to_numeric(historico_predecir["201817"])+pd.to_numeric(historico_predecir["201816"])

test["realuufaltantessum12UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])+pd.to_numeric(historico_predecir["201903"])+pd.to_numeric(historico_predecir["201902"])+pd.to_numeric(historico_predecir["201901"])+pd.to_numeric(historico_predecir["201818"])+pd.to_numeric(historico_predecir["201817"])+pd.to_numeric(historico_predecir["201816"])+pd.to_numeric(historico_predecir["201815"])+pd.to_numeric(historico_predecir["201814"])+pd.to_numeric(historico_predecir["201813"])
historico_consultora = venta.pivot_table(index="idconsultora",columns="campana",values="realuudevueltas", aggfunc="sum")

historico_consultora = pd.DataFrame(historico_consultora.to_records())

historico_consultora=historico_consultora.fillna(0)



historico_predecir = pd.merge(predecir,historico_consultora,on='idconsultora',how='left')



train["realuudevueltassumUM"] = historico_consultora["201905"]

train["realuudevueltassum2UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])

train["realuudevueltassum3UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])

train["realuudevueltassum6UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])+pd.to_numeric(historico_consultora["201902"])+pd.to_numeric(historico_consultora["201901"])+pd.to_numeric(historico_consultora["201818"])

train["realuudevueltassum9UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])+pd.to_numeric(historico_consultora["201902"])+pd.to_numeric(historico_consultora["201901"])+pd.to_numeric(historico_consultora["201818"])+pd.to_numeric(historico_consultora["201817"])+pd.to_numeric(historico_consultora["201816"])+pd.to_numeric(historico_consultora["201815"])

train["realuudevueltassum12UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])+pd.to_numeric(historico_consultora["201902"])+pd.to_numeric(historico_consultora["201901"])+pd.to_numeric(historico_consultora["201818"])+pd.to_numeric(historico_consultora["201817"])+pd.to_numeric(historico_consultora["201816"])+pd.to_numeric(historico_consultora["201815"])+pd.to_numeric(historico_consultora["201814"])+pd.to_numeric(historico_consultora["201813"])+pd.to_numeric(historico_consultora["201812"])



test["realuudevueltassumUM"] = historico_predecir["201906"]

test["realuudevueltassum2UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])

test["realuudevueltassum3UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])

test["realuudevueltassum6UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])+pd.to_numeric(historico_predecir["201903"])+pd.to_numeric(historico_predecir["201902"])+pd.to_numeric(historico_predecir["201901"])

test["realuudevueltassum9UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])+pd.to_numeric(historico_predecir["201903"])+pd.to_numeric(historico_predecir["201902"])+pd.to_numeric(historico_predecir["201901"])+pd.to_numeric(historico_predecir["201818"])+pd.to_numeric(historico_predecir["201817"])+pd.to_numeric(historico_predecir["201816"])

test["realuudevueltassum12UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])+pd.to_numeric(historico_predecir["201903"])+pd.to_numeric(historico_predecir["201902"])+pd.to_numeric(historico_predecir["201901"])+pd.to_numeric(historico_predecir["201818"])+pd.to_numeric(historico_predecir["201817"])+pd.to_numeric(historico_predecir["201816"])+pd.to_numeric(historico_predecir["201815"])+pd.to_numeric(historico_predecir["201814"])+pd.to_numeric(historico_predecir["201813"])
historico_consultora = venta.pivot_table(index="idconsultora",columns="campana",values="realuuanuladas", aggfunc="sum")

historico_consultora = pd.DataFrame(historico_consultora.to_records())

historico_consultora=historico_consultora.fillna(0)



historico_predecir = pd.merge(predecir,historico_consultora,on='idconsultora',how='left')



train["realuuanuladassumUM"] = historico_consultora["201905"]

train["realuuanuladassum2UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])

train["realuuanuladassum3UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])

train["realuuanuladassum6UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])+pd.to_numeric(historico_consultora["201902"])+pd.to_numeric(historico_consultora["201901"])+pd.to_numeric(historico_consultora["201818"])

train["realuuanuladassum9UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])+pd.to_numeric(historico_consultora["201902"])+pd.to_numeric(historico_consultora["201901"])+pd.to_numeric(historico_consultora["201818"])+pd.to_numeric(historico_consultora["201817"])+pd.to_numeric(historico_consultora["201816"])+pd.to_numeric(historico_consultora["201815"])

train["realuuanuladassum12UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])+pd.to_numeric(historico_consultora["201902"])+pd.to_numeric(historico_consultora["201901"])+pd.to_numeric(historico_consultora["201818"])+pd.to_numeric(historico_consultora["201817"])+pd.to_numeric(historico_consultora["201816"])+pd.to_numeric(historico_consultora["201815"])+pd.to_numeric(historico_consultora["201814"])+pd.to_numeric(historico_consultora["201813"])+pd.to_numeric(historico_consultora["201812"])



test["realuuanuladassumUM"] = historico_predecir["201906"]

test["realuuanuladassum2UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])

test["realuuanuladassum3UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])

test["realuuanuladassum6UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])+pd.to_numeric(historico_predecir["201903"])+pd.to_numeric(historico_predecir["201902"])+pd.to_numeric(historico_predecir["201901"])

test["realuuanuladassum9UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])+pd.to_numeric(historico_predecir["201903"])+pd.to_numeric(historico_predecir["201902"])+pd.to_numeric(historico_predecir["201901"])+pd.to_numeric(historico_predecir["201818"])+pd.to_numeric(historico_predecir["201817"])+pd.to_numeric(historico_predecir["201816"])

test["realuuanuladassum12UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])+pd.to_numeric(historico_predecir["201903"])+pd.to_numeric(historico_predecir["201902"])+pd.to_numeric(historico_predecir["201901"])+pd.to_numeric(historico_predecir["201818"])+pd.to_numeric(historico_predecir["201817"])+pd.to_numeric(historico_predecir["201816"])+pd.to_numeric(historico_predecir["201815"])+pd.to_numeric(historico_predecir["201814"])+pd.to_numeric(historico_predecir["201813"])
historico_consultora = venta.pivot_table(index="idconsultora",columns="campana",values="realdevmnneto", aggfunc="mean")

historico_consultora = pd.DataFrame(historico_consultora.to_records())

historico_consultora=historico_consultora.fillna(0)



historico_predecir = pd.merge(predecir,historico_consultora,on='idconsultora',how='left')



train["realdevmnnetomeanUM"] = historico_consultora["201905"]

train["realdevmnnetomean2UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])

train["realdevmnnetomean3UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])

train["realdevmnnetomean6UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])+pd.to_numeric(historico_consultora["201902"])+pd.to_numeric(historico_consultora["201901"])+pd.to_numeric(historico_consultora["201818"])

train["realdevmnnetomean9UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])+pd.to_numeric(historico_consultora["201902"])+pd.to_numeric(historico_consultora["201901"])+pd.to_numeric(historico_consultora["201818"])+pd.to_numeric(historico_consultora["201817"])+pd.to_numeric(historico_consultora["201816"])+pd.to_numeric(historico_consultora["201815"])

train["realdevmnnetomean12UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])+pd.to_numeric(historico_consultora["201902"])+pd.to_numeric(historico_consultora["201901"])+pd.to_numeric(historico_consultora["201818"])+pd.to_numeric(historico_consultora["201817"])+pd.to_numeric(historico_consultora["201816"])+pd.to_numeric(historico_consultora["201815"])+pd.to_numeric(historico_consultora["201814"])+pd.to_numeric(historico_consultora["201813"])+pd.to_numeric(historico_consultora["201812"])



test["realdevmnnetomeanUM"] = historico_predecir["201906"]

test["realdevmnnetomean2UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])

test["realdevmnnetomean3UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])

test["realdevmnnetomean6UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])+pd.to_numeric(historico_predecir["201903"])+pd.to_numeric(historico_predecir["201902"])+pd.to_numeric(historico_predecir["201901"])

test["realdevmnnetomean9UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])+pd.to_numeric(historico_predecir["201903"])+pd.to_numeric(historico_predecir["201902"])+pd.to_numeric(historico_predecir["201901"])+pd.to_numeric(historico_predecir["201818"])+pd.to_numeric(historico_predecir["201817"])+pd.to_numeric(historico_predecir["201816"])

test["realdevmnnetomean12UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])+pd.to_numeric(historico_predecir["201903"])+pd.to_numeric(historico_predecir["201902"])+pd.to_numeric(historico_predecir["201901"])+pd.to_numeric(historico_predecir["201818"])+pd.to_numeric(historico_predecir["201817"])+pd.to_numeric(historico_predecir["201816"])+pd.to_numeric(historico_predecir["201815"])+pd.to_numeric(historico_predecir["201814"])+pd.to_numeric(historico_predecir["201813"])
historico_consultora = venta.pivot_table(index="idconsultora",columns="campana",values="realanulmnneto", aggfunc="mean")

historico_consultora = pd.DataFrame(historico_consultora.to_records())

historico_consultora=historico_consultora.fillna(0)



historico_predecir = pd.merge(predecir,historico_consultora,on='idconsultora',how='left')



train["realanulmnnetomeanUM"] = historico_consultora["201905"]

train["realanulmnnetomean3UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])

train["realanulmnnetomean6UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])+pd.to_numeric(historico_consultora["201902"])+pd.to_numeric(historico_consultora["201901"])+pd.to_numeric(historico_consultora["201818"])

train["realanulmnnetomean9UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])+pd.to_numeric(historico_consultora["201902"])+pd.to_numeric(historico_consultora["201901"])+pd.to_numeric(historico_consultora["201818"])+pd.to_numeric(historico_consultora["201817"])+pd.to_numeric(historico_consultora["201816"])+pd.to_numeric(historico_consultora["201815"])

train["realanulmnnetomean12UM"] = pd.to_numeric(historico_consultora["201905"])+ pd.to_numeric(historico_consultora["201904"])+pd.to_numeric(historico_consultora["201903"])+pd.to_numeric(historico_consultora["201902"])+pd.to_numeric(historico_consultora["201901"])+pd.to_numeric(historico_consultora["201818"])+pd.to_numeric(historico_consultora["201817"])+pd.to_numeric(historico_consultora["201816"])+pd.to_numeric(historico_consultora["201815"])+pd.to_numeric(historico_consultora["201814"])+pd.to_numeric(historico_consultora["201813"])+pd.to_numeric(historico_consultora["201812"])



test["realanulmnnetomeanUM"] = historico_predecir["201906"]

test["realanulmnnetomean3UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])

test["realanulmnnetomean6UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])+pd.to_numeric(historico_predecir["201903"])+pd.to_numeric(historico_predecir["201902"])+pd.to_numeric(historico_predecir["201901"])

test["realanulmnnetomean9UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])+pd.to_numeric(historico_predecir["201903"])+pd.to_numeric(historico_predecir["201902"])+pd.to_numeric(historico_predecir["201901"])+pd.to_numeric(historico_predecir["201818"])+pd.to_numeric(historico_predecir["201817"])+pd.to_numeric(historico_predecir["201816"])

test["realanulmnnetomean12UM"] = pd.to_numeric(historico_predecir["201906"])+ pd.to_numeric(historico_predecir["201905"])+pd.to_numeric(historico_predecir["201904"])+pd.to_numeric(historico_predecir["201903"])+pd.to_numeric(historico_predecir["201902"])+pd.to_numeric(historico_predecir["201901"])+pd.to_numeric(historico_predecir["201818"])+pd.to_numeric(historico_predecir["201817"])+pd.to_numeric(historico_predecir["201816"])+pd.to_numeric(historico_predecir["201815"])+pd.to_numeric(historico_predecir["201814"])+pd.to_numeric(historico_predecir["201813"])
train.shape,test.shape
#detalle_venta = pd.read_csv("Matriz_road_cruce.csv")
#detalle_venta_train=detalle_venta[detalle_venta['campana']==201906]

#detalle_venta_train=detalle_venta_train.drop(['campana'],axis=1)

#detalle_venta_test=detalle_venta[detalle_venta['campana']==201907]

#detalle_venta_test=detalle_venta_test.drop(['campana'],axis=1)

#detalle_venta_train.head(5)
#detalle_venta_train.shape,detalle_venta_test.shape
#train=pd.merge(train,detalle_venta_train,on='idconsultora',how='left')

#test=pd.merge(test,detalle_venta_test,on='idconsultora',how='left')
#Columnas_con_missing(train)
#train.factura_app12UM.unique
import lightgbm as lgb

from sklearn import preprocessing

import gc

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score
features=[ x for x in train.columns if x not in ['campana','idconsultora','flagpasopedido']]



categorical=['estadocivil','geografiaUM']



cat_ind=[features.index(x) for x in categorical if x in features]



target='flagpasopedido'
for l in categorical:

    le = preprocessing.LabelEncoder()

    le.fit(list(train[l].dropna())+list(test[l].dropna()))

    train.loc[~train[l].isnull(),l]=le.transform(train.loc[~train[l].isnull(),l])

    test.loc[~test[l].isnull(),l]=le.transform(test.loc[~test[l].isnull(),l])
kf_previo=StratifiedKFold(n_splits=15,random_state=256,shuffle=True)



i=1



r=[]



importancias=pd.DataFrame()



importancias['variable']=features



for train_index,test_index in kf_previo.split(train,train[target]):



    lgb_train = lgb.Dataset(train.loc[train_index,features].values,train.loc[train_index,target].values.ravel())

    lgb_eval = lgb.Dataset(train.loc[test_index,features].values,train.loc[test_index,target].values.ravel(), reference=lgb_train)



    params = {

        'task': 'train',

        'boosting_type': 'gbdt',

        'objective': 'binary',

        'metric': { 'auc'},

        "max_depth":6,

        "num_leaves":10,

        'learning_rate': 0.1,

    "min_child_samples": 100,

        'feature_fraction': 0.5,

     "bagging_freq":1,

        'bagging_fraction': 0.9,

        "lambda_l1":1,

        "lambda_l2":1,

       #"scale_pos_weight":30,



        'verbose': 1    

    }



    lgbm3 = lgb.train(params,lgb_train,num_boost_round=13100,valid_sets=lgb_eval,early_stopping_rounds=50,verbose_eval=25,categorical_feature=cat_ind)

    test["TARGET_FOLD"+str(i)]=lgbm3.predict(test[features].values, num_iteration=lgbm3.best_iteration)

 

    importancias['gain_'+str(i)]=lgbm3.feature_importance(importance_type="gain")



    

    print ("Fold_"+str(i))

    a= (roc_auc_score(train.loc[test_index,target],lgbm3.predict(train.loc[test_index,features].values, num_iteration=lgbm3.best_iteration)))

    r.append(a)

    print (a)

    print ("")

    

    i=i+1



print ("mean: "+str(np.mean(np.array(r))))

print ("std: "+str(np.std(np.array(r))))
w=[x for x in importancias.columns if 'gain_' in x]

importancias['gain-avg']=importancias[w].mean(axis=1)

importancias=importancias.sort_values('gain-avg',ascending=False).reset_index(drop=True)
importancias.head(10)
(importancias['gain-avg']==0).value_counts()
importancias['gain-avg'].describe()
features_selected=importancias.loc[(importancias["gain-avg"]>100),"variable"].to_list()

cat_ind_selected=[features_selected.index(x) for x in categorical if x in features_selected]
len(features_selected)
train[target].value_counts()
kf_previo=StratifiedKFold(n_splits=15,random_state=256,shuffle=True)



i=1



r=[]



importancias_2=pd.DataFrame()



importancias_2['variable']=features_selected





for train_index,test_index in kf_previo.split(train,train[target]):



    lgb_train = lgb.Dataset(train.loc[train_index,features_selected].values,train.loc[train_index,target].values.ravel())

    lgb_eval = lgb.Dataset(train.loc[test_index,features_selected].values,train.loc[test_index,target].values.ravel(), reference=lgb_train)



    params = {

        'task': 'train',

        'boosting_type': 'gbdt',

        'objective': 'binary',

        'metric': { 'auc'},

        "max_depth":6,

        "num_leaves":32,

        'learning_rate': 0.05,

    "min_child_samples": 150,

        'feature_fraction': 0.5,

     "bagging_freq":1,

        'bagging_fraction': 0.95,

        "lambda_l1":1,

        "lambda_l2":1,

       #"scale_pos_weight":30,



        'verbose': 1    

    }









    lgbm3 = lgb.train(params,lgb_train,num_boost_round=13100,valid_sets=lgb_eval,early_stopping_rounds=50,verbose_eval=25,categorical_feature=cat_ind_selected)

    test["TARGET_FOLD"+str(i)]=lgbm3.predict(test[features_selected].values, num_iteration=lgbm3.best_iteration)

    

    importancias_2['gain_'+str(i)]=lgbm3.feature_importance(importance_type="gain")



    

    print ("Fold_"+str(i))

    a= (roc_auc_score(train.loc[test_index,target],lgbm3.predict(train.loc[test_index,features_selected].values, num_iteration=lgbm3.best_iteration)))

    r.append(a)

    print (a)

    print ("")

    

    i=i+1



print ("mean: "+str(np.mean(np.array(r))))

print ("std: "+str(np.std(np.array(r))))
w=[x for x in importancias_2.columns if 'gain_' in x]

importancias_2['gain-avg']=importancias_2[w].mean(axis=1)

importancias_2=importancias_2.sort_values('gain-avg',ascending=False).reset_index(drop=True)
importancias_2.head(10)
w=[x for x in test.columns if 'FOLD' in x]



test['flagpasopedido']=test[w].max(axis=1)
test[['idconsultora','flagpasopedido']].to_csv('mean2.csv',index=False)
#features=[ x for x in train.columns if x not in ['campana','idconsultora','flagpasopedido']]



#categorical=['estadocivil','geografiaUM']



#cat_ind=[features.index(x) for x in categorical if x in features]



#target='flagpasopedido'
#for l in categorical:

 #   le = preprocessing.LabelEncoder()

 #   le.fit(list(train[l].dropna())+list(test[l].dropna()))



    #train.loc[~train[l].isnull(),l]=le.transform(train.loc[~train[l].isnull(),l])



    #test.loc[~test[l].isnull(),l]=le.transform(test.loc[~test[l].isnull(),l])
#train_df=train

#test_df=test

#full_vars = features

#cat_vars = categorical



#full_vars = features

#target_var ="flagpasopedido"

#train_x = train_df[full_vars].values

#train_y = train_df[target_var].values

#test_x = test_df[full_vars].values#



#import copy

#default_lgb_params = {}

#default_lgb_params["learning_rate"] = 0.1 

#default_lgb_params["metric"] = 'auc'

#default_lgb_params["bagging_freq"] = 1

#default_lgb_params["seed"] = 42

#default_lgb_params["objective"] = "binary"

#default_lgb_params["boost_from_average"] = "false"#



#params_lgb_space = {}

#params_lgb_space['feature_fraction'] = np.arange(0.1, 1, 0.1)

#params_lgb_space['num_leaves'] = [2, 4, 8, 16, 32,64]  

#params_lgb_space['max_depth'] = [3 ,4 ,5 ,6, -1,8]

#params_lgb_space['min_gain_to_split'] = [0, 0.1, 0.3, 1, 1.5, 2, 3]

#params_lgb_space['bagging_fraction'] = np.arange(0.1, 1, 0.1)

#params_lgb_space['min_sum_hessian_in_leaf'] = [1, 5, 10, 30, 100]

#params_lgb_space['lambda_l1'] = [0, 0.01, 0.1, 1, 5,10, 100, 300]

#params_lgb_space['lambda_l2'] = [0, 0.01, 0.1, 1, 5,10, 100, 300]#



#greater_is_better = True#



#best_lgb_params = copy.copy(default_lgb_params)#

#



#for p in params_lgb_space: 

#    print ("\n Tuning parameter %s in %s" % (p, params_lgb_space[p]))

#    params = best_lgb_params

#    scores = []    

#    for v in params_lgb_space[p]: 

#        gc.collect()

#        print ('\n    %s: %s' % (p, v), end="\n")

#        params[p] = v

#        

#        cv_results = lgb.cv(params, 

#                        lgb.Dataset(train_x, label=train_y), 

#                        stratified=True,

#                        shuffle=True,

#                        nfold=5,

#                        num_boost_round=100000,

#                        early_stopping_rounds=100,

#                        verbose_eval=0)

#        

#        best_lgb_score = max(cv_results['auc-mean'])

#        print ('Score: %f ' % (best_lgb_score))

#        scores.append([v, best_lgb_score])#



#    # best param value in the space

#    best_param_value = sorted(scores, key=lambda x:x[1],reverse=greater_is_better)[0][0]

#    best_param_score = sorted(scores, key=lambda x:x[1],reverse=greater_is_better)[0][1]

#    best_lgb_params[p] = best_param_value

#    print ("Best %s is %s with a score of %f" %(p, best_param_value, best_param_score))#



#    

#best_param_value = sorted(scores, key=lambda x:x[1],reverse=greater_is_better)[0][0]

#best_param_score = sorted(scores, key=lambda x:x[1],reverse=greater_is_better)[0][1]

#best_lgb_params[p] = best_param_value

#print ("Best %s is %s with a score of %f" %(p, best_param_value, best_param_score))

#print ('\n Best manually tuned parameters:', best_lgb_params)   
#kf_previo=StratifiedKFold(n_splits=10,random_state=256,shuffle=True)#



#i=1#



#r=[]#



#importancias=pd.DataFrame()#



#importancias['variable']=features#



#for train_index,test_index in kf_previo.split(train,train[target]):#



#    lgb_train = lgb.Dataset(train.loc[train_index,features].values,train.loc[train_index,target].values.ravel())

#    lgb_eval = lgb.Dataset(train.loc[test_index,features].values,train.loc[test_index,target].values.ravel(), reference=lgb_train)#



#    params = {

#        'task': 'train',

#        'boosting_type': 'gbdt',

#        'objective': 'binary',

#        'metric': { 'auc'},

#        "max_depth":3,

#        "num_leaves":4,

#        "min_gain_to_split":1,

#        'learning_rate': 0.1,

#        "min_child_samples": 100,

#        'feature_fraction': 0.6,

#        'min_sum_hessian_in_leaf':5,

#        "bagging_freq":1,

#        'bagging_fraction': 0.8,

#        "lambda_l1":1,

#        "lambda_l2":0.1,

#       # "scale_pos_weight":30,#



#        'verbose': 1    

#    }#



#    lgbm3 = lgb.train(params,lgb_train,num_boost_round=13100,valid_sets=lgb_eval,early_stopping_rounds=50,verbose_eval=25,categorical_feature=cat_ind)

#    test["TARGET_FOLD"+str(i)]=lgbm3.predict(test[features].values, num_iteration=lgbm3.best_iteration)

# 

#    importancias['gain_'+str(i)]=lgbm3.feature_importance(importance_type="gain")#



#    

#    print ("Fold_"+str(i))

#    a= (roc_auc_score(train.loc[test_index,target],lgbm3.predict(train.loc[test_index,features].values, num_iteration=lgbm3.best_iteration)))

#    r.append(a)

#    print (a)

#    print ("")

#    

#    i=i+1#



#print ("mean: "+str(np.mean(np.array(r))))

#print ("std: "+str(np.std(np.array(r))))
#w=[x for x in importancias.columns if 'gain_' in x]

#importancias['gain-avg']=importancias[w].mean(axis=1)

#importancias=importancias.sort_values('gain-avg',ascending=False).reset_index(drop=True)

#importancias
#(importancias['gain-avg']==0).value_counts()
#importancias['gain-avg'].describe()
#features_selected=importancias.loc[(importancias["gain-avg"]>100),"variable"].to_list()

#cat_ind_selected=[features_selected.index(x) for x in categorical if x in features_selected]
#len(features_selected)
#kf_previo=StratifiedKFold(n_splits=30,random_state=256,shuffle=True)#



#i=1#



#r=[]#



#importancias_2=pd.DataFrame()#



#importancias_2['variable']=features_selected#

#



#for train_index,test_index in kf_previo.split(train,train[target]):#



#    lgb_train = lgb.Dataset(train.loc[train_index,features_selected].values,train.loc[train_index,target].values.ravel())

#    lgb_eval = lgb.Dataset(train.loc[test_index,features_selected].values,train.loc[test_index,target].values.ravel(), reference=lgb_train)#



#    params = {

#        'task': 'train',

#        'boosting_type': 'gbdt',

#        'objective': 'binary',

#        'metric': { 'auc'},

#        "max_depth":3,

#        "num_leaves":4,

#        "min_gain_to_split":1,

#        'learning_rate': 0.1,

#        "min_child_samples": 100,

#        'feature_fraction': 0.6,

#        'min_sum_hessian_in_leaf':5,

#        "bagging_freq":1,

#        'bagging_fraction': 0.8,

#        "lambda_l1":1,

#        "lambda_l2":0.1,

#       # "scale_pos_weight":30,#



#        'verbose': 1    

#    }#

#

#

#



#    lgbm3 = lgb.train(params,lgb_train,num_boost_round=13100,valid_sets=lgb_eval,early_stopping_rounds=50,verbose_eval=25,categorical_feature=cat_ind_selected)

#    test["TARGET_FOLD"+str(i)]=lgbm3.predict(test[features_selected].values, num_iteration=lgbm3.best_iteration)

#    

#    importancias_2['gain_'+str(i)]=lgbm3.feature_importance(importance_type="gain")#



#    

#    print ("Fold_"+str(i))

#    a= (roc_auc_score(train.loc[test_index,target],lgbm3.predict(train.loc[test_index,features_selected].values, num_iteration=lgbm3.best_iteration)))

#    r.append(a)

#    print (a)

#    print ("")

#    

#    i=i+1#



#print ("mean: "+str(np.mean(np.array(r))))

#print ("std: "+str(np.std(np.array(r))))
#w=[x for x in importancias_2.columns if 'gain_' in x]

#importancias_2['gain-avg']=importancias_2[w].mean(axis=1)

#importancias_2=importancias_2.sort_values('gain-avg',ascending=False).reset_index(drop=True)

#importancias_2
#w=[x for x in test.columns if 'FOLD' in x]



#test['flagpasopedido']=test[w].mean(axis=1)
#test[['idconsultora','flagpasopedido']].to_csv('mean_tune.csv',index=False)