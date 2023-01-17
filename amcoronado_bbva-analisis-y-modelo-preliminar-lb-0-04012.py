# importando librerias
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import collections
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost
import math as mt
import warnings

# Ploting styles
# styles: 'fivethirtyeight', 'classic', 'ggplot', 'seaborn-notebook'
# styles: 'seaborn-poster', 'bmh', 'grayscale', 'seaborn-whitegrid'
matplotlib.style.use('bmh')
%matplotlib inline
pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)
#print(plt.style.available)
warnings.filterwarnings("ignore")
# dataset de entrenamiento a nivel de transacciones
TrainTrx = pd.read_csv('../input/bbvadata/01dataBaseTrainTrxRec.csv')

TrainTrx.info()
# eliminando columnas no relevantes
TrainTrx.drop(['fechaOper', 'ctdTrx'], axis=1, inplace=True)

# ordenando por cliente y establecimiento
TrainTrx.sort_values(by=['codCliente','codEstab']).head()
# dataset del perfil de cliente
TrainPer = pd.read_csv('../input/bbvadata/02dataBasePerfilRec.csv')

TrainPer.info()
# ordenando por cliente
TrainPer.sort_values(by=['codCliente']).head()
# fusionando datos de transacciones y clientes
TrainTot = pd.merge(TrainTrx, TrainPer, on='codCliente')

TrainTot.info()
# ordenando por cliente y establecimiento
TrainTot.sort_values(by=['codCliente','codEstab']).head()
# descriptive statistics summary
TrainTot['ratingMonto'].describe()
# boxplot
sns.boxplot(x=TrainTot['ratingMonto']);
# definiendo un nuevo data frame con ratingMonto<0.02
TrainTot1 = TrainTot[TrainTot['ratingMonto'] < 0.02]
# boxplot
sns.boxplot(x=TrainTot1['ratingMonto']);
TrainTot1.flagGenero.plot.hist();
TrainTot1.flagLimaProvEstab.plot.hist();
TrainTot1.flagLimaProvCliente.plot.hist();
TrainTot1.flagBxi.plot.hist();
sns.jointplot(x='codCliente', y='ratingMonto', data=TrainTot1, kind="hex");
sns.jointplot(x='codGiro', y='ratingMonto', data=TrainTot1, kind="hex");
sns.jointplot(x='codEstab', y='ratingMonto', data=TrainTot1, kind="hex");
sns.jointplot(x='ubigeoEstab', y='ratingMonto', data=TrainTot1, kind="hex");
f, ax = plt.subplots(figsize=(12, 5))
sns.boxplot(x='rangoEdad', y='ratingMonto', data=TrainTot1);
f, ax = plt.subplots(figsize=(12, 5))
sns.boxplot(x='rangoIngreso', y='ratingMonto', data=TrainTot1);
f, ax = plt.subplots(figsize=(12, 5))
sns.boxplot(x='rangoCtdProdAct', y='ratingMonto', data=TrainTot1);
f, ax = plt.subplots(figsize=(12, 5))
sns.boxplot(x='rangoCtdProdPas', y='ratingMonto', data=TrainTot1);
f, ax = plt.subplots(figsize=(12, 5))
sns.boxplot(x='rangoCtdProdSeg', y='ratingMonto', data=TrainTot1);
f, ax = plt.subplots(figsize=(12, 5))
sns.boxplot(x='saldoTcEntidad1', y='ratingMonto', data=TrainTot1);
f, ax = plt.subplots(figsize=(12, 5))
sns.boxplot(x='saldoTcEntidad2', y='ratingMonto', data=TrainTot1);
f, ax = plt.subplots(figsize=(12, 5))
sns.boxplot(x='saldoTcEntidad3', y='ratingMonto', data=TrainTot1);
f, ax = plt.subplots(figsize=(12, 5))
sns.boxplot(x='saldoTcEntidad4', y='ratingMonto', data=TrainTot1);
# dataset de prueba para base a nivel de clientes-establecimientos
BaseSub = pd.read_csv('../input/bbvadata/03dataBaseTestRec.csv')

BaseSub.info()
# desagregado el campo [codClienteCodEstab] en "codCliente" y "codEstab"
BasePred = pd.read_csv('../input/bbvadata/05dataBaseTestKeyRec.csv')

BasePred.sort_values(by=['codCliente', 'codEstab']).head()
BasePred.info()
# aislando los features de establecimientos
FeatEstab = TrainTrx.drop(['codCliente','ratingMonto'], axis=1)

# eliminando duplicados
FeatEstab = FeatEstab.drop_duplicates()

FeatEstab.info()
# ordenando por establecimiento
FeatEstab.sort_values(by=['codEstab']).head()
# generando features para data de test
TesTot = pd.merge(BasePred, TrainPer, how='left', on=['codCliente'])

TesTot = pd.merge(TesTot, FeatEstab, how='left', on=['codEstab'])

TesTot.sort_values(by=['codCliente', 'codEstab']).head()
TesTot.info()
# concatenando features de train y test
TrainTest = pd.concat((TrainTot.drop(['ratingMonto'],axis=1),
                                   TesTot))

TrainTest.info()
# conversion de variables categoricas
TrainTest = pd.get_dummies(TrainTest)

TrainTest.sort_values(by=['codCliente','codEstab']).head()                        
# datos de entrenamiento
X = TrainTest[:TrainTot.shape[0]]
y = TrainTot.ratingMonto

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)

# datos de validacion
X_val = TrainTest[TrainTot.shape[0]:]
# xgboost
xgb = xgboost.XGBRegressor(n_jobs=-1,n_estimators=300,subsample=0.75,max_depth=10)
# ajuste de modelo
xgb.fit(X_train,y_train)
# RMSE para datos de entrenamiento
y_p_train = xgb.predict(X_train)
print('RSME',mt.sqrt(mean_squared_error(y_train, y_p_train)))
# RMSE para datos de test
y_p_test = xgb.predict(X_test)
print('RSME',mt.sqrt(mean_squared_error(y_test, y_p_test)))
# prediccion para datos de validacion
y_val = xgb.predict(X_val)

y_val = pd.DataFrame(y_val,columns=['ratingMonto'])

y_val.info()
y_sub = BasePred.copy()

y_sub['ratingMonto'] = y_val

y_sub.info()
# fusion de columnas
col = (y_sub['codCliente'].map(str) + y_sub['codEstab'].map(str))

y_sub['codClienteCodEstab'] = col

# reordenando columnas
y_sub = y_sub[['codClienteCodEstab','ratingMonto']]

# convirtiendo a enteros columna codClienteCodEstab
y_sub['codClienteCodEstab']=y_sub['codClienteCodEstab'].apply(int)

y_sub.info()
# reordenamiento de resultados
y_fin = pd.merge(BaseSub,y_sub,on=['codClienteCodEstab'])

y_fin.info()
# generacion de archivo de submision
y_fin.to_csv('BBVA.csv', index=False)
