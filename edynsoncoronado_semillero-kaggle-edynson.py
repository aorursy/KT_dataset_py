import pandas as pd

import numpy as np

import matplotlib.pylab as pylab 

from matplotlib import pyplot as plt

import seaborn as sns

from scipy import stats 

from sklearn import metrics as mt



%matplotlib inline

pylab.rcParams['figure.figsize'] = 6,4



# Ignorar warnings

import warnings

warnings.filterwarnings("ignore")
# Seleccion de Variables a usar en este BASELINE:

features_iniciales = ['ID',

 'Sexo',

 'AdultoMayor',

 'MesesCliente',

 'ServicioTelefonico',

 'LineasMultiples',

 'ProteccionDispositivo',

 'SoporteTecnico',

 'FacturacionElectronica',

 'MontoCargadoMes']
import os

print(os.listdir("../input"))
# Import from



path = "../input/"

df_train = pd.read_csv(path+"churn_data_train.csv",encoding='latin-1', usecols=features_iniciales+['Churn'])

df_test = pd.read_csv(path+"churn_data_test.csv",encoding='latin-1', usecols=features_iniciales)
df_train.shape, df_test.shape
df_train.head()
df_test.head()
df_train.info()
df_test.info()
# Defining features types

ID = 'ID'

TARGET = 'Churn'
# Distribución del Target

df_train[TARGET].value_counts(dropna=False)
df_train[TARGET].value_counts(dropna=False, normalize = True)*100
# Generar estadisticos básicos para cada variable:

### count: Count number of non-NA/null observations.	

### unique: Count uniques numbers of non-NA/null observations.

### top: Mean of the values.

### freq: Mean of the values.



### mean: Mean of the values.

### std: Standard deviation of the observations.



### min: Minimum of the values in the object.

### X%: The value of Quartil: 25% - Q1 , 50% - Q2, 75% - Q3

### max: Maximum of the values in the object.



df_train['AdultoMayor'] = df_train['AdultoMayor'].astype(str) # Convertir a variable categorica

df_train.describe(include = 'all').T
df_train['AdultoMayor'] = df_train['AdultoMayor'].astype(float) # Convertir a variable numerica
import missingno as msno

msno.matrix(df_train)
msno.matrix(df_test)
None
None
# Copy dataset and then apply transformation to copied dataset

ds_train = df_train.copy()
ds_test = df_test.copy()
# AdultoMayor (imputacion por MODA)

ds_train["AdultoMayor"].fillna(0, inplace = True)

ds_test["AdultoMayor"].fillna(0, inplace = True)



# MesesCliente (imputacion por MEDIA)

ds_train["MesesCliente"].fillna(32, inplace = True)

ds_test["MesesCliente"].fillna(32, inplace = True)



# ProteccionDispositivo (imputacion por MODA)

ds_train["ProteccionDispositivo"].fillna('No', inplace = True)

ds_test["ProteccionDispositivo"].fillna('No', inplace = True)



# SoporteTecnico (imputacion por MODA)

ds_train["SoporteTecnico"].fillna('No', inplace = True)

ds_test["SoporteTecnico"].fillna('No', inplace = True)



# FacturacionElectronica (imputacion por MEDIA)

ds_train["FacturacionElectronica"].fillna('Si', inplace = True)

ds_test["FacturacionElectronica"].fillna('Si', inplace = True)



# MontoCargadoMes (imputacion por MEDIA)

ds_train["MontoCargadoMes"].fillna(68.7, inplace = True)

ds_test["MontoCargadoMes"].fillna(68.7, inplace = True)
None
ds_train.head()
# Sexo 

dicc_sexo = {'Masculino': 1, 'Femenino':0 }

ds_train["Sexo"] = ds_train["Sexo"].map(dicc_sexo)

ds_test["Sexo"] = ds_test["Sexo"].map(dicc_sexo)
# ServicioTelefonico 

dicc_serv_telef = {'Si': 1, 'No':0 }

ds_train["ServicioTelefonico"] = ds_train["ServicioTelefonico"].map(dicc_serv_telef)

ds_test["ServicioTelefonico"] = ds_test["ServicioTelefonico"].map(dicc_serv_telef)
# LineasMultiples 

dicc_lin_mult = {'Si': 2, 'No':1, 'Sin servicio telefonico':0 }

ds_train["LineasMultiples"] = ds_train["LineasMultiples"].map(dicc_lin_mult)

ds_test["LineasMultiples"] = ds_test["LineasMultiples"].map(dicc_lin_mult)
# FacturacionElectronica 

dicc_fact_elect = {'Si': 1, 'No':0 }

ds_train["FacturacionElectronica"] = ds_train["FacturacionElectronica"].map(dicc_fact_elect)

ds_test["FacturacionElectronica"] = ds_test["FacturacionElectronica"].map(dicc_fact_elect)
# Crear Features Dummies

ds_train.loc[ds_train['ProteccionDispositivo']=='Sin servicio de internet', 'ProteccionDispositivo'] = 'SinServInter'

ds_train.loc[ds_train['SoporteTecnico']=='Sin servicio de internet', 'SoporteTecnico'] = 'SinServInter'



ds_test.loc[ds_test['ProteccionDispositivo']=='Sin servicio de internet', 'ProteccionDispositivo'] = 'SinServInter'

ds_test.loc[ds_test['SoporteTecnico']=='Sin servicio de internet', 'SoporteTecnico'] = 'SinServInter'



ds_train = pd.get_dummies(ds_train, columns=['ProteccionDispositivo','SoporteTecnico'])

ds_test = pd.get_dummies(ds_test, columns=['ProteccionDispositivo','SoporteTecnico'])
ds_train.head()
# New Feature 1

tmp_byAdultoMayor_medianMontoMes = ds_train.groupby(['AdultoMayor'])['MontoCargadoMes'].median().round()

tmp_byAdultoMayor_medianMontoMes
ds_train['flg_bySexo_mayorMedianMontoMes'] = ds_train.apply(lambda x: 1 if x.MontoCargadoMes >= tmp_byAdultoMayor_medianMontoMes[x.AdultoMayor] else 0,

                                                       axis = 1)



ds_test['flg_bySexo_mayorMedianMontoMes'] = ds_test.apply(lambda x: 1 if x.MontoCargadoMes >= tmp_byAdultoMayor_medianMontoMes[x.AdultoMayor] else 0,

                                                       axis = 1)
ds_train.head(10)
# New Feature 2,3,4, ...

### Here
features_to_model = list(ds_train.columns)



features_to_model.remove(TARGET) # Eliminar variable Target

features_to_model.remove(ID) # Eliminar variable ID



list(features_to_model)
# Selección de variables. 

### Una opción es: en base a un modelo basado en árboles, generar la importancia de Variables y seleccionar los features mas importantes.

features_to_model = features_to_model # ['var1', 'var2', 'varn'] 
len(features_to_model)
# Features & Target

X = ds_train[features_to_model]

y = ds_train[TARGET]



X_summit = ds_test[features_to_model]
print("train: ", X.shape,", summit: ", X_summit.shape)
from sklearn import model_selection

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size = 0.70, random_state=9)

print((len(X_train), len(y_train)), (len(X_test), len(y_test)))
X_train.info()
from sklearn.linear_model import LogisticRegression
# Create  model objet 

model_rlog = LogisticRegression(C=0.01, max_iter= 100, random_state=0, n_jobs = 4, penalty = 'l1')



# Fit the model:

model_rlog.fit(X_train, y_train)



model = model_rlog 
df_weights = pd.DataFrame({'feature':X_train.columns.values, 'beta': np.round(model_rlog.coef_[0],4) })

df_weights
# Generar las predicciones:

y_pred_train = model.predict(X_train)

y_pred_test = model.predict(X_test)



# Generar las probabilidades

y_pred_proba_train = model.predict_proba(X_train)[:,1]

y_pred_proba_test = model.predict_proba(X_test)[:,1]
accuracy_train = mt.accuracy_score(y_train, y_pred_train)

accuracy_test = mt.accuracy_score(y_test, y_pred_test)



print("Accuracy - Train: {}".format(accuracy_train))

print("Accuracy - Test : {}".format(accuracy_test))
list_accuracy_test = []

for threshold in range(0,100):

  pred_0_1 = [1 if x >= threshold/100 else 0 for x in y_pred_proba_test]

  list_accuracy_test.append(mt.accuracy_score(y_test, pred_0_1))
xs = [x/100 for x in range(0,100)]

ys = list_accuracy_test

plt.plot(xs, ys)
best_scoring = max(list_accuracy_test)

best_threshold = list_accuracy_test.index(best_scoring)/100

print("El mejor threshold es: {}".format(best_threshold))
accuracy_train = mt.accuracy_score(y_train, [1 if x >= best_threshold else 0 for x in y_pred_proba_train])

accuracy_test = mt.accuracy_score(y_test, [1 if x >= best_threshold else 0 for x in y_pred_proba_test])



print("Accuracy - Train: {}".format(accuracy_train))

print("Accuracy - Test : {}".format(accuracy_test))
from sklearn.tree import DecisionTreeClassifier

DecisionTreeClassifier()
# Create  model objet 

model_tree = DecisionTreeClassifier(max_depth=6, min_samples_leaf=5,random_state=0)



# Fit the model:

model_tree.fit(X_train, y_train)



model = model_tree
# Generar las predicciones:

y_pred_train = model.predict(X_train)

y_pred_test = model.predict(X_test)



# Generar las probabilidades

y_pred_proba_train = model.predict_proba(X_train)[:,1]

y_pred_proba_test = model.predict_proba(X_test)[:,1]
accuracy_train = mt.accuracy_score(y_train, y_pred_train)

accuracy_test = mt.accuracy_score(y_test, y_pred_test)



print("Accuracy - Train: {}".format(accuracy_train))

print("Accuracy - Test : {}".format(accuracy_test))
list_accuracy_test = []

for threshold in range(0,100):

  pred_0_1 = [1 if x >= threshold/100 else 0 for x in y_pred_proba_test]

  list_accuracy_test.append(mt.accuracy_score(y_test, pred_0_1))
xs = [x/100 for x in range(0,100)]

ys = list_accuracy_test

plt.plot(xs, ys)
best_scoring = max(list_accuracy_test)

best_threshold = list_accuracy_test.index(best_scoring)/100

print("El mejor threshold es: {}".format(best_threshold))
accuracy_train = mt.accuracy_score(y_train, [1 if x >= best_threshold else 0 for x in y_pred_proba_train])

accuracy_test = mt.accuracy_score(y_test, [1 if x >= best_threshold else 0 for x in y_pred_proba_test])



print("Accuracy - Train: {}".format(accuracy_train))

print("Accuracy - Test : {}".format(accuracy_test))
df_feature_importances = pd.DataFrame()

df_feature_importances['feature'] = X_train.columns

df_feature_importances['importance'] = model.feature_importances_/model.feature_importances_.sum()

df_feature_importances = df_feature_importances.sort_values(by = ['importance','feature'],ascending=False)

df_feature_importances.reset_index(drop = True,inplace=True)



df_feature_importances
df_feature_importances[['feature','importance']].sort_values(by=['importance'],

                                                             ascending = [True]).plot(kind='barh',

                                                             x='feature',

                                                             y='importance',

                                                             legend=True, 

                                                             figsize=(5, 5))
from sklearn.ensemble import RandomForestClassifier

RandomForestClassifier()
# Create  model objet 

model_rf = RandomForestClassifier(n_estimators = 150, random_state = 0, max_depth=5, 

                                  max_features = 0.5, min_samples_leaf = 10, 

                                  n_jobs = -1)



# Fit the model:

model_rf.fit(X_train, y_train)



model = model_rf
# Generar las predicciones:

y_pred_train = model.predict(X_train)

y_pred_test = model.predict(X_test)



# Generar las probabilidades

y_pred_proba_train = model.predict_proba(X_train)[:,1]

y_pred_proba_test = model.predict_proba(X_test)[:,1]
accuracy_train = mt.accuracy_score(y_train, y_pred_train)

accuracy_test = mt.accuracy_score(y_test, y_pred_test)



print("Accuracy - Train: {}".format(accuracy_train))

print("Accuracy - Test : {}".format(accuracy_test))
list_accuracy_test = []

for threshold in range(0,100):

  pred_0_1 = [1 if x >= threshold/100 else 0 for x in y_pred_proba_test]

  list_accuracy_test.append(mt.accuracy_score(y_test, pred_0_1))
xs = [x/100 for x in range(0,100)]

ys = list_accuracy_test

plt.plot(xs, ys)
best_scoring = max(list_accuracy_test)

best_threshold = list_accuracy_test.index(best_scoring)/100

print("El mejor threshold es: {}".format(best_threshold))
accuracy_train = mt.accuracy_score(y_train, [1 if x >= best_threshold else 0 for x in y_pred_proba_train])

accuracy_test = mt.accuracy_score(y_test, [1 if x >= best_threshold else 0 for x in y_pred_proba_test])



print("Accuracy - Train: {}".format(accuracy_train))

print("Accuracy - Test : {}".format(accuracy_test))
df_feature_importances = pd.DataFrame()

df_feature_importances['feature'] = X_train.columns

df_feature_importances['importance'] = model.feature_importances_/model.feature_importances_.sum()

df_feature_importances = df_feature_importances.sort_values(by = ['importance','feature'],ascending=False)

df_feature_importances.reset_index(drop = True,inplace=True)



df_feature_importances
df_feature_importances[['feature','importance']].sort_values(by=['importance'],

                                                             ascending = [True]).plot(kind='barh',

                                                             x='feature',

                                                             y='importance',

                                                             legend=True, 

                                                             figsize=(5, 5))
pred_prob_subm = model_rf.predict_proba(X_summit)[:,1]

pred_subm = [1 if x >= best_threshold else 0 for x in pred_prob_subm]
Y_summit_pred = pd.DataFrame()

Y_summit_pred[ID] = df_test[ID]

Y_summit_pred[TARGET] = pred_subm #pred_prob_subm

Y_summit_pred.head()
Y_summit_pred.to_csv("krfc_submission_01_baseline.csv", index = False)