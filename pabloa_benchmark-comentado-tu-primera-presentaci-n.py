import numpy as np # Biblioteca de funciones matemáticas de alto nivel para operar con esos vectores o matrices

import pandas as pd # Manipulación y análisis de datos, Data Frames, lectura de CSV 

from lightgbm import LGBMClassifier # Implementación de un algoritmo de Boosting de Arboles de Decisión con descenso de gradiente

from sklearn import model_selection # Separación en Train y Test

from sklearn.metrics import roc_auc_score # Calculo del Area bajo la Curva ROC

# Leemos el archivo "pageviews" y observamos las primeras filas

data = pd.read_csv("../input/pageviews/pageviews.csv",

                   parse_dates=["FEC_EVENT"]) # Le indicamos que la columna FEC_EVENT debe leerse como tipo Fecha

data.head()
X_test = [] # Primero creamos el objeto vacío

for c in data.drop(["USER_ID", "FEC_EVENT"], axis=1).columns: # iteramos sobre todas las columnas de "data", menos la fecha y el Id de Usuario

    print("haciendo", c) # Mostramos en que variable está trabajando el loop

    temp = pd.crosstab(data.USER_ID, data[c]) # * Realizamos una tabla cruzada de la Variable por Usuario colocando la frecuencia de cada valor posible como columna 

    temp.columns = [c + "_" + str(v) for v in temp.columns] # El nombre de cada columna lo renombramos como: Variable + "_" + Valor de la Variable

    X_test.append(temp.apply(lambda x: x / x.sum(), axis=1)) # Aplicamos una función lambda para calcular la proporción de frecuencia de cada variable

X_test = pd.concat(X_test, axis=1) # Concatenamos todas las variables en el mismo objeto
# SOLO A MODO DE EXPLICACIÖN DEL CODIGO ANTERIOR

# * Podemos ver un ejemplo con PAGE de como se ve la ejecución solo de esta linea, armando la tabla de contingencia y graficando las primeras 5 filas

pd.crosstab(data.USER_ID, data["PAGE"]).head()
X_test.head()
X_test.shape
X_test.iloc[0,0:1725].sum()
data = data[data.FEC_EVENT.dt.month < 10] # Limitamos los registros a eventos anteriores a Octubre (mes 10)

X_train = [] # Creo un objeto vacío para Train

for c in data.drop(["USER_ID", "FEC_EVENT"], axis=1).columns: # Repito el proceso que vimos anteriormente

    print("haciendo", c)

    temp = pd.crosstab(data.USER_ID, data[c])

    temp.columns = [c + "_" + str(v) for v in temp.columns]

    X_train.append(temp.apply(lambda x: x / x.sum(), axis=1))

X_train = pd.concat(X_train, axis=1)
features = list(set(X_train.columns).intersection(set(X_test.columns))) # Creamos una lista con las columnas que se encuentran en ambos datasets

X_train = X_train[features] # Filtramos en el dataset de Train las columnas que son comunes a ambos

X_test = X_test[features] # Filtramos en el dataset de Test las columnas que son comunes a ambos
X_train.head()
y_prev = pd.read_csv("../input/conversiones/conversiones.csv") # Leemos el archivo CSV

y_train = pd.Series(0, index=X_train.index) # Creamos un objeto para cada Usuario con valor cero en todos los casos

idx = set(y_prev[y_prev.mes >= 10].USER_ID.unique()).intersection(

        set(X_train.index)) # Buscamos a los Usuarios que hayan convertido de Octubre en adelante

y_train.loc[list(idx)] = 1 # Asignamos el valor "1" a los casos que crucen con el objeto creado antes
y_train.head(23)
# Entrenamos el modelo LGBM Classifier



fi = [] # Creamos un objeto vacío para guardar la importancia de las variables de los modelos entrenados

test_probs = [] # Creamos un objeto vacío para guardar las probabilidades estimadas

i = 0 

for train_idx, valid_idx in model_selection.KFold(n_splits=10, shuffle=True).split(X_train): # Iterams sobre 10 folds que creamos para entrenar y validar

    i += 1

    Xt = X_train.iloc[train_idx] # Definimos el set de entrenamiento a partir de los indices que coincidan con los 9 folds de entrenamiento

    yt = y_train.loc[X_train.index].iloc[train_idx] # Definimos el Target de entrenamiento con los Usuarios que coincidan con los que se encuentran en estos folds



    Xv = X_train.iloc[valid_idx] # Definimos el set de validación a partir los indices que coincidan con el fold de validación

    yv = y_train.loc[X_train.index].iloc[valid_idx] # Definimos el Target de validación con los Usuarios que coincidan con los que se encuentran en este fold



    learner = LGBMClassifier(n_estimators=10000) # Entrenamos un modelo lightgbm con los parámetros por default usando un máximo de 10.000 árboles

    learner.fit(Xt, yt,  early_stopping_rounds=10, eval_metric="auc",

                eval_set=[(Xt, yt), (Xv, yv)]) # Definimos un early stop de 10 y el método de evaluación como AUC

    

    test_probs.append(pd.Series(learner.predict_proba(X_test)[:, -1],

                                index=X_test.index, name="fold_" + str(i))) # Predecimos sobre la base total y nos guardamos las probabilidades estimadas

    fi.append(pd.Series(learner.feature_importances_ / learner.feature_importances_.sum(), index=Xt.columns)) # Guardamos la proporción de importancia de cada variable en el modelo



test_probs = pd.concat(test_probs, axis=1).mean(axis=1) # Caluclamos la media de las probabilidades

test_probs.index.name="USER_ID" # Renombramos las columnas

test_probs.name="SCORE" # Renombramos las columnas

test_probs.to_csv("benchmark.zip", header=True, compression="zip") # Guardamos la predicción final en un zip para subirlo a la plataforma

fi = pd.concat(fi, axis=1).mean(axis=1) # Como explicación del modelo guardamos la importancia de las variables de los modelos entrenados
test_probs