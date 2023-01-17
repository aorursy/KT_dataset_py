# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("/kaggle/input/bcu-ratings-20/train.csv/train.csv")

test = pd.read_csv("/kaggle/input/bcu-ratings-20/test.csv/test.csv")



train.head()
features = list(set(train.columns.values) - set(['ID', 'TARGET']))

cat_features = [feat for feat in features if train[feat].dtype.name in ['category', 'object']]

num_features = list(set(features) - set(cat_features))



print(f"Categorical features: \n {cat_features}")

print(f"Numerical features: \n {num_features}")
for feat in num_features:

    # cuenta si hay missing en el train o en el test

    if (np.sum(train[feat].isna() * 1) + np.sum(test[feat].isna() * 1)) > 0:

        # calcula la mediana

        mediana = train.loc[np.logical_not(train[feat].isna()), feat].median()

        

        # rellena los missing tanto del train como del test con la mediana

        train.loc[train[feat].isna(), feat] = mediana

        test.loc[test[feat].isna(), feat] = mediana
# relleno de missing

for feat in cat_features:

    train.loc[train[feat].isna(), feat] = "NA"

    test.loc[test[feat].isna(), feat] = "NA"
# one-hot encoding de categóricas

test['TARGET'] = np.nan

data = pd.concat([train, test], axis = 0)



data = pd.get_dummies(data, drop_first = True, columns = cat_features) # el parámetro drop first elimina uno de los valores, tal y como queremos que suceda



# volvemos a calcular los features, ya que el nombre de algunas columnas ha cambiado

features = list(set(data.columns.values) - set(['ID', 'TARGET']))



# separamos otra vez el train del test (el test serán aquellas observaciones para las que el target es NA)

train = data.loc[np.logical_not(data['TARGET'].isna())]

test = data.loc[data['TARGET'].isna()]



train.head()
# partimos la muestra en train y test

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(train[features], train['TARGET'], test_size = 0.25,

                                                  stratify = train['TARGET'], random_state = 1234)



# escalamos los features

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(X_train)

X_train = pd.DataFrame(scaler.transform(X_train), columns = X_train.columns.values)

X_val = pd.DataFrame(scaler.transform(X_val), columns = X_val.columns.values)



# entrenamos el modelo y hacemos las prediciones

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score



model = LogisticRegression(penalty = 'elasticnet', l1_ratio = 0, C = 1, solver = 'saga', max_iter = 10000)

model.fit(X_train, y_train)

preds = [i[1] for i in model.predict_proba(X_val)]

print(f"Roc auc score del modelo: {roc_auc_score(y_val, preds)}")
from sklearn.model_selection import KFold



def cross_validation_roc_auc(modelo, X, y, number_folds):

    # lista auxiliar para guardar el resultado de cada fold

    metrics = []

    

    # reset index de X e y para poder hacer el split (necesario para hacer que los indices

    # de X e y sean iguales)

    X = X.reset_index(drop = True); y = y.reset_index(drop = True)

    

    # utilizando el objeto KFold de sklearn, creamos los splits, y para cada split

    # entrenamos un modelo y calculamos su métrica en el fold de validación

    for train_index, val_index in KFold(n_splits = number_folds).split(X):

        modelo.fit(X.loc[train_index], y[train_index])

        val_pred = [i[1] for i in modelo.predict_proba(X.loc[val_index])]

        metrics.append(roc_auc_score(y[val_index], val_pred))

    

    # la función devuelve la media y la d. estándar de los valores de CV score

    return np.round(np.mean(metrics), 3), np.round(np.std(metrics), 4)
C_parametros = [0.01, 0.1, 1, 10, 100, 1000]



for c in C_parametros:

    model = LogisticRegression(penalty = 'elasticnet', l1_ratio = 0, C = c, solver = 'saga', max_iter = 10000)

    mean_cv, std_cv = cross_validation_roc_auc(model, X_train, y_train, number_folds = 5)

    print(f"C = {c} | Resultado = {mean_cv} +- {std_cv}")
model = LogisticRegression(penalty = 'elasticnet', l1_ratio = 0, C = 1000, solver = 'saga', max_iter = 10000)

model.fit(X_train, y_train)

preds = [i[1] for i in model.predict_proba(X_val)]

print(f"Roc auc score del modelo: {roc_auc_score(y_val, preds)}")
def score_function(c):

    

    # esto no es necesario en este caso, pero en caso de utilizar otro modelo, en el que alguno de los parámetros

    # solamente pudiera tomar valores enteros, deberíamos asegurar que el parámetro es un entero

    c = int(c)

    

    # definimos el modelo con el parámetro c que hemos pasado en la función

    model = LogisticRegression(penalty = 'elasticnet', l1_ratio = 0, C = c, solver = 'saga', max_iter = 10000)

    

    # utilizamos la función auxiliar anterior para calcular el cross-validation ROC AUC score anterior

    score, _ = cross_validation_roc_auc(model, X_train, y_train, number_folds = 5)

    return score

    
from bayes_opt import BayesianOptimization



bounds = {'c': (1, 1000)}



optimizer = BayesianOptimization(f = score_function, pbounds = bounds, random_state = 1234)

optimizer.maximize(init_points = 10, n_iter = 20)
model = LogisticRegression(penalty = 'elasticnet', l1_ratio = 0, C = 1, solver = 'saga', max_iter = 10000)

model.fit(X_train, y_train)



# escalado y predicción

test_scaled = pd.DataFrame(scaler.transform(test[features]), columns = test[features].columns.values)

test_pred = model.predict_proba(test_scaled)

test_pred = [i[1] for i in test_pred]



submission = pd.DataFrame({'ID': test['ID'], 'Pred': test_pred}).to_csv("reg_logistic_regression.csv", index = False)