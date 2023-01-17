# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.options.display.max_columns = None



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("/kaggle/input/bcu-ratings/train.csv")

test = pd.read_csv("/kaggle/input/bcu-ratings/test.csv")



train.head()
features = list(set(train.columns.values) - set(['ID', 'TARGET']))

cat_features = [feat for feat in features if train[feat].dtype.name in ['category', 'object']]

num_features = list(set(features) - set(cat_features))



print(f"Categorical features: \n {cat_features}")

print(f"Numerical features: \n {num_features}")
# Función para contar el número de missing de un feature del train y del test

def count_missing(feature):

    num_missing_train = np.sum(train[feature].isna())

    num_missing_test = np.sum(test[feature].isna())

    

    # solo mostramos mensaje si el feature tiene missing

    if (num_missing_train + num_missing_test) > 0:

        print(f"Número de missing del feature {feature}: train = {num_missing_train} | test = {num_missing_test}")

        

    return (num_missing_train + num_missing_test)

    

# contamos el número de missing del train y del test para cada feature

feat_missing = []

for feat in features:

    num_miss = count_missing(feat)

    if num_miss > 0:

        feat_missing.append(feat)
for feat in [i for i in feat_missing if i in num_features]:

    # calculamos la mediana solamente con los valores del train, e imputamos tanto el train como el test con el valor mediano

    mediana = train[feat].median()

    train.loc[train[feat].isna(), feat] = mediana

    test.loc[test[feat].isna(), feat] = mediana
for feat in [i for i in feat_missing if i in cat_features]:

    train.loc[train[feat].isna(), feat] = "NA"

    test.loc[test[feat].isna(), feat] = "NA"
# contamos el número de missing del train y del test para cada feature

for feat in features:

    count_missing(feat)
test['TARGET'] = np.nan

data = pd.concat([train, test], axis = 0)
data = pd.get_dummies(data, drop_first = True, columns = cat_features) # el parámetro drop first elimina uno de los valores, tal y como queremos que suceda



# volvemos a calcular los features, ya que el nombre de algunas columnas ha cambiado

features = list(set(data.columns.values) - set(['ID', 'TARGET']))



# separamos otra vez el train del test (el test serán aquellas observaciones para las que el target es NA)

train = data.loc[np.logical_not(data['TARGET'].isna())]

test = data.loc[data['TARGET'].isna()]



train.head()
from sklearn.model_selection import train_test_split



X_train, X_val, y_train, y_val = train_test_split(train[features], train['TARGET'], test_size = 0.25,

                                                  stratify = train['TARGET'], random_state = 1234)
from sklearn.linear_model import LogisticRegression



log_regressor = LogisticRegression(max_iter = 1e5) # en realidad la logistic regression de sklearn es regularizada, 

                                                   # pero de momento esto no es importante para el ejercicio

log_regressor.fit(X = X_train, y = y_train)
from sklearn.metrics import roc_auc_score



y_pred_val = log_regressor.predict_proba(X_val)

y_pred_val = [i[1] for i in y_pred_val]



score = roc_auc_score(y_val, y_pred_val)



print(f"ROC AUC score of the model: {np.round(score, 5)}")
test_pred = log_regressor.predict_proba(test[features])

test_pred = [i[1] for i in test_pred]



submission = pd.DataFrame({'ID': test['ID'], 'Pred': test_pred}).to_csv("logistic_regression.csv", index = False)