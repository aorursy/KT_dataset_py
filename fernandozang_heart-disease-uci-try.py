# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from sklearn.metrics import mean_absolute_error, precision_score

from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier



from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

print("Imported")
path = "../input/heart.csv"

dados = pd.read_csv(path)



dados.columns
#Transformando os dados de chest pain type em novos atributos

chest_pain_type_attr = pd.get_dummies(dados["cp"], prefix="chest_pain_type")



colunas = dados.columns.drop('cp')

dados = dados[colunas]



dados = dados.join(chest_pain_type_attr)



dados.columns

#Using "weka.attributeSelection.InfoGainAttributeEval" on "WEKA", de following attributes are irrelevant to classification

colunas = dados.columns.drop(['chol', 'fbs', 'trestbps', 'restecg'])



#dados = dados[colunas]

print(dados.columns)
thal_attr = pd.get_dummies(dados["thal"], prefix="thal")



colunas = dados.columns.drop('thal')

dados = dados[colunas]



dados = dados.join(thal_attr)



dados.columns
y = dados['target']



columns = dados.columns.drop('target')



X = dados[columns]



print("Classes: " + str(y.unique()) )

print("Columns: " + str(X.columns) )

#Split Dataset

X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=1, train_size=0.66)



print("Dataset Splited")
model = MLPClassifier(random_state=1, learning_rate_init=0.0003, max_iter=3000, activation='relu',

                      hidden_layer_sizes=(300))

print(model)

print("\n --> Model Created <--")

model.fit(X_train, y_train)

print("\n --> MODEL FITTED <--")





y_pred = model.predict(X_val)

print("\n --> y PREDICTED <--")



mae = mean_absolute_error(y_pred, y_val)

print("MAE: " + str(mae))

print("Score: " + str(precision_score(y_val, y_pred)) )
model = KNeighborsClassifier(n_neighbors=23, weights='distance', p=1 )



print(model)

print("\n --> Model Created <--")

model.fit(X_train, y_train)

print("\n --> MODEL FITTED <--")



y_pred = model.predict(X_val)

print("\n --> y PREDICTED <--")



mae = mean_absolute_error(y_pred, y_val)

print("MAE: " + str(mae))

print("Score: " + str(precision_score(y_val, y_pred)) )
model = DecisionTreeClassifier(criterion='gini')



print(model)

print("\n --> Model Created <--")

model.fit(X_train, y_train)

print("\n --> MODEL FITTED <--")



y_pred = model.predict(X_val)

print("\n --> y PREDICTED <--")



mae = mean_absolute_error(y_pred, y_val)

print("MAE: " + str(mae))

print("Score: " + str(precision_score(y_val, y_pred)) )
from sklearn import svm



model = svm.SVC(kernel="linear")



print(model)

print("\n --> Model Created <--")

model.fit(X_train, y_train)

print("\n --> MODEL FITTED <--")



y_pred = model.predict(X_val)

print("\n --> y PREDICTED <--")



mae = mean_absolute_error(y_pred, y_val)

print("MAE: " + str(mae))

print("Score: " + str(precision_score(y_val, y_pred)) )