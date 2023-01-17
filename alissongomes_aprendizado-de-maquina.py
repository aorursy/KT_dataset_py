import numpy as np

import pandas as pd

from time import time

from IPython.display import display

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.ensemble import RandomForestRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression



from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.preprocessing import StandardScaler



%matplotlib inline



import warnings

warnings.filterwarnings('ignore')
data = pd.read_csv("../input/compresive_strength_concrete.csv")

display(data)
#Renomeando as colunas

data.columns = ['cimento', 'escoria', 'cinzas', 'agua', 'superplastificante', 'ag_grosso', 'ag_fino', 'idade', 'resistencia']

display(data)
data.isnull().any()
data.info()
sns.distplot(data["resistencia"])
data.describe()
sns.set(font_scale=1.5)

sns.pairplot(data)
corr = data.corr()

plt.figure(figsize=(14, 14)) #deixando a imagem maior

sns.heatmap(corr,annot = True, xticklabels=corr.columns.values, yticklabels=corr.columns.values,cmap = "coolwarm")
#Colocando os dados em ordem aleatória 

randomdata = (data.sample(n=1030, replace=False))



#Treino e teste

X = data.drop('resistencia', axis = 1)

y = data['resistencia']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)



#Padronizando dados

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.fit_transform(X_test)
###Decision Tree

DTR = DecisionTreeRegressor()

DTR = DTR.fit(X_train, y_train)

pred_dtr = DTR.predict(X_test)



print('Arvores de Decisão:')

print('Erro Quadrático Médio:', mean_squared_error(y_test, pred_dtr))

print('Raiz do Erro Quadrático Médio:', np.sqrt(mean_squared_error(y_test, pred_dtr)))



errors = abs(pred_dtr - y_test)

aux = 100 * (errors / y_test)

dtr_eval = round(100 - np.mean(aux), 2)

print ("Acurácia:", dtr_eval, "%")



#Validação cruzada

dtr_cv = cross_val_score(estimator = DTR, X = X_train, y = y_train, cv = 10)

dtr_cv = round((dtr_cv.mean()*100), 2)

print("Acurácia usando validação cruzada:",dtr_cv, "%\n")





###Random Forest

RFR = RandomForestRegressor(n_estimators=200)

RFR = RFR.fit(X_train, y_train)

pred_rfr = RFR.predict(X_test)



print('Florestas aleatórias de decisão:')

print('Erro Quadrático Médio:', mean_squared_error(y_test, pred_rfr))

print('Raiz do Erro Quadrático Médio:', np.sqrt(mean_squared_error(y_test, pred_rfr)))



errors = abs(pred_rfr - y_test)

aux = 100 * (errors / y_test)

rfr_eval = round(100 - np.mean(aux), 2)

print("Acurácia:", rfr_eval, "%")



#Validação cruzada

rfr_cv = cross_val_score(estimator = RFR, X = X_train, y = y_train, cv = 10)

rfr_cv = round((rfr_cv.mean()*100), 2)

print("Acurácia usando validação cruzada:",rfr_cv, "%\n")





###Regressao Linear

LIR = LinearRegression()

LIR = LIR.fit(X_train, y_train)

pred_lir = LIR.predict(X_test)



print('Regressão Linear:')

print('Erro Quadrático Médio:', mean_squared_error(y_test, pred_lir))

print('Raiz do Erro Quadrático Médio:', np.sqrt(mean_squared_error(y_test, pred_lir)))



errors = abs(pred_lir - y_test)

aux = 100 * (errors / y_test)

lir_eval = round(100 - np.mean(aux), 2)

print("Acurácia:", lir_eval, "%")



#Validação cruzada

lir_cv = cross_val_score(estimator = LIR, X = X_train, y = y_train, cv = 10)

lir_cv = round((lir_cv.mean()*100), 2)

print("Acurácia usando validação cruzada:",lir_cv, "%\n")





###KNN

KNN = KNeighborsRegressor(n_neighbors=3)

KNN = KNN.fit(X_train, y_train)

pred_knn = KNN.predict(X_test)



print('K vizinhos mais próximos:')

print('Erro Quadrático Médio:', mean_squared_error(y_test, pred_knn))

print('Raiz do Erro Quadrático Médio:', np.sqrt(mean_squared_error(y_test, pred_knn)))



errors = abs(pred_knn - y_test)

aux = 100 * (errors / y_test)

knn_eval = round(100 - np.mean(aux),2)

print("Acurácia:", knn_eval, "%")



#Validação cruzada

knn_cv = cross_val_score(estimator = KNN, X = X_train, y = y_train, cv = 10)

knn_cv = round((knn_cv.mean()*100), 2)

print("Acurácia usando validação cruzada:",knn_cv, "%\n")



conc0 = {

    'Holdout':[dtr_eval,rfr_eval,lir_eval,knn_eval] ,

    'CrossValidation':[dtr_cv,rfr_cv,lir_cv,knn_cv]

}

concl0 = pd.DataFrame(conc0, columns=['Holdout', 'CrossValidation'], index=['DT', 'RF', 'LR', 'KNN'])