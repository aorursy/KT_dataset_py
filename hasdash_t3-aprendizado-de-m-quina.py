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

from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter(action="ignore", category=ConvergenceWarning)
data = pd.read_csv("../input/compresive_strength_concrete.csv")

display(data)
#Renomeando as colunas

data.columns = ['cimento', 'escoria', 'cinzas', 'agua', 'superplastificante', 'ag_grosso', 'ag_fino', 'idade', 'resistencia']
data.isnull().any()
data.info()
sns.distplot(data["resistencia"])
sns.set(font_scale=1.5)

sns.pairplot(data)
#Colocando os dados em ordem aleatória 

randomdata = (data.sample(n=1030, replace=False, random_state = 42))



#Treino e teste

X = randomdata.drop('resistencia', axis = 1)

y = randomdata['resistencia']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)



#Padronizando dados

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.fit_transform(X_test)
model = RandomForestRegressor(max_depth=None, random_state=None)

model = model.fit(X_train, y_train)



importances = model.feature_importances_



z = np.column_stack((X.keys(), importances))

z = sorted(z,key=lambda x: x[1])

display(z)
corr = X.corr()

plt.figure(figsize=(14, 14)) #deixando a imagem maior

sns.heatmap(corr,annot = True, xticklabels=corr.columns.values, yticklabels=corr.columns.values,cmap = "coolwarm")
#Treino e teste

X = randomdata.drop('resistencia', axis = 1)

y = randomdata['resistencia']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)



#Padronizando dados

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.fit_transform(X_test)
###Decision Tree

DTR = DecisionTreeRegressor(random_state = 42)

DTR = DTR.fit(X_train, y_train)

pred_dtr = DTR.predict(X_test)

print('Arvores de Decisão:')

print('Erro Quadrático Médio:', mean_squared_error(y_test, pred_dtr))

print('Raiz do Erro Quadrático Médio:', np.sqrt(mean_squared_error(y_test, pred_dtr)))

dtr0 = np.sqrt(mean_squared_error(y_test, pred_dtr))

#Validação cruzada

dtr_cv = cross_val_score(estimator = DTR, X = X_train, y = y_train, cv = 10, scoring='neg_mean_squared_error')

print("Erro Quadrático Médio usando validação cruzada:",(dtr_cv.mean()*(-1)))

print('Raiz do Erro Quadrático Médio usando validação cruzada:', np.sqrt((dtr_cv.mean()*(-1))), "\n")

dtr1 = np.sqrt((dtr_cv.mean()*(-1)))



###Random Forest

RFR = RandomForestRegressor(n_estimators=200, random_state = 42)

RFR = RFR.fit(X_train, y_train)

pred_rfr = RFR.predict(X_test)

print('Florestas aleatórias de decisão:')

print('Erro Quadrático Médio:', mean_squared_error(y_test, pred_rfr))

print('Raiz do Erro Quadrático Médio:', np.sqrt(mean_squared_error(y_test, pred_rfr)))

rfr0 = np.sqrt(mean_squared_error(y_test, pred_rfr))

#Validação cruzada

rfr_cv = cross_val_score(estimator = RFR, X = X_train, y = y_train, cv = 10, scoring='neg_mean_squared_error')

print("Erro Quadrático Médio usando validação cruzada:",(rfr_cv.mean()*(-1)))

print('Raiz do Erro Quadrático Médio usando validação cruzada:', np.sqrt((rfr_cv.mean()*(-1))), "\n")

rfr1 = np.sqrt((rfr_cv.mean()*(-1)))



###Regressao Linear

LIR = LinearRegression()

LIR = LIR.fit(X_train, y_train)

pred_lir = LIR.predict(X_test)

print('Regressão Linear:')

print('Erro Quadrático Médio:', mean_squared_error(y_test, pred_lir))

print('Raiz do Erro Quadrático Médio:', np.sqrt(mean_squared_error(y_test, pred_lir)))

lir0 = np.sqrt(mean_squared_error(y_test, pred_lir))

#Validação cruzada

lir_cv = cross_val_score(estimator = LIR, X = X_train, y = y_train, cv = 10, scoring='neg_mean_squared_error')

print("Erro Quadrático Médio usando validação cruzada:",(lir_cv.mean()*(-1)))

print('Raiz do Erro Quadrático Médio usando validação cruzada:', np.sqrt((lir_cv.mean()*(-1))), "\n")

lir1 = np.sqrt((lir_cv.mean()*(-1)))



###KNN

KNN = KNeighborsRegressor(n_neighbors=3)

KNN = KNN.fit(X_train, y_train)

pred_knn = KNN.predict(X_test)

print('K vizinhos mais próximos:')

print('Erro Quadrático Médio:', mean_squared_error(y_test, pred_knn))

print('Raiz do Erro Quadrático Médio:', np.sqrt(mean_squared_error(y_test, pred_knn)))

knn0 = np.sqrt(mean_squared_error(y_test, pred_knn))

#Validação cruzada

knn_cv = cross_val_score(estimator = KNN, X = X_train, y = y_train, cv = 10, scoring='neg_mean_squared_error')

print("Erro Quadrático Médio usando validação cruzada:",(knn_cv.mean()*(-1)))

print('Raiz do Erro Quadrático Médio usando validação cruzada:', np.sqrt((knn_cv.mean()*(-1))), "\n")

knn1 = np.sqrt((knn_cv.mean()*(-1)))



conc0 = {

    'Holdout':[dtr0,rfr0,lir0,knn0] ,

    'CrossValidation':[dtr1,rfr1,lir1,knn1]

}

concl0 = pd.DataFrame(conc0, columns=['Holdout', 'CrossValidation'], index=['DT', 'RF', 'LR', 'KNN'])
#Treino e teste

X = randomdata.drop(['resistencia','cinzas','ag_grosso','ag_fino','superplastificante'], axis = 1)

y = randomdata['resistencia']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)



#Padronizando dados

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.fit_transform(X_test)
###Decision Tree

DTR = DecisionTreeRegressor(random_state = 42)

DTR = DTR.fit(X_train, y_train)

pred_dtr = DTR.predict(X_test)

print('Arvores de Decisão:')

print('Erro Quadrático Médio:', mean_squared_error(y_test, pred_dtr))

print('Raiz do Erro Quadrático Médio:', np.sqrt(mean_squared_error(y_test, pred_dtr)))

dtr0 = np.sqrt(mean_squared_error(y_test, pred_dtr))

#Validação cruzada

dtr_cv = cross_val_score(estimator = DTR, X = X_train, y = y_train, cv = 10, scoring='neg_mean_squared_error')

print("Erro Quadrático Médio usando validação cruzada:",(dtr_cv.mean()*(-1)))

print('Raiz do Erro Quadrático Médio usando validação cruzada:', np.sqrt((dtr_cv.mean()*(-1))), "\n")

dtr1 = np.sqrt((dtr_cv.mean()*(-1)))



###Random Forest

RFR = RandomForestRegressor(n_estimators=200, random_state = 42)

RFR = RFR.fit(X_train, y_train)

pred_rfr = RFR.predict(X_test)

print('Florestas aleatórias de decisão:')

print('Erro Quadrático Médio:', mean_squared_error(y_test, pred_rfr))

print('Raiz do Erro Quadrático Médio:', np.sqrt(mean_squared_error(y_test, pred_rfr)))

rfr0 = np.sqrt(mean_squared_error(y_test, pred_rfr))

#Validação cruzada

rfr_cv = cross_val_score(estimator = RFR, X = X_train, y = y_train, cv = 10, scoring='neg_mean_squared_error')

print("Erro Quadrático Médio usando validação cruzada:",(rfr_cv.mean()*(-1)))

print('Raiz do Erro Quadrático Médio usando validação cruzada:', np.sqrt((rfr_cv.mean()*(-1))), "\n")

rfr1 = np.sqrt((rfr_cv.mean()*(-1)))



###Regressao Linear

LIR = LinearRegression()

LIR = LIR.fit(X_train, y_train)

pred_lir = LIR.predict(X_test)

print('Regressão Linear:')

print('Erro Quadrático Médio:', mean_squared_error(y_test, pred_lir))

print('Raiz do Erro Quadrático Médio:', np.sqrt(mean_squared_error(y_test, pred_lir)))

lir0 = np.sqrt(mean_squared_error(y_test, pred_lir))

#Validação cruzada

lir_cv = cross_val_score(estimator = LIR, X = X_train, y = y_train, cv = 10, scoring='neg_mean_squared_error')

print("Erro Quadrático Médio usando validação cruzada:",(lir_cv.mean()*(-1)))

print('Raiz do Erro Quadrático Médio usando validação cruzada:', np.sqrt((lir_cv.mean()*(-1))), "\n")

lir1 = np.sqrt((lir_cv.mean()*(-1)))



###KNN

KNN = KNeighborsRegressor(n_neighbors=3)

KNN = KNN.fit(X_train, y_train)

pred_knn = KNN.predict(X_test)

print('K vizinhos mais próximos:')

print('Erro Quadrático Médio:', mean_squared_error(y_test, pred_knn))

print('Raiz do Erro Quadrático Médio:', np.sqrt(mean_squared_error(y_test, pred_knn)))

knn0 = np.sqrt(mean_squared_error(y_test, pred_knn))

#Validação cruzada

knn_cv = cross_val_score(estimator = KNN, X = X_train, y = y_train, cv = 10, scoring='neg_mean_squared_error')

print("Erro Quadrático Médio usando validação cruzada:",(knn_cv.mean()*(-1)))

print('Raiz do Erro Quadrático Médio usando validação cruzada:', np.sqrt((knn_cv.mean()*(-1))), "\n")

knn1 = np.sqrt((knn_cv.mean()*(-1)))



conc4 = {

    'Holdout':[dtr0,rfr0,lir0,knn0] ,

    'CrossValidation':[dtr1,rfr1,lir1,knn1]

}

concl4 = pd.DataFrame(conc4, columns=['Holdout', 'CrossValidation'], index=['DT', 'RF', 'LR', 'KNN'])
#Treino e teste

X = randomdata.drop(['resistencia','cinzas','ag_grosso','ag_fino'], axis = 1)

y = randomdata['resistencia']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)



#Padronizando dados

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.fit_transform(X_test)
###Decision Tree

DTR = DecisionTreeRegressor(random_state = 42)

DTR = DTR.fit(X_train, y_train)

pred_dtr = DTR.predict(X_test)

print('Arvores de Decisão:')

print('Erro Quadrático Médio:', mean_squared_error(y_test, pred_dtr))

print('Raiz do Erro Quadrático Médio:', np.sqrt(mean_squared_error(y_test, pred_dtr)))

dtr0 = np.sqrt(mean_squared_error(y_test, pred_dtr))

#Validação cruzada

dtr_cv = cross_val_score(estimator = DTR, X = X_train, y = y_train, cv = 10, scoring='neg_mean_squared_error')

print("Erro Quadrático Médio usando validação cruzada:",(dtr_cv.mean()*(-1)))

print('Raiz do Erro Quadrático Médio usando validação cruzada:', np.sqrt((dtr_cv.mean()*(-1))), "\n")

dtr1 = np.sqrt((dtr_cv.mean()*(-1)))



###Random Forest

RFR = RandomForestRegressor(n_estimators=200, random_state = 42)

RFR = RFR.fit(X_train, y_train)

pred_rfr = RFR.predict(X_test)

print('Florestas aleatórias de decisão:')

print('Erro Quadrático Médio:', mean_squared_error(y_test, pred_rfr))

print('Raiz do Erro Quadrático Médio:', np.sqrt(mean_squared_error(y_test, pred_rfr)))

rfr0 = np.sqrt(mean_squared_error(y_test, pred_rfr))

#Validação cruzada

rfr_cv = cross_val_score(estimator = RFR, X = X_train, y = y_train, cv = 10, scoring='neg_mean_squared_error')

print("Erro Quadrático Médio usando validação cruzada:",(rfr_cv.mean()*(-1)))

print('Raiz do Erro Quadrático Médio usando validação cruzada:', np.sqrt((rfr_cv.mean()*(-1))), "\n")

rfr1 = np.sqrt((rfr_cv.mean()*(-1)))



###Regressao Linear

LIR = LinearRegression()

LIR = LIR.fit(X_train, y_train)

pred_lir = LIR.predict(X_test)

print('Regressão Linear:')

print('Erro Quadrático Médio:', mean_squared_error(y_test, pred_lir))

print('Raiz do Erro Quadrático Médio:', np.sqrt(mean_squared_error(y_test, pred_lir)))

lir0 = np.sqrt(mean_squared_error(y_test, pred_lir))

#Validação cruzada

lir_cv = cross_val_score(estimator = LIR, X = X_train, y = y_train, cv = 10, scoring='neg_mean_squared_error')

print("Erro Quadrático Médio usando validação cruzada:",(lir_cv.mean()*(-1)))

print('Raiz do Erro Quadrático Médio usando validação cruzada:', np.sqrt((lir_cv.mean()*(-1))), "\n")

lir1 = np.sqrt((lir_cv.mean()*(-1)))



###KNN

KNN = KNeighborsRegressor(n_neighbors=3)

KNN = KNN.fit(X_train, y_train)

pred_knn = KNN.predict(X_test)

print('K vizinhos mais próximos:')

print('Erro Quadrático Médio:', mean_squared_error(y_test, pred_knn))

print('Raiz do Erro Quadrático Médio:', np.sqrt(mean_squared_error(y_test, pred_knn)))

knn0 = np.sqrt(mean_squared_error(y_test, pred_knn))

#Validação cruzada

knn_cv = cross_val_score(estimator = KNN, X = X_train, y = y_train, cv = 10, scoring='neg_mean_squared_error')

print("Erro Quadrático Médio usando validação cruzada:",(knn_cv.mean()*(-1)))

print('Raiz do Erro Quadrático Médio usando validação cruzada:', np.sqrt((knn_cv.mean()*(-1))), "\n")

knn1 = np.sqrt((knn_cv.mean()*(-1)))



conc3 = {

    'Holdout':[dtr0,rfr0,lir0,knn0] ,

    'CrossValidation':[dtr1,rfr1,lir1,knn1]

}

concl3 = pd.DataFrame(conc3, columns=['Holdout', 'CrossValidation'], index=['DT', 'RF', 'LR', 'KNN'])
#Treino e teste

X = randomdata.drop(['resistencia','cinzas','ag_grosso'], axis = 1)

y = randomdata['resistencia']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)



#Padronizando dados

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.fit_transform(X_test)
###Decision Tree

DTR = DecisionTreeRegressor(random_state = 42)

DTR = DTR.fit(X_train, y_train)

pred_dtr = DTR.predict(X_test)

print('Arvores de Decisão:')

print('Erro Quadrático Médio:', mean_squared_error(y_test, pred_dtr))

print('Raiz do Erro Quadrático Médio:', np.sqrt(mean_squared_error(y_test, pred_dtr)))

dtr0 = np.sqrt(mean_squared_error(y_test, pred_dtr))

#Validação cruzada

dtr_cv = cross_val_score(estimator = DTR, X = X_train, y = y_train, cv = 10, scoring='neg_mean_squared_error')

print("Erro Quadrático Médio usando validação cruzada:",(dtr_cv.mean()*(-1)))

print('Raiz do Erro Quadrático Médio usando validação cruzada:', np.sqrt((dtr_cv.mean()*(-1))), "\n")

dtr1 = np.sqrt((dtr_cv.mean()*(-1)))



###Random Forest

RFR = RandomForestRegressor(n_estimators=200, random_state = 42)

RFR = RFR.fit(X_train, y_train)

pred_rfr = RFR.predict(X_test)

print('Florestas aleatórias de decisão:')

print('Erro Quadrático Médio:', mean_squared_error(y_test, pred_rfr))

print('Raiz do Erro Quadrático Médio:', np.sqrt(mean_squared_error(y_test, pred_rfr)))

rfr0 = np.sqrt(mean_squared_error(y_test, pred_rfr))

#Validação cruzada

rfr_cv = cross_val_score(estimator = RFR, X = X_train, y = y_train, cv = 10, scoring='neg_mean_squared_error')

print("Erro Quadrático Médio usando validação cruzada:",(rfr_cv.mean()*(-1)))

print('Raiz do Erro Quadrático Médio usando validação cruzada:', np.sqrt((rfr_cv.mean()*(-1))), "\n")

rfr1 = np.sqrt((rfr_cv.mean()*(-1)))



###Regressao Linear

LIR = LinearRegression()

LIR = LIR.fit(X_train, y_train)

pred_lir = LIR.predict(X_test)

print('Regressão Linear:')

print('Erro Quadrático Médio:', mean_squared_error(y_test, pred_lir))

print('Raiz do Erro Quadrático Médio:', np.sqrt(mean_squared_error(y_test, pred_lir)))

lir0 = np.sqrt(mean_squared_error(y_test, pred_lir))

#Validação cruzada

lir_cv = cross_val_score(estimator = LIR, X = X_train, y = y_train, cv = 10, scoring='neg_mean_squared_error')

print("Erro Quadrático Médio usando validação cruzada:",(lir_cv.mean()*(-1)))

print('Raiz do Erro Quadrático Médio usando validação cruzada:', np.sqrt((lir_cv.mean()*(-1))), "\n")

lir1 = np.sqrt((lir_cv.mean()*(-1)))



###KNN

KNN = KNeighborsRegressor(n_neighbors=3)

KNN = KNN.fit(X_train, y_train)

pred_knn = KNN.predict(X_test)

print('K vizinhos mais próximos:')

print('Erro Quadrático Médio:', mean_squared_error(y_test, pred_knn))

print('Raiz do Erro Quadrático Médio:', np.sqrt(mean_squared_error(y_test, pred_knn)))

knn0 = np.sqrt(mean_squared_error(y_test, pred_knn))

#Validação cruzada

knn_cv = cross_val_score(estimator = KNN, X = X_train, y = y_train, cv = 10, scoring='neg_mean_squared_error')

print("Erro Quadrático Médio usando validação cruzada:",(knn_cv.mean()*(-1)))

print('Raiz do Erro Quadrático Médio usando validação cruzada:', np.sqrt((knn_cv.mean()*(-1))), "\n")

knn1 = np.sqrt((knn_cv.mean()*(-1)))



conc2 = {

    'Holdout':[dtr0,rfr0,lir0,knn0] ,

    'CrossValidation':[dtr1,rfr1,lir1,knn1]

}

concl2 = pd.DataFrame(conc2, columns=['Holdout', 'CrossValidation'], index=['DT', 'RF', 'LR', 'KNN'])
#Treino e teste

X = data.drop(['resistencia','cinzas'], axis = 1)

y = data['resistencia']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)



#Padronizando dados

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.fit_transform(X_test)
###Decision Tree

DTR = DecisionTreeRegressor(random_state = 42)

DTR = DTR.fit(X_train, y_train)

pred_dtr = DTR.predict(X_test)

print('Arvores de Decisão:')

print('Erro Quadrático Médio:', mean_squared_error(y_test, pred_dtr))

print('Raiz do Erro Quadrático Médio:', np.sqrt(mean_squared_error(y_test, pred_dtr)))

dtr0 = np.sqrt(mean_squared_error(y_test, pred_dtr))

#Validação cruzada

dtr_cv = cross_val_score(estimator = DTR, X = X_train, y = y_train, cv = 10, scoring='neg_mean_squared_error')

print("Erro Quadrático Médio usando validação cruzada:",(dtr_cv.mean()*(-1)))

print('Raiz do Erro Quadrático Médio usando validação cruzada:', np.sqrt((dtr_cv.mean()*(-1))), "\n")

dtr1 = np.sqrt((dtr_cv.mean()*(-1)))



###Random Forest

RFR = RandomForestRegressor(n_estimators=200, random_state = 42)

RFR = RFR.fit(X_train, y_train)

pred_rfr = RFR.predict(X_test)

print('Florestas aleatórias de decisão:')

print('Erro Quadrático Médio:', mean_squared_error(y_test, pred_rfr))

print('Raiz do Erro Quadrático Médio:', np.sqrt(mean_squared_error(y_test, pred_rfr)))

rfr0 = np.sqrt(mean_squared_error(y_test, pred_rfr))

#Validação cruzada

rfr_cv = cross_val_score(estimator = RFR, X = X_train, y = y_train, cv = 10, scoring='neg_mean_squared_error')

print("Erro Quadrático Médio usando validação cruzada:",(rfr_cv.mean()*(-1)))

print('Raiz do Erro Quadrático Médio usando validação cruzada:', np.sqrt((rfr_cv.mean()*(-1))), "\n")

rfr1 = np.sqrt((rfr_cv.mean()*(-1)))



###Regressao Linear

LIR = LinearRegression()

LIR = LIR.fit(X_train, y_train)

pred_lir = LIR.predict(X_test)

print('Regressão Linear:')

print('Erro Quadrático Médio:', mean_squared_error(y_test, pred_lir))

print('Raiz do Erro Quadrático Médio:', np.sqrt(mean_squared_error(y_test, pred_lir)))

lir0 = np.sqrt(mean_squared_error(y_test, pred_lir))

#Validação cruzada

lir_cv = cross_val_score(estimator = LIR, X = X_train, y = y_train, cv = 10, scoring='neg_mean_squared_error')

print("Erro Quadrático Médio usando validação cruzada:",(lir_cv.mean()*(-1)))

print('Raiz do Erro Quadrático Médio usando validação cruzada:', np.sqrt((lir_cv.mean()*(-1))), "\n")

lir1 = np.sqrt((lir_cv.mean()*(-1)))



###KNN

KNN = KNeighborsRegressor(n_neighbors=3)

KNN = KNN.fit(X_train, y_train)

pred_knn = KNN.predict(X_test)

print('K vizinhos mais próximos:')

print('Erro Quadrático Médio:', mean_squared_error(y_test, pred_knn))

print('Raiz do Erro Quadrático Médio:', np.sqrt(mean_squared_error(y_test, pred_knn)))

knn0 = np.sqrt(mean_squared_error(y_test, pred_knn))

#Validação cruzada

knn_cv = cross_val_score(estimator = KNN, X = X_train, y = y_train, cv = 10, scoring='neg_mean_squared_error')

print("Erro Quadrático Médio usando validação cruzada:",(knn_cv.mean()*(-1)))

print('Raiz do Erro Quadrático Médio usando validação cruzada:', np.sqrt((knn_cv.mean()*(-1))), "\n")

knn1 = np.sqrt((knn_cv.mean()*(-1)))



conc1 = {

    'Holdout':[dtr0,rfr0,lir0,knn0] ,

    'CrossValidation':[dtr1,rfr1,lir1,knn1]

}

concl1 = pd.DataFrame(conc1, columns=['Holdout', 'CrossValidation'], index=['DT', 'RF', 'LR', 'KNN'])
print('Selecionando 4 atributos mais relevantes:')

print(concl4)

print('\nSelecionando 5 atributos mais relevantes:')

print(concl3)

print('\nSelecionando 6 atributos mais relevantes:')

print(concl2)

print('\nSelecionando 7 atributos mais relevantes:')

print(concl1)

print('\nSelecionando todos os atributos:')

print(concl0)