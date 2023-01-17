#Imports necessários
import numpy as np
import pandas as pds
import collections
from sklearn import preprocessing
from pandas import concat
import matplotlib
#Analise do dataset
data = pds.read_csv('../input/chicago-bike-sharedatasetprediction/dataset_prediction.csv')
print(len(data)) 
data.head()
#Correlação de atributos
corrData = data.copy()
df_corr = corrData.corr()
df_corr
#Dados de locacao de dia de semana por fim de semana
data.groupby(['is_weekend'])['rentals'].sum().plot(kind='bar')
#Dados de locacao por clima (chuva ou neve)
data.groupby(['rain_or_snow'])['rentals'].sum().plot(kind='bar')
#Dados de locacao por clima nublado
data.groupby(['tstorms'])['rentals'].sum().plot(kind='bar')
#Dados de locacao por clima nublado
data.groupby(['cloudy'])['rentals'].sum().plot(kind='bar')
data.groupby(['month'])['rentals'].sum().plot(kind='bar')
data.groupby(['year'])['rentals'].sum().plot(kind='bar')
data.groupby(['hour'])['rentals'].sum().plot(kind='bar')
#Pre-Processamento dos dados

#Normalizacao
data.year = preprocessing.scale(list(data.year))
data.month = preprocessing.scale(list(data.month))
data.week = preprocessing.scale(list(data.week))
data.day = preprocessing.scale(list(data.day))
data.hour = preprocessing.scale(list(data.hour))
data.mean_temperature = preprocessing.scale(list(data.mean_temperature))
data.median_temperature = preprocessing.scale(list(data.median_temperature))

#Visualização dos dados
data.head()
#Verificacao de dados do dataset
collections.Counter(data.is_weekend)
#Modelos a serem testados
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression, BayesianRidge, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor

#Retirada da variável target das features de predição
X = data.drop('rentals',1)
y = data.rentals

#Separação de conjunto de testes
X_model, X_test, y_model, y_test = train_test_split(X, y, test_size=0.2, random_state=0)  

#Separação de conjunto de validação
X_train, X_val, y_train, y_val = train_test_split(X_model, y_model, test_size=0.2, random_state=0)  
#Treinamento de modelos 
lr = LinearRegression(n_jobs=5, fit_intercept=True)
logr = LogisticRegression(penalty='l2', C=1.0, fit_intercept=True, random_state=0, solver='liblinear', n_jobs=5)
dt = DecisionTreeRegressor(max_depth=10, criterion='mse', splitter='best', random_state=0, presort=True)
dtr = AdaBoostRegressor(dt,n_estimators=500, learning_rate=0.1, random_state=0)
rf = RandomForestRegressor(n_estimators=100, criterion='mse', max_features='auto', random_state=0, n_jobs=5)
blr = BayesianRidge(n_iter=1000, fit_intercept=True)

#Criacao de vetor de modelos
algs = []
algs.append(lr)
algs.append(logr)
algs.append(dt)
algs.append(dtr)
algs.append(rf)
algs.append(blr)

#Fit dos modelos
for alg in algs:
    print('Fitting: ', type(alg).__name__)
    alg.fit(X_model, y_model)  
#Definição de dataframe para exibição de resultados
results = pds.DataFrame(columns=['Name', 'Type', 'R2', 'MAE', 'MSE'])
#Função para display de resultados
def appendResult(alg, dataType, X, y):
    algName = type(alg).__name__
    predicted = alg.predict(X)
    mae = mean_absolute_error(y, predicted)
    mse = mean_squared_error(y, predicted)
    r2 = r2_score(y, predicted)
    results.loc[len(results)]=[algName, dataType, r2, mae, mse]
#Treinamento
for alg in algs:
    appendResult(alg, 'Train', X_train, y_train)
#Validação do treinamento
for alg in algs:
    appendResult(alg, 'Validation', X_val, y_val)
#Teste final
for alg in algs:
    appendResult(alg, 'Test', X_test, y_test)
    
results