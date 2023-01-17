import numpy as np 

import pandas as pd 

import tensorflow as tf

import matplotlib.pyplot as plt



import math

from sklearn import preprocessing, model_selection

from sklearn.metrics import mean_squared_error

import xgboost as xgb

import scipy.stats as stats

df_belem = pd.read_csv("../input/temperature-timeseries-for-some-brazilian-cities/station_belem.csv")

df_curitiba = pd.read_csv("../input/temperature-timeseries-for-some-brazilian-cities/station_curitiba.csv")
#Questão 1

display(df_belem.shape, df_curitiba.shape)
#Questão 2

df_belem.set_index('YEAR',inplace=True)

df_curitiba.set_index('YEAR',inplace=True)

display(df_belem.head())

display(df_curitiba.head())
#exercício 3

#plota o histograma dos valores 

#vemos que há vários outliers próximos de 1000 que provavelmente não são valores de temperatura válidos

plt.figure(figsize=(25,25))

#df_belem.boxplot()

df_belem.hist()



#mostra a quantidade de valores únicos no mẽs de janeiro

#verificando os valores únicos confirmamos que o valor 999.90 é o único outlier

display(df_belem['JAN'].value_counts())

display(df_curitiba['JAN'].value_counts())
#exercício 4

#para tratar os outliers, podemos excluir os dados ausentes (999.90) ou substituí-lo pela média do ano anterior e posterior.

#adotarei a solução de substituir os nulos pela média.





#cria um novo dataset transformando o outlier em nulo para aplicação das funções de tratamento

df_belem_t = df_belem.replace(999.90,np.nan)

#substitui os valores nulos restantes pela média do ano anterior e posterior

df_belem_t = df_belem_t.fillna(df_belem_t.mean())

display(df_belem_t)



#cria um novo dataset transformando o outlier em nulo para aplicação das funções de tratamento

df_curitiba_t = df_curitiba.replace(999.90,np.nan)

#substitui os valores nulos restantes pela média do ano anterior e posterior

df_curitiba_t = df_curitiba_t.fillna(df_curitiba_t.mean())

display(df_curitiba_t)

#exercício 5

plt.figure(figsize=(10,10))

fig=plt.figure()

ax=fig.add_axes([0,0,1,1])

ax.scatter(df_curitiba_t.index, df_curitiba_t.JUL, color='r')

ax.scatter(df_belem_t.index, df_belem_t.JUL, color='b')

ax.set_xlabel('Ano')

ax.set_ylabel('Temperatura (ºC)')

ax.legend(["Curitiba - Julho", "Belém - Julho"])

ax.set_title('scatter plot')

plt.show()
#questão 6

display(df_curitiba_t['JUL'].describe())

display(df_belem_t['JUL'].describe())

stats.f_oneway(df_belem_t['JUL'], df_curitiba_t['JUL'])
#exercício 7

df_curitiba_jan = pd.DataFrame(df_curitiba_t['JAN'],columns=['JAN'])

#cria o dataset de previsão com os valores dos 3 anos anteriores

df_curitiba_jan['A1'] = df_curitiba_jan['JAN'].shift(1)

df_curitiba_jan['A2'] = df_curitiba_jan['JAN'].shift(2)

df_curitiba_jan['A3'] = df_curitiba_jan['JAN'].shift(3)

#dropa os primeiros anos (que não tem anos anteriores para montar o dataset)

df_curitiba_jan = df_curitiba_jan.dropna()

display(df_curitiba_jan.head())

#separa em conjuntos de teste e treinamento

X_train, X_test, y_train, y_test = model_selection.train_test_split(df_curitiba_jan.drop(columns=['JAN']),df_curitiba_jan['JAN'],test_size=0.25, random_state=33)

#realiza regressão com o regressor de gradient boosting XGBoost

#ele frequentemente apresenta resultados iniciais melhores que uma rede neural sem ajustes

model = xgb.XGBRegressor()

model.fit(X_train,y_train)

p_train = model.predict(data=X_train)

p_test = model.predict(data=X_test)
#calcula os erros de previsão

trainScore = math.sqrt(mean_squared_error(p_train, y_train))

print('Pontuação para o treinamento: %.2f RMSE' % (trainScore))

testScore = math.sqrt(mean_squared_error(p_test, y_test))

print('Pontuação para o teste: %.2f RMSE' % (testScore))
#plota o resultado previsto em relação ao real

df_plot = pd.DataFrame({'YEAR': X_test.index, 'PRED': p_test, 'REAL': y_test}).reset_index(drop=True)

display(df_plot.sort_values(['YEAR']).set_index('YEAR'))

plt.figure(figsize=(10,10))

fig=plt.figure()

ax=fig.add_axes([0,0,1,1])

ax.scatter(df_plot['YEAR'],df_plot['PRED'] , color='r')

ax.scatter(df_plot['YEAR'],df_plot['REAL'] , color='b')

ax.set_xlabel('Ano')

ax.set_ylabel('Temperatura (ºC)')

ax.legend(["Curitiba - Janeiro - Previsto", "Curitiba - Janeiro - Real"])

ax.set_title('scatter plot')

plt.show()