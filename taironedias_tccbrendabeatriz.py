# from sklearn import linear_model, datasets



import scipy as sp



from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from  sklearn.model_selection  import  TimeSeriesSplit

from sklearn.metrics import accuracy_score

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error

from sklearn import metrics

from sklearn import preprocessing

import seaborn as sns

import math

from sklearn.ensemble import RandomForestRegressor

from sklearn.datasets import make_regression

from sklearn.linear_model import LinearRegression

from sklearn.neighbors import KNeighborsRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score

from sklearn.preprocessing  import StandardScaler 



import os

import warnings

warnings.filterwarnings('ignore')

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight') 

# Above is a special style template for matplotlib, highly useful for visualizing time series data

%matplotlib inline

from pylab import rcParams

from plotly import tools

import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.figure_factory as ff

import statsmodels.api as sm

from numpy.random import normal, seed

from scipy.stats import norm

from statsmodels.tsa.arima_model import ARMA

from statsmodels.tsa.stattools import adfuller

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from statsmodels.tsa.arima_process import ArmaProcess

from statsmodels.tsa.arima_model import ARIMA



print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
path = '../input/database2.xls'

file = pd.read_excel(path)

file.head()
file['PrecipitacaoTotal'].plot(figsize=(16, 8))

plt.title('')

plt.xlabel('Meses')

plt.ylabel('Precipitação (mm)')


tam = file['Data'].count()



for i in range(0, tam):

    # Pegando o valor de cada linha da coluna Data

    strData = file['Data'].iloc[[i][0]]

    # Convertendo em string

    str(strData)

    # Utilizando a funcao strftime para recuperar o mes e ano em numeros

    mm = int(strData.strftime('%m'))

    file['Mes'].iloc[[i]] = mm



file.head()
# Removendo a coluna Data da base original

coluna=['Data']

file.drop(coluna, axis=1, inplace=True)

file.head()
%matplotlib inline

boxplot = file.boxplot(column='DirecaoVento')
%matplotlib inline

boxplot = file.boxplot(column='NebulosidadeMedia')
%matplotlib inline

boxplot = file.boxplot(column='PressaoMedia')
%matplotlib inline

boxplot = file.boxplot(column='PrecipitacaoTotal')
plt.figure(figsize=(7,4))

sns.heatmap(file.corr(), annot=True, cmap='cubehelix_r')

plt.show()
# Algoritmo KNN para todos os meses

print('Algoritmo KNN para todos os meses')

for i in range(1,13):

    print('\nMÊS: {}'.format(i))

    aux = file[~(((file.Mes < i) | (file.Mes > i)))].copy(['Mes','Ano','DirecaoVento','NebulosidadeMedia','PressaoMedia','PrecipitacaoTotal'])

    

    temp = aux[~(( aux.Ano == 2018))].copy(['Mes','Ano','DirecaoVento','NebulosidadeMedia','PressaoMedia','PrecipitacaoTotal'])

    

    X = temp[~(( temp.Ano == 2017))].drop(['PrecipitacaoTotal'], axis=1)

    

    y = temp[~(( temp.Ano == 2017))].drop(['Mes','Ano','DirecaoVento','NebulosidadeMedia','PressaoMedia'], axis=1)

    

    X_test = temp[(temp.Ano == 2017)].drop(['PrecipitacaoTotal'], axis=1)

    

    y_test = temp[(temp.Ano == 2017)].drop(['Ano','Mes','PressaoMedia','NebulosidadeMedia','DirecaoVento'], axis=1)

    

    X_valid = aux[(aux.Ano == 2018)].drop(['PrecipitacaoTotal'], axis=1)

    

    y_valid = aux[(aux.Ano == 2018)].drop(['Ano','Mes','PressaoMedia','NebulosidadeMedia','DirecaoVento'], axis=1)

    

    knn = KNeighborsRegressor(n_neighbors=3)

    knn.fit(X,y)

    trainPredicted_knn = knn.predict(X)

    testPredicted_knn = knn.predict(X_test)

    validPredicted_knn = knn.predict(X_valid)

#     print('\nValores previsto para Treino: ')

#     print(trainPredicted_knn)

#     print('\nValores previsto para Teste: ')

#     print(testPredicted_knn)

#     print('\nValores previsto para Validação: ')

#     print(validPredicted_knn)



    trainScore_knn = math.sqrt(mean_squared_error(y, trainPredicted_knn))

    testScore_knn = math.sqrt(mean_squared_error(y_test, testPredicted_knn))

    validScore_knn = math.sqrt(mean_squared_error(y_valid, validPredicted_knn))

    print('\tAvaliação do Treino KNN: %.2f REQM' % (trainScore_knn))

    print('\tAvaliação do Teste KNN: %.2f REQM' % (testScore_knn))

    print('\tAvaliação da Validação KNN: %.2f REQM\n' % (validScore_knn))

    

    trainR2Score_knn = r2_score(y, trainPredicted_knn)

    print('\tAvaliação do Treino KNN: %.2f R2' % (trainR2Score_knn))
# Algoritmo Regressão Linear para todos os meses

print('Algoritmo Regressão Linear para todos os meses')

for i in range(1,13):

    print('\nMÊS: {}'.format(i))

    aux = file[~(((file.Mes < i) | (file.Mes > i)))].copy(['Mes','Ano','DirecaoVento','NebulosidadeMedia','PressaoMedia','PrecipitacaoTotal'])

    temp = aux[~(( aux.Ano == 2018))].copy(['Mes','Ano','DirecaoVento','NebulosidadeMedia','PressaoMedia','PrecipitacaoTotal'])

    X = temp[~(( temp.Ano == 2017))].drop(['PrecipitacaoTotal'], axis=1)

    y = temp[~(( temp.Ano == 2017))].drop(['Mes','Ano','DirecaoVento','NebulosidadeMedia','PressaoMedia'], axis=1)

    print('\n')

    print(X)

    print('\n')

    print(y)

    print('\n')

    

    X_test = temp[(temp.Ano == 2017)].drop(['PrecipitacaoTotal'], axis=1)

    y_test = temp[(temp.Ano == 2017)].drop(['Ano','Mes','PressaoMedia','NebulosidadeMedia','DirecaoVento'], axis=1)



    X_valid = aux[(aux.Ano == 2018)].drop(['PrecipitacaoTotal'], axis=1)

    y_valid = aux[(aux.Ano == 2018)].drop(['Ano','Mes','PressaoMedia','NebulosidadeMedia','DirecaoVento'], axis=1)

    

    lr = LinearRegression(normalize = True)

    lr.fit(X,y)

    trainPredicted_lr = lr.predict(X)

    testPredicted_lr = lr.predict(X_test)

    validPredicted_lr = lr.predict(X_valid)

#     print('\nValores previsto para Treino: ')

#     print(trainPredicted_lr)

#     print('\nValores previsto para Teste: ')

#     print(testPredicted_lr)

#     print('\nValores previsto para Validação: ')

#     print(validPredicted_lr)

    

    trainScore_lr = math.sqrt(mean_squared_error(y, trainPredicted_lr))

    testScore_lr = math.sqrt(mean_squared_error(y_test, testPredicted_lr))

    validScore_lr = math.sqrt(mean_squared_error(y_valid, validPredicted_lr))

    print('\tAvaliação do Treino RL: %.2f REQM' % (trainScore_lr))

    print('\tAvaliação do Teste RL: %.2f REQM' % (testScore_lr))

    print('\tAvaliação da Validação RL: %.2f REQM\n' % (validScore_lr))

    

    trainR2Score_lr = r2_score(y, trainPredicted_lr)

    print('\tAvaliação do Treino RL: %.2f R2' % (trainR2Score_lr))
# Algoritmo Árvore de Decisão para todos os meses

print('Algoritmo Árvore de Decisão para todos os meses')

for i in range(1,13):

    print('\nMÊS: {}'.format(i))

    aux = file[~(((file.Mes < i) | (file.Mes > i)))].copy(['Mes','Ano','DirecaoVento','NebulosidadeMedia','PressaoMedia','PrecipitacaoTotal'])

    temp = aux[~(( aux.Ano == 2018))].copy(['Mes','Ano','DirecaoVento','NebulosidadeMedia','PressaoMedia','PrecipitacaoTotal'])

    X = temp[~(( temp.Ano == 2017))].drop(['PrecipitacaoTotal'], axis=1)

    y = temp[~(( temp.Ano == 2017))].drop(['Mes','Ano','DirecaoVento','NebulosidadeMedia','PressaoMedia'], axis=1)

    

    X_test = temp[(temp.Ano == 2017)].drop(['PrecipitacaoTotal'], axis=1)

    y_test = temp[(temp.Ano == 2017)].drop(['Ano','Mes','PressaoMedia','NebulosidadeMedia','DirecaoVento'], axis=1)

    

    X_valid = aux[(aux.Ano == 2018)].drop(['PrecipitacaoTotal'], axis=1)

    y_valid = aux[(aux.Ano == 2018)].drop(['Ano','Mes','PressaoMedia','NebulosidadeMedia','DirecaoVento'], axis=1)

    

    dtr = DecisionTreeRegressor(max_depth=2)

    dtr.fit(X,y)

    trainPredicted_dtr = dtr.predict(X)

    testPredicted_dtr = dtr.predict(X_test)

    validPredicted_dtr = dtr.predict(X_valid)

#     print('\nValores previsto para Treino: ')

#     print(trainPredicted_dtr)

#     print('\nValores previsto para Teste: ')

#     print(testPredicted_dtr)

#     print('\nValores previsto para Validação: ')

#     print(validPredicted_dtr)

    

    trainScore_dtr = math.sqrt(mean_squared_error(y, trainPredicted_dtr))

    testScore_dtr = math.sqrt(mean_squared_error(y_test, testPredicted_dtr))

    validScore_dtr = math.sqrt(mean_squared_error(y_valid, validPredicted_dtr))

    print('\tAvaliação do Treino AD: %.2f REQM' % (trainScore_dtr))

    print('\tAvaliação do Teste AD: %.2f REQM' % (testScore_dtr))

    print('\tAvaliação da Validação AD: %.2f REQM\n' % (validScore_dtr))

    

    trainR2Score_dtr = r2_score(y, trainPredicted_dtr)

    print('\tAvaliação do Treino AD: %.2f R2' % (trainR2Score_dtr))
# Algoritmo Floresta Aleatória para todos os meses

print('Algoritmo Floresta Aleatória para todos os meses')

for i in range(1,13):

    print('\nMÊS: {}'.format(i))

    aux = file[~(((file.Mes < i) | (file.Mes > i)))].copy(['Mes','Ano','DirecaoVento','NebulosidadeMedia','PressaoMedia','PrecipitacaoTotal'])

    temp = aux[~(( aux.Ano == 2018))].copy(['Mes','Ano','DirecaoVento','NebulosidadeMedia','PressaoMedia','PrecipitacaoTotal'])

    X = temp[~(( temp.Ano == 2017))].drop(['PrecipitacaoTotal'], axis=1)

    y = temp[~(( temp.Ano == 2017))].drop(['Mes','Ano','DirecaoVento','NebulosidadeMedia','PressaoMedia'], axis=1)

    

    X_test = temp[(temp.Ano == 2017)].drop(['PrecipitacaoTotal'], axis=1)

    y_test = temp[(temp.Ano == 2017)].drop(['Ano','Mes','PressaoMedia','NebulosidadeMedia','DirecaoVento'], axis=1)

    

    X_valid = aux[(aux.Ano == 2018)].drop(['PrecipitacaoTotal'], axis=1)

    y_valid = aux[(aux.Ano == 2018)].drop(['Ano','Mes','PressaoMedia','NebulosidadeMedia','DirecaoVento'], axis=1)

    

    rfr = RandomForestRegressor(max_depth=2, n_estimators=7, random_state=0)

    rfr.fit(X, y)

    trainPredicted_rfr = rfr.predict(X)

    testPredicted_rfr = rfr.predict(X_test)

    validPredicted_rfr = rfr.predict(X_valid)

#     print('\nValores previsto para Treino: ')

#     print(trainPredicted_rfr)

#     print('\nValores previsto para Teste: ')

#     print(testPredicted_rfr)

#     print('\nValores previsto para Validação: ')

#     print(validPredicted_rfr)

    

    trainScore_rfr = math.sqrt(mean_squared_error(y, trainPredicted_rfr))

    testScore_rfr = math.sqrt(mean_squared_error(y_test, testPredicted_rfr))

    validScore_rfr = math.sqrt(mean_squared_error(y_valid, validPredicted_rfr))

    print('\tAvaliação do Treino FA: %.2f REQM' % (trainScore_rfr))

    print('\tAvaliação do Teste FA: %.2f REQM' % (testScore_rfr))

    print('\tAvaliação da Validação FA: %.2f REQM\n' % (validScore_rfr))

    

    trainR2Score_rfr = r2_score(y, trainPredicted_rfr)

    print('\tAvaliação do Treino FA: %.2f R2' % (trainR2Score_rfr))
#Fazendo um exemplo para plotar um gráfico de precipitação por um mês específico de 2007 até 2016. Considerando esse mês específico o mês de agosto, teremos:

mes = 8

aux = file[~(((file.Mes < mes) | (file.Mes > mes)))].copy(['Mes','Ano','DirecaoVento','NebulosidadeMedia','PressaoMedia','PrecipitacaoTotal'])

temp = aux[~(( aux.Ano == 2018))].copy(['Mes','Ano','DirecaoVento','NebulosidadeMedia','PressaoMedia','PrecipitacaoTotal'])

X_aux = temp[~(( temp.Ano == 2017))].drop(['Mes','DirecaoVento','NebulosidadeMedia','PressaoMedia','PrecipitacaoTotal'], axis=1)

y = temp[~(( temp.Ano == 2017))].drop(['Mes','Ano','DirecaoVento','NebulosidadeMedia','PressaoMedia'], axis=1)

# print(X_aux)

# print(y)

plt.figure()



plt.scatter(X_aux, y, s=20, edgecolor="black", c="darkorange", label="dados")

plt.xlabel('Anos (mês: agosto)')

plt.ylabel('Precipitação (mm)')

plt.title('')

plt.legend()

plt.show()
aux = file[~(((file.Mes < 8) | (file.Mes > 8)))].copy(['Mes','Ano','DirecaoVento','NebulosidadeMedia','PressaoMedia','PrecipitacaoTotal'])

temp = aux[~(( aux.Ano == 2018))].copy(['Mes','Ano','DirecaoVento','NebulosidadeMedia','PressaoMedia','PrecipitacaoTotal'])

X = temp[~(( temp.Ano == 2017))].drop(['PrecipitacaoTotal'], axis=1)

y = temp[~(( temp.Ano == 2017))].drop(['Mes','Ano','DirecaoVento','NebulosidadeMedia','PressaoMedia'], axis=1)

print('Dados X e y:')

print(X)

print(y)



X_test = temp[(temp.Ano == 2017)].drop(['PrecipitacaoTotal'], axis=1)

y_test = temp[(temp.Ano == 2017)].drop(['Ano','Mes','PressaoMedia','NebulosidadeMedia','DirecaoVento'], axis=1)

print('\nDados X_test e y_test:')

print(X_test)

print(y_test)



X_valid = aux[(aux.Ano == 2018)].drop(['PrecipitacaoTotal'], axis=1)

y_valid = aux[(aux.Ano == 2018)].drop(['Ano','Mes','PressaoMedia','NebulosidadeMedia','DirecaoVento'], axis=1)

print('\nDados X_valid e y_valid:')

print(X_valid)

print(y_valid)
knn = KNeighborsRegressor(n_neighbors=3)

knn.fit(X,y)

trainPredicted_knn = knn.predict(X)

testPredicted_knn = knn.predict(X_test)

validPredicted_knn = knn.predict(X_valid)

print(testPredicted_knn)

print(validPredicted_knn)
lr = LinearRegression(normalize = True)

lr.fit(X,y)

trainPredicted_lr = lr.predict(X)

testPredicted_lr = lr.predict(X_test)

validPredicted_lr = lr.predict(X_valid)

print(testPredicted_lr)

print(validPredicted_lr)
dtr = DecisionTreeRegressor()

dtr.fit(X,y)

trainPredicted_dtr = dtr.predict(X)

testPredicted_dtr = dtr.predict(X_test)

validPredicted_dtr = dtr.predict(X_valid)

print(testPredicted_dtr)

print(validPredicted_dtr)
# ANALISANDO QUAL O MELHOR VALOR PARA ATRIBUIR AO max_depth E n_estimators

# for i in range(1,6):

#     for j in range(2,8):

#         rfr = RandomForestRegressor(max_depth=i, n_estimators=j, random_state=0).fit(X,y)

#         trainPredicted_rfr = rfr.predict(X)

#         testPredicted_rfr = rfr.predict(X_test)

#         validPredicted_rfr = rfr.predict(X_valid)

#         print('\nmax_depth={} e n_estimators={}'.format(i,j))

#         print(testPredicted_rfr)

#         print(validPredicted_rfr)



#         testScore_rfr = math.sqrt(mean_squared_error(y_test, testPredicted_rfr))

#         validScore_rfr = math.sqrt(mean_squared_error(y_valid, validPredicted_rfr))

#         print('Avaliação do Teste Random Forest Regressor: %.2f REQM' % (testScore_rfr))

#         print('Avaliação da Validação Random Forest Regressor: %.2f REQM' % (validScore_rfr))

#         print('Disparidade entre os erros: %.2f' %(testScore_rfr - validScore_rfr))
rfr = RandomForestRegressor(max_depth=2, 

                            n_estimators=7, 

                            random_state=0)

rfr.fit(X, y)

trainPredicted_rfr = rfr.predict(X)

testPredicted_rfr = rfr.predict(X_test)

validPredicted_rfr = rfr.predict(X_valid)

print(testPredicted_rfr)

print(validPredicted_rfr)
trainScore_knn = math.sqrt(mean_squared_error(y, trainPredicted_knn))

testScore_knn = math.sqrt(mean_squared_error(y_test, testPredicted_knn))

validScore_knn = math.sqrt(mean_squared_error(y_valid, validPredicted_knn))

print('Avaliação do Treino KNN: %.2f REQM' % (trainScore_knn))

print('Avaliação do Teste KNN: %.2f REQM' % (testScore_knn))

print('Avaliação da Validação KNN: %.2f REQM' % (validScore_knn))



print('\n\nTestes para o R2')

print(y)

print(trainPredicted_knn)

trainR2Score_knn = r2_score(y, trainPredicted_knn)

print('\tAvaliação do Treino KNN: %.2f R2\n' % (trainR2Score_knn))

print(y_test)

print(testPredicted_knn)

testR2Score_knn = r2_score(y_test, testPredicted_knn)

print('\tAvaliação do Treino KNN: %.2f R2\n' % (testR2Score_knn))

# test = []

# for i in range(y_valid['PrecipitacaoTotal'].count()):

#     aux = [y_valid['PrecipitacaoTotal'].iloc[[i][0]]]

#     print(aux)

#     test.append(aux)

#     print(test)

# print(test)

# print(validPredicted_knn)

# validR2Score_knn = r2_score(test, validPredicted_knn)

# print('\tAvaliação do Treino KNN: %.2f R2' % (validR2Score_knn))
trainScore_knn = math.sqrt(mean_squared_error(y, trainPredicted_knn))

testScore_knn = math.sqrt(mean_squared_error(y_test, testPredicted_knn))

validScore_knn = math.sqrt(mean_squared_error(y_valid, validPredicted_knn))

print('Avaliação do Treino KNN: %.2f REQM' % (trainScore_knn))

print('Avaliação do Teste KNN: %.2f REQM' % (testScore_knn))

print('Avaliação da Validação KNN: %.2f REQM' % (validScore_knn))



trainScore_lr = math.sqrt(mean_squared_error(y, trainPredicted_lr))

testScore_lr = math.sqrt(mean_squared_error(y_test, testPredicted_lr))

validScore_lr = math.sqrt(mean_squared_error(y_valid, validPredicted_lr))

print('\nAvaliação do Treino RL: %.2f REQM' % (trainScore_lr))

print('Avaliação do Teste RL: %.2f REQM' % (testScore_lr))

print('Avaliação da Validação RL: %.2f REQM' % (validScore_lr))



trainScore_dtr = math.sqrt(mean_squared_error(y, trainPredicted_dtr))

testScore_dtr = math.sqrt(mean_squared_error(y_test, testPredicted_dtr))

validScore_dtr = math.sqrt(mean_squared_error(y_valid, validPredicted_dtr))

print('\nAvaliação do Treino AD: %.2f REQM' % (trainScore_dtr))

print('Avaliação do Teste AD: %.2f REQM' % (testScore_dtr))

print('Avaliação da Validação AD: %.2f REQM' % (validScore_dtr))



trainScore_rfr = math.sqrt(mean_squared_error(y, trainPredicted_rfr))

testScore_rfr = math.sqrt(mean_squared_error(y_test, testPredicted_rfr))

validScore_rfr = math.sqrt(mean_squared_error(y_valid, validPredicted_rfr))

print('\nAvaliação do Treino Random Forest Regressor: %.2f REQM' % (trainScore_rfr))

print('Avaliação do Teste Random Forest Regressor: %.2f REQM' % (testScore_rfr))

print('Avaliação da Validação Random Forest Regressor: %.2f REQM' % (validScore_rfr))
y_true = [[1]]

y_pred = [[3]]

print(r2_score(y_true, y_pred))

y_true = [[1],[2]]

y_pred = [[3],[1]]

print(r2_score(y_true, y_pred))
# Algoritmo KNN para todos os meses, com test sendo os anos de 2017 e 2018

print('Algoritmo KNN para todos os meses')

for i in range(1,13):

    print('\nMÊS: {}'.format(i))

    aux = file[~(((file.Mes < i) | (file.Mes > i)))].copy(['Mes','Ano','DirecaoVento','NebulosidadeMedia','PressaoMedia','PrecipitacaoTotal'])

    temp = aux[~(( aux.Ano == 2018))].copy(['Mes','Ano','DirecaoVento','NebulosidadeMedia','PressaoMedia','PrecipitacaoTotal'])

    X = temp[~(( temp.Ano == 2017))].drop(['PrecipitacaoTotal'], axis=1)

    y = temp[~(( temp.Ano == 2017))].drop(['Mes','Ano','DirecaoVento','NebulosidadeMedia','PressaoMedia'], axis=1)

    print('Dados X:')

    print(X)

    print('\nDados y:')

    print(y)

    

    X_test = aux[~((aux.Ano < 2017))].drop(['PrecipitacaoTotal'], axis=1)

    y_test = aux[~((aux.Ano < 2017))].drop(['Ano','Mes','PressaoMedia','NebulosidadeMedia','DirecaoVento'], axis=1)

    print('\nDados X_test:')

    print(X_test)

    print('\nDados y_test:')

    print(y_test)

    

    knn = KNeighborsRegressor(n_neighbors=3)

    knn.fit(X,y)

    trainPredicted_knn = knn.predict(X)

    testPredicted_knn = knn.predict(X_test)

    

    trainScore_knn = math.sqrt(mean_squared_error(y, trainPredicted_knn))

    testScore_knn = math.sqrt(mean_squared_error(y_test, testPredicted_knn))

    print('\tAvaliação do Treino KNN: %.2f REQM' % (trainScore_knn))

    print('\tAvaliação do Teste KNN: %.2f REQM\n' % (testScore_knn))

    

    trainR2Score_knn = r2_score(y, trainPredicted_knn)

    print('\tAvaliação do Treino KNN: %.2f R2' % (trainR2Score_knn))

    testR2Score_knn = r2_score(y_test, testPredicted_knn)

    print('\tAvaliação do Teste KNN: %.2f R2' % (testR2Score_knn))