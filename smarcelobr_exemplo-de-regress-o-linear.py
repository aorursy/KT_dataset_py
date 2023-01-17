%matplotlib inline

import matplotlib

import matplotlib.pyplot as plt

import pandas as pd



from scipy import stats

import numpy as np  

import scipy as sp

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

from sklearn.preprocessing import StandardScaler, MinMaxScaler
teste = pd.DataFrame({'input': [0,1,2,3,4,5,6,7,8,9,10,11],

                     'output': [100,81,64,49,36,25,25,36,49,64,81,100]})
def prepararX(X):

    # esta funcao prepara o X (dados de entrada) adicionando features quadráticas.

    X_new = PolynomialFeatures(degree=2, interaction_only=False, include_bias=True).fit_transform(X) 

    #X_new = MinMaxScaler().fit_transform(X_new)

    return X_new





#veja a documenação do classificador e veja o que é possível alterar.

X = teste['input'].values.reshape(-1,1) # dados de entrada

y = teste['output'].values              # resultados esperados



# poe no gráfico os dados esperados

plt.plot(teste['input'], teste['output'],  'ko') # 



lr = LinearRegression(fit_intercept=True, normalize=True) 

lr.fit(prepararX(X),y) # treinamento dos dados



# dados para testes:

X_test = [[-1],[0],[1],[2],[3],[5],[7],[8],[10],[12]]

Y_predict = lr.predict( prepararX(X_test) ) # faz a previsão dos dados dos testes.



# poe no gráfico os dados previstos

plt.plot(X_test, Y_predict, color='blue')



#=== Meus pesos (que fiz manualmente)

a = 2.43

b = 26.8

c = 100



# poe no gráfico os que botei manualmente acima

plt.plot(X_test, [ (a*(x[0]**2) - (b*x[0]) + c) for x in X_test],  color='red')