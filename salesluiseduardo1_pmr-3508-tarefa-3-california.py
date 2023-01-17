import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
import os
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression, Lasso, Ridge
train = pd.read_csv("../input/californian-data/train.csv")
train.pop("Id")
train.shape
test = pd.read_csv("../input/californian-data/train.csv")
test.pop("Id")
test.head()
train.describe()
train.corr()
plt.matshow(train.corr())
plt.colorbar() # a partir da análise do gráfico temos a informação que o preço tem um correlação relativamente alta com a média 
#salarial
correlacao = train.corr()
correlacao_value = correlacao["median_house_value"].abs()
correlacao_value.sort_values() #é interessante notar que a latitude também apresenta correlação relativamente alta junto com o número de cómodos
#faz sentido isso a partir do pressuposto que quanto maior a casa, maior o preço, e a latitude está relacionada a posição das áreas
#de grande densidade demográfica como São Francisco e Los Angeles
train.hist(figsize=(12,8),bins=60)
trainY = train['median_house_value']
trainX = train.drop(columns = ['median_house_value'])
trainY
trainX
testX = test
regression_linear = linear_model.LinearRegression()
regression_linear.fit(trainX, trainY)
print('Coefficients: \n', regression_linear.coef_)
scores = cross_val_score(LinearRegression(), trainX, trainY, cv=10, scoring = "neg_mean_squared_error")
print("Erro RSME médio: ", np.sqrt(-scores.mean())/trainY.mean())#Linear Regression
regression_ridge = linear_model.Ridge (alpha = .5) #Ridge Regression
regression_ridge.fit(trainX, trainY)
regression_ridge.coef_
scores = cross_val_score(Ridge(), trainX, trainY, cv=10, scoring = "neg_mean_squared_error")
print("Erro RSME médio: ", np.sqrt(-scores.mean())/trainY.mean())
scores = cross_val_score(Lasso(), trainX, trainY, cv=10, scoring = "neg_mean_squared_error")
print("Erro RSME médio: ", np.sqrt(-scores.mean())/trainY.mean())#Lasso Regression
knr = KNeighborsRegressor(n_neighbors = 30)
scores = cross_val_score(knr, trainX, trainY, cv=10, scoring = "neg_mean_squared_error")
print("Erro RSME médio: ", np.sqrt(-scores.mean())/trainY.mean())#KNN Regression
min(0.39164335063637146,0.337570191679481,0.3375701542868961,0.3375702051504674)
#O menor erro obtido foi no caso em que utilizamos a Ridge regression