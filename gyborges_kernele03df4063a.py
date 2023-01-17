#imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn 
from sklearn.neighbors import KNeighborsRegressor
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
import os
#leitura dos arquivos 
trainData = pd.read_csv("../input/databa/train.csv")
testData = pd.read_csv("../input/databa/test.csv")
#cabeçalho
trainData.head()
#cabeçalho
testData.head()
#número de linhas e colunas
trainData.shape
#número de linhas e colunas
testData.shape
#função que calcula rmsle
def calc_rmsle(Yreal, Ypred):
    sum=0.0
    n=len(Yreal)
    for x in range(n):
        if Ypred[x]<0:
            sum = sum + (0 - np.log(Yreal[x]+1))**2            
        else:
            sum = sum + (np.log(Ypred[x]+1) - np.log(Yreal[x]+1))**2
    return np.sqrt((sum/n))

Xtrain = trainData
Xtrain = Xtrain.drop('Id', axis=1)
Xtrain = Xtrain.drop('median_house_value', axis=1)
Ytrain = trainData['median_house_value']

Xtest = testData
Xtest = Xtest.drop('Id', axis=1)
#knn


best_rmsle = 1
for i in range(3, 30):
    knn = KNeighborsRegressor(n_neighbors=i)
    knn.fit(Xtrain, Ytrain)
    Ypred = knn.predict(Xtrain)
    if calc_rmsle(Ytrain, Ypred) < best_rmsle:
        best_n = i
        best_rmsle = calc_rmsle(Ytrain, Ypred)
print(best_n)
print(best_rmsle)
#Lasso
best_rmsle = 1
for i in range(1, 10):
    lasso = linear_model.Lasso(alpha = i*0.1)
    lasso.fit(Xtrain, Ytrain)
    Ypred = lasso.predict(Xtrain)
    if calc_rmsle(Ytrain, Ypred)<best_rmsle:
        best_alpha = i*0.1
        best_rmsle = calc_rmsle(Ytrain, Ypred)

print(best_alpha)
print(best_rmsle)
#Random Forest
best_rmsle = 1
for i in range(1,30):
    forest = RandomForestRegressor(max_depth=i, random_state=0, n_estimators=50)
    forest.fit(Xtrain, Ytrain)
    Ypred = forest.predict(Xtrain)
    if calc_rmsle(Ytrain, Ypred)<best_rmsle:
        best_d = i
        best_rmsle = calc_rmsle(Ytrain, Ypred)        

print(best_d)
print(best_rmsle)
#Random forest foi o melhor método, com depth 26, então ele será usado para predição
forest = RandomForestRegressor(max_depth=26, random_state=0, n_estimators=50)
forest.fit(Xtrain, Ytrain)

Ytest = forest.predict(Xtest)
predictions = pd.DataFrame({"Id":testData.Id, "median_house_value":Ytest})
predictions.to_csv("predictions.csv", index=False)
predictions