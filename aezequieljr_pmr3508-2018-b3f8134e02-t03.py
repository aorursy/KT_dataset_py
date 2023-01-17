# Importando o que utilizaremos no programa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,fbeta_score
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
trainDb = pd.read_csv("../input/mycalifdb/train.csv")
testDb = pd.read_csv("../input/mycalifdb/test.csv")
trainDb.head()
trainDb.info()
def RMSLE(Y, Y_hat):
    sumError = 0.0
    assert len(Y) == len(Y_hat)
    for i in range(len(Y)): 
        if Y[i] < 0: Y[i] = 0 # Se o método supôs um valor negativo, considero que ele quis dizer que a casa era 'de graça'
        sumError += (np.log(Y[i] + 1) - np.log(Y_hat[i] + 1))**2
    return np.sqrt(sumError/len(Y))
Xtrain = trainDb.drop("Id", axis = 1).drop("median_house_value", axis = 1)
Xtest = testDb.drop("Id", axis = 1)
Ytrain = trainDb["median_house_value"]
kNNScores = []
kList = [3, 5, 8, 10, 14, 18, 22, 25, 30]

for i in kList:
    kNNMethod = KNeighborsRegressor(n_neighbors=i)
    scores = cross_val_score(kNNMethod, Xtrain, Ytrain, cv=3)
    kNNScores.append(scores.mean())

plt.title("Escolha do número de vizinhos (k)")
plt.xlabel("Número de vizinhos (k)")
plt.ylabel("Pontuação obtida")
plt.plot(kList, kNNScores, color = 'blue', label = 'kNN')
kNNMethod = KNeighborsRegressor(n_neighbors=10)
kNNMethod.fit(Xtrain, Ytrain)
scores = cross_val_score(kNNMethod, Xtrain, Ytrain, cv=3)
Y = kNNMethod.predict(Xtrain)
print('Pontuação média kNN: ' + str(scores.mean()))
print('RMSLE kNN: ' + str(RMSLE(Y, Ytrain)))
lassoScores = []
alphaList = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1]

for i in alphaList:
    lassoMethod = linear_model.Lasso(alpha = i)
    scores = cross_val_score(lassoMethod, Xtrain, Ytrain, cv=3)
    lassoScores.append(scores.mean())

plt.title("Escolha do alfa no método Lasso")
plt.xlabel("Alpha")
plt.ylabel("Pontuação obtida")
plt.scatter(alphaList, lassoScores, color = 'blue', label = 'Lasso')
lassoMethod = linear_model.Lasso(alpha = 0.1)
lassoMethod.fit(Xtrain, Ytrain)
scores = cross_val_score(lassoMethod, Xtrain, Ytrain, cv=3)
Y = lassoMethod.predict(Xtrain)
print('Pontuação média Lasso: ' + str(scores.mean()))
print('RMSLE Lasso: ' + str(RMSLE(Y, Ytrain)))
forestScores = []
depthList = [3, 5, 8, 12, 18]

for i in depthList:
    forestMethod = RandomForestRegressor(max_depth=i, random_state=0, n_estimators=100)
    scores = cross_val_score(forestMethod, Xtrain, Ytrain, cv=3)
    forestScores.append(scores.mean())

plt.title("Escolha da profundidade do Random Forest")
plt.xlabel("max_depth")
plt.ylabel("Pontuação obtida")
plt.scatter(depthList, forestScores, color = 'blue', label = 'Forest')
forestMethod = RandomForestRegressor(max_depth=25, random_state=0, n_estimators=100)
forestMethod.fit(Xtrain, Ytrain)
scores = cross_val_score(forestMethod, Xtrain, Ytrain, cv=3)
Y = forestMethod.predict(Xtrain)
print('Pontuação média Forest: ' + str(scores.mean()))
print('RMSLE Forest: ' + str(RMSLE(Y, Ytrain)))
forestMethod = RandomForestRegressor(max_depth=25, random_state=0, n_estimators=100)
forestMethod.fit(Xtrain, Ytrain)
Ytest = forestMethod.predict(Xtest)
predictionTest = pd.DataFrame({"Id":testDb.Id, "median_house_value":Ytest})
predictionTest.to_csv("predictTable.csv", index=False)
predictionTest