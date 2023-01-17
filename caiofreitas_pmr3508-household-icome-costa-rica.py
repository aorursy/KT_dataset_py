import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import pandas as pd # para Processamento de Dados
import matplotlib.pyplot as plt #Para plotagem de gráficos
import numpy as np # Algebra Linear
household = pd.read_csv("../input/poverty/train.csv", index_col = 'Id', na_values=("NaN", " "))
household.shape
household
trainX = household[["rooms", "refrig", "escolari", "computer", "television", "mobilephone", "SQBescolari", "SQBdependency"]]
trainY = household.Target
trainX
scoresarray = np.empty(100, dtype=float, order='C')
i = 1
while i<100:
    kNN = KNeighborsClassifier(n_neighbors=i)
    scores = cross_val_score(kNN, trainX, trainY, cv=10)
    mean = scores.mean()
    scoresarray[i] = mean
    i = i + 1
scoresarray
scoresarray.max()
scoresarray.argmax()
kNN = KNeighborsClassifier(n_neighbors=71)
kNN.fit(trainX, trainY)
testX = pd.read_csv("../input/poverty/test.csv", index_col = 'Id', na_values=("NaN", " "))
testX = testX[["rooms", "refrig", "escolari", "computer", "television", "mobilephone", "SQBescolari", "SQBdependency"]]
trainY = household.Target
predictions = kNN.predict(testX)
predict = pd.DataFrame(index = testX.index)
predict["Target"] = predictions
predict["Target"].value_counts().plot.bar()

predict.to_csv("predictions.csv")

