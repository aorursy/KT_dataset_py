#Importando bibliotecas necessárias

import pandas as pd

import sklearn



from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score
# Encontrando o endereço dos arquivos com as bases

import os

cwd = "/kaggle/input/adult-pmr3508/"

testDataPath = cwd + "/test_data.csv"

trainDataPath = cwd + "/train_data.csv"



# Lendo os arquivos csv com as bases de teste e treino e limpando as bases

headers = [

            "Age", "Workclass", "fnlwgt", "Education",

            "Education-Num", "Martial Status",

            "Occupation", "Relationship", "Race", "Sex",

            "Capital Gain", "Capital Loss",

            "Hours per week", "Country", "Target"

        ]



testData = pd.read_csv(

    testDataPath,

    names=headers[:-1],

    sep=r'\s*,\s*',

    engine='python',

    na_values="?",

    skiprows=1

).dropna()



trainData = pd.read_csv(

    trainDataPath,

    names=headers,

    sep=r'\s*,\s*',

    engine='python',

    na_values="?",

    skiprows=1

).dropna()
# Checando se os arquivos foram corretamente importados

print("TestData Shape: ", testData.shape)

print("TrainData Shape: ", trainData.shape)
# Separando dados da base para construção do modelo de treino

trainX = trainData[[

    "Age",

    "Education-Num",

    "Capital Gain",

    "Capital Loss",

    "Hours per week"

]]



trainY = trainData.Target



# Separando dados do arquivo de teste para fazer a predição

testX = testData[[

    "Age",

    "Education-Num",

    "Capital Gain",

    "Capital Loss",

    "Hours per week"

]]
# Criando o classificador KNN e gerando modelo

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(trainX, trainY)
# Predizendo o resultado

testY_pred = knn.predict(testX)



testY_pred
# Exportando predições para um CSV

predictionList = pd.DataFrame(testData.index)

predictionList["income"] = testY_pred

predictionList.to_csv("prediction.csv", index=False)