import pandas as pd

import numpy as np

import sklearn

import matplotlib.pyplot as plt
adult = pd.read_csv("../input/adult-pmr3508/train_data.csv", names=["Age","Workclass","fnlwgt","Education","Education-Num","Martial Status","Occupation","Relationship","Race","Sex","Capital Gain","Capital Loss","Hour per week","Country","Target"],sep=r'\s*,\s*',engine='python',skiprows=1,na_values="?")
testadult = pd.read_csv("../input/adult-pmr3508/test_data.csv", names=["Age","Workclass","fnlwgt","Education","Education-Num","Martial Status","Occupation","Relationship","Race","Sex","Capital Gain","Capital Loss","Hour per week","Country","Target"],sep=r'\s*,\s*',engine='python',skiprows=1,na_values="?")
adult.shape
adult.head() #Comando para mostrar uma tabela sample de adult

adult.shape #Comando para mostrar o tamanho da variavel adult

adult["Country"].value_counts() #Conta o nº de incidencias dos dados, da coluna Country

adult["Age"].value_counts().plot(kind="bar") #Plota os dados em grafico de barras

adult["Occupation"].value_counts().plot(kind="bar")

adult["Education"].value_counts().plot(kind="bar")
adult["Country"].value_counts()
adult["Age"].value_counts().plot(kind="bar")
adult["Occupation"].value_counts().plot(kind="bar")
adult["Education"].value_counts().plot(kind="bar")
adult["Relationship"].value_counts().plot(kind="bar")
nadult = adult.dropna() #Comando para retirar as linhas com dados faltantes

ntestadult = testadult.dropna()

from sklearn.neighbors import KNeighborsClassifier #Importando classificador KNN

from sklearn.model_selection import cross_val_score #Importando teste de validação cruzada

from sklearn.metrics import accuracy_score #Importando teste de acurácia da base de teste

from sklearn import preprocessing  #importando preprocessamento de dados, para dados strings, como "Country" e "Martial Status"
numadult = nadult.apply(preprocessing.LabelEncoder().fit_transform) #Preprocessamento para transformar dados não-numericos em dados numericos

numtestadult = ntestadult.apply(preprocessing.LabelEncoder().fit_transform)



Xadult = numadult[["Age","Workclass","Education-Num","Martial Status","Occupation","Relationship","Race","Sex","Capital Gain","Capital Loss","Hour per week","Country"]]

Yadult = numadult.Target

Xtestadult = numtestadult[["Age","Workclass","Education-Num","Martial Status","Occupation","Relationship","Race","Sex","Capital Gain","Capital Loss","Hour per week","Country"]]

Ytestadult = numtestadult.Target
knn = KNeighborsClassifier(n_neighbors=28)
scores = cross_val_score(knn,Xadult,Yadult,cv=10)
print(scores)
knn.fit(Xadult,Yadult) #Treinando o classificador com os dados de treino

Ytestpred = knn.predict(Xtestadult) #Prevendo rotulos da base de teste, usando o classificador
pred = []; #lista utilizada para passar os dados de YtestPred, que estão escritos em 0 ou 1, para <=50K e >50K

for i in range(len(Ytestpred)-1):

    pred.append(0)

    if Ytestpred[i] == 0:

        pred[i] = "<=50K"

    elif Ytestpred[i] == 1:

        pred[i] = ">50K"

pred
from sklearn.neural_network import MLPClassifier #importa o modelo.

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score #importa métricas de avaliação

mlp_model = MLPClassifier() #cria o modelo. Os Hiperparâmetros dos classificador serão o default

print(mlp_model)

mlp_model.fit(Xadult, Yadult) #treina o modelo

YtestpredMLP = mlp_model.predict(Xtestadult) #predição do conjunto de testes
print("-------Confusion Matrix------- \n", confusion_matrix(Ytestpred, YtestpredMLP))

print('-----Classification Report----- \n', classification_report(Ytestpred, YtestpredMLP))

print('Accuracy Score of MLP, compared to KNN: ', accuracy_score(Ytestpred, YtestpredMLP))
print(accuracy_score(Ytestpred, YtestpredMLP))
from sklearn.ensemble import GradientBoostingClassifier
lr_list = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]



for learning_rate in lr_list:

    gbc_model = GradientBoostingClassifier(n_estimators=20, learning_rate=learning_rate, random_state=0)

    gbc_model.fit(Xadult, Yadult)

    

    print("Learning Rate: ", learning_rate)

    print("Accuracy Score (training): {0:.3F}".format(gbc_model.score(Xadult, Yadult)))

    print("Accuracy Score (validation with KNN): {0:.3F}".format(gbc_model.score(Xtestadult, Ytestpred)))

    print("\n")

gbc_model2 = GradientBoostingClassifier(n_estimators=20, learning_rate=0.05, random_state=0)

gbc_model2.fit(Xadult, Yadult)

YtestpredGBC = gbc_model2.predict(Xtestadult)
print("-------Confusion Matrix------- \n", confusion_matrix(Ytestpred, YtestpredGBC))
print('-----Classification Report----- \n', classification_report(Ytestpred, YtestpredGBC))

print('Accuracy Score of MLP, compared to KNN: ', accuracy_score(Ytestpred, YtestpredGBC))
print("-------Confusion Matrix------- \n", confusion_matrix(YtestpredMLP, YtestpredGBC))
print('-----Classification Report----- \n', classification_report(YtestpredMLP, YtestpredGBC))

print('Accuracy Score of MLP, compared to KNN: ', accuracy_score(YtestpredMLP, YtestpredGBC))