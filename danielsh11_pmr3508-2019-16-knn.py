import pandas as pd

import sklearn

import matplotlib.pyplot as plt
adult = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv", names=["Age","Workclass","fnlwgt","Education","Education-Num","Martial Status","Occupation","Relationship","Race","Sex","Capital Gain","Capital Loss","Hour per week","Country","Target"],sep=r'\s*,\s*',engine='python',skiprows=1,na_values="?")
testadult = pd.read_csv("/kaggle/input/adult-pmr3508/test_data.csv", names=["Age","Workclass","fnlwgt","Education","Education-Num","Martial Status","Occupation","Relationship","Race","Sex","Capital Gain","Capital Loss","Hour per week","Country","Target"],sep=r'\s*,\s*',engine='python',skiprows=1,na_values="?")
adult.shape
adult.head() #Comando para mostrar uma tabela sample de adult

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



Xadult = numadult[["Age","Education-Num","Capital Gain","Capital Loss","Hour per week"]]

Yadult = numadult.Target

Xtestadult = numtestadult[["Age","Education-Num","Capital Gain","Capital Loss","Hour per week"]]

Ytestadult = numtestadult.Target
mediascores = []

for i in list(range(25,30)): #Testado k de 25 à 39

    knn = KNeighborsClassifier(n_neighbors=i) #Criando objeto knn, com n de vizinhos como parametro

    scores = cross_val_score(knn,Xadult,Yadult,cv=10) #Validação cruzada, com parametro de nº de grupos em que os dados serão divididos



    soma=0

    for j in scores:

        soma=soma+j

    media = soma/10

    mediascores.append([i,media])
print(mediascores)
numadult = nadult.apply(preprocessing.LabelEncoder().fit_transform) #Preprocessamento para transformar dados não-numericos em dados numericos

numtestadult = ntestadult.apply(preprocessing.LabelEncoder().fit_transform)



Xadult = numadult[["Age","Workclass","fnlwgt","Education","Education-Num","Martial Status","Occupation","Relationship","Race","Sex","Capital Gain","Capital Loss","Hour per week","Country"]]

Yadult = numadult.Target

Xtestadult = numtestadult[["Age","Workclass","fnlwgt","Education","Education-Num","Martial Status","Occupation","Relationship","Race","Sex","Capital Gain","Capital Loss","Hour per week","Country"]]

Ytestadult = numtestadult.Target
mediascores = []

for i in list(range(25,30)): #Testado k de 25 à 39

    knn = KNeighborsClassifier(n_neighbors=i) #Criando objeto knn, com n de vizinhos como parametro

    scores = cross_val_score(knn,Xadult,Yadult,cv=10) #Validação cruzada, com parametro de nº de grupos em que os dados serão divididos



    soma=0

    for j in scores:

        soma=soma+j

    media = soma/10

    mediascores.append([i,media])
print(mediascores)
numadult = nadult.apply(preprocessing.LabelEncoder().fit_transform) #Preprocessamento para transformar dados não-numericos em dados numericos

numtestadult = ntestadult.apply(preprocessing.LabelEncoder().fit_transform)



Xadult = numadult[["Age","Workclass","Education-Num","Martial Status","Occupation","Relationship","Race","Sex","Capital Gain","Capital Loss","Hour per week","Country"]]

Yadult = numadult.Target

Xtestadult = numtestadult[["Age","Workclass","Education-Num","Martial Status","Occupation","Relationship","Race","Sex","Capital Gain","Capital Loss","Hour per week","Country"]]

Ytestadult = numtestadult.Target
mediascores = []

for i in list(range(25,30)): #Testado k de 25 à 39

    knn = KNeighborsClassifier(n_neighbors=i) #Criando objeto knn, com n de vizinhos como parametro

    scores = cross_val_score(knn,Xadult,Yadult,cv=10) #Validação cruzada, com parametro de nº de grupos em que os dados serão divididos



    soma=0

    for j in scores:

        soma=soma+j

    media = soma/10

    mediascores.append([i,media])
print(mediascores)
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
#criar arquivo de resultado

savepath = "predictions.csv"

prev = pd.DataFrame(pred, columns = ["income"])

prev.to_csv(savepath, index_label="Id")

prev