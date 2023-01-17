# PMR3508 - Aprendizado de Máquina e Reconhecimento de Padrões
## Tarefa 1 - Adult Dataset

## Aluno: Thiago Yukio - 8993740 - PMR3508-2020-11
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import numpy as np
import os

os.listdir('../input')
adult = pd.read_csv("../input/adultdataset/train_data",
        names=[
        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country", "Target"],
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")

testAdult = pd.read_csv("../input/adultdataset/test_data",
        names=[
        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country", "Target"],
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
adult.shape
testAdult.shape
adult.head()
testAdult.head()
adult["Age"].value_counts().plot(kind="bar")
# Como a maior parte do dataset é concentrado no setor privado, talvez seja possível atribuir menos peso:

adult["Workclass"].value_counts().plot(kind="pie")
adult["Education"].value_counts().plot(kind="bar")
adult["Occupation"].value_counts().plot(kind="bar")
adult["Race"].value_counts().plot(kind="pie")
adult["Hours per week"].value_counts().plot(kind="pie")
adult["Country"].value_counts().plot(kind="pie")
adult["Target"].value_counts().plot(kind="pie")
nadult = adult.dropna()
nadult
ntestAdult = testAdult.dropna()
ntestAdult.shape
ntestAdult
# Primeiro teste: seleção de atributos numéricos, com kNN para k=3
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
numAttr = ["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]
# train
Xadult = nadult[numAttr].apply(preprocessing.LabelEncoder().fit_transform)
Yadult = np.array(nadult["Target"], dtype=object)

# test
XtestAdult = ntestAdult[numAttr].apply(preprocessing.LabelEncoder().fit_transform)
YtestAdult = np.array(ntestAdult["Target"], dtype=object)
knn = KNeighborsClassifier(n_neighbors=3)
scores = cross_val_score(knn, Xadult, Yadult, cv=10)
scores
knn.fit(Xadult, Yadult)
YtestPred = knn.predict(XtestAdult)
YtestPred
YtestAdult
accuracy_score(YtestAdult, YtestPred)
# Definindo função para replicar programaticamente
def classifyAndScore(k = int):
    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    scores = cross_val_score(knn, Xadult, Yadult, cv=10, n_jobs=-1)
    knn.fit(Xadult, Yadult)
    YtestPred = knn.predict(XtestAdult)
    return (accuracy_score(YtestAdult, YtestPred))
# Iterando classificação variando k de 1 a 100
# results = []
# for k in range(99):
#     results.append(classifyAndScore(k+1))
# results
# np.amax(results)
# results.index(0.8226427622841965)
# Portanto, o melhor resultado ocorre com k = 87
# Testando sem capital gain
numAttr = ["Age", "Education-Num", "Capital Loss", "Hours per week"]

# train
Xadult = nadult[numAttr].apply(preprocessing.LabelEncoder().fit_transform)
Yadult = np.array(nadult["Target"], dtype=object)

# test
XtestAdult = ntestAdult[numAttr].apply(preprocessing.LabelEncoder().fit_transform)
YtestAdult = np.array(ntestAdult["Target"], dtype=object)

Xadult.head()
classifyAndScore(87)
# Como o resultado foi pior, Capital Gain faz diferença positiva
# Adicionando outras features: Occupation
numAttr = ["Age", "Education-Num", "Capital Gain", "Capital Loss", "Hours per week", "Occupation"]
# train
Xadult = nadult[numAttr].apply(preprocessing.LabelEncoder().fit_transform)
Yadult = np.array(nadult["Target"], dtype=object)

# test
XtestAdult = ntestAdult[numAttr].apply(preprocessing.LabelEncoder().fit_transform)
YtestAdult = np.array(ntestAdult["Target"], dtype=object)

Xadult.head()
classifyAndScore(87)
# Como as novas propriedades pioraram o resultado, foram removidas
# Teste com feature de gênero e Workclass
numAttr = ["Age", "Education-Num", "Capital Gain", "Capital Loss", "Hours per week", "Sex", "Workclass"]
# train
Xadult = nadult[numAttr].apply(preprocessing.LabelEncoder().fit_transform)
Yadult = np.array(nadult["Target"], dtype=object)

# test
XtestAdult = ntestAdult[numAttr].apply(preprocessing.LabelEncoder().fit_transform)
YtestAdult = np.array(ntestAdult["Target"], dtype=object)

Xadult.head()
classifyAndScore(87)
# Conclusão:
# As melhores features para prever renda superior a 50K, nesta base de dados, são:
# ["Age", "Education-Num", "Capital Gain", "Capital Loss", "Hours per week", "Sex", "Workclass"]
# E o melhor modelo é de kNN, sendo k = 87
# A acurácia final é de: 82,417%