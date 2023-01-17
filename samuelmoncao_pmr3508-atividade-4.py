#Bibliotecas para leitura e tratamentos dos dados
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score

#Biblioteca para os classificadores aplicados
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

print(os.listdir("../input/"))
adult = pd.read_csv("../input/train_data.csv",
        sep=',',
        engine='python',
        na_values="?")
adult.shape
#Ao ler o arquivo, tiramos as linhas nulas para que a base de treino seja melhor aproveitada
nadult = adult.dropna()
nadult.describe()
adult["age"].value_counts().plot(kind="bar")
adult["native.country"].value_counts()
adult["hours.per.week"].value_counts().plot(kind="bar")
plt.axis([0,10, 0, 16000])
adult["education"].value_counts().plot(kind="bar")
adult["occupation"].value_counts().plot(kind="pie")
adult["race"].value_counts().plot(kind="pie")
adult["sex"].value_counts().plot(kind="pie")
adult["workclass"].value_counts().plot(kind="barh")
#Consideraremos inicialmente todos os valores numéricos
Xadult_v1 = nadult[["age","fnlwgt","education.num","capital.gain","capital.loss","hours.per.week"]]
Yadult = nadult.income

knn = KNeighborsClassifier(n_neighbors=26)

scores = cross_val_score(knn, Xadult_v1, Yadult, cv=10)
scores.mean()

#Aqui podemos suspeitar, baseado nas análises anteriores, que os valores de 'fnlwgt'
#na verdade não são bons exemplos de teste para o modelo. Se trata de um valor que é
#único na maioria dos casos, como mostrado nos gráficos.
#Agora, transformaremos todas as colunas em número afim de aproveitar melhor as possibilidades estudadas.
numAdult = nadult.apply(preprocessing.LabelEncoder().fit_transform)

Xadult_v3 = numAdult[["age", "workclass", "education", "education.num", "occupation","sex", "capital.gain", "capital.loss","hours.per.week"]]
Yadult = numAdult.income
knn = KNeighborsClassifier(n_neighbors=70)
scores = cross_val_score(knn, Xadult_v3, Yadult, cv=10)

scores.mean()
#Agora aplicaremos o algoritmo AdaBoost para verificar se a estimativa melhoraria, dada as colunas 
#escolhidas.
#"Cross Validation" ainda será utilizado para que o treinamento seja mais efetivo

Xadult_v3 = numAdult[["age", "workclass", "education", "education.num", "occupation","sex", "capital.gain", "capital.loss","hours.per.week"]]
Yadult = numAdult.income
AdaB = AdaBoostClassifier(n_estimators=100)
scores = cross_val_score(AdaB, Xadult_v3, Yadult, cv=10)

scores.mean()
Xadult_v3 = numAdult[["age", "workclass", "education", "education.num", "occupation","sex", "capital.gain", "capital.loss","hours.per.week"]]
Yadult = numAdult.income
random_forest = RandomForestClassifier(n_estimators=50)
scores = cross_val_score(random_forest, Xadult_v3, Yadult, cv=10)

scores.mean()
Xadult_v3 = numAdult[["age", "workclass", "education", "education.num", "occupation","sex", "capital.gain", "capital.loss","hours.per.week"]]
Yadult = numAdult.income
RNA = MLPClassifier()
scores = cross_val_score(RNA, Xadult_v3, Yadult, cv=10)

scores.mean()