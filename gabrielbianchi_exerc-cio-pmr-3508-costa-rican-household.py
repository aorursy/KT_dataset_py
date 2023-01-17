#Carregando a base de dados
import pandas as pd
import numpy as np
data = pd.read_csv("../input/train.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")

data.head()
data.shape

import matplotlib.pyplot as plt
plt.figure(figsize=(15,5))
plt.subplot(121)
data.Target.value_counts().plot(kind="bar")
plt.ylabel("Número de famílias")
plt.xlabel("Classes")
t=plt.title("Distribuição das classes na base de dados")
plt.subplot(122)
(data.Target.value_counts()*100/data.Target.value_counts().sum()).plot(kind="bar")
plt.ylabel("%")
plt.xlabel("Classes")
t=plt.title("Distribuição porcentual das classes na base de dados")

ndata = data.dropna()
ndata.shape
print("Colunas que possuem missing data:")
print(data.columns[data.isnull().any()])
#Removendo essas colunas
dropped_data = data.drop(labels=['v2a1', 'v18q1', 'rez_esc', 'meaneduc', 'SQBmeaned'], axis=1)
dropped_data.shape
numeric_data = dropped_data.select_dtypes(include=[np.number])
print("(Linhas, Nro. Features) = ", numeric_data.shape)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

knn = KNeighborsClassifier(n_neighbors=100)
scores = cross_val_score(knn, numeric_data, numeric_data.Target, cv=5)
print("O resultado da validacao cruzada foi (media): ", scores.mean())
score_list = []
k_list = np.arange(1,501,10)
for k in k_list:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, numeric_data, numeric_data.Target, cv=5)
    score_list.append(scores.mean())



import matplotlib.pyplot as plt
k_list = np.arange(1,501,10)
plt.plot(k_list,score_list)
plt.xlabel("num_neighbors")
plt.title("Score X num_neighbors")
print("O valor máximo de score é: ", np.stack(score_list).max(), " e o valor de K correspondente é :", k_list[np.argmax(np.stack(score_list))])
from sklearn import preprocessing
encoded_data = dropped_data.apply(preprocessing.LabelEncoder().fit_transform)
knn = KNeighborsClassifier(n_neighbors=131)
scores = cross_val_score(knn, encoded_data, encoded_data.Target, cv=5)
print("O resultado da validacao cruzada foi (media): ", scores.mean())
from sklearn.feature_selection import SelectKBest
sel = SelectKBest(k=50)
selected = sel.fit_transform(encoded_data.loc[:,encoded_data.columns != "Target"], encoded_data.Target)
knn = KNeighborsClassifier(n_neighbors=131)
scores = cross_val_score(knn, selected, encoded_data.Target, cv=5)
print("O resultado da validacao cruzada foi (media): ", scores.mean())
