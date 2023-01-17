#Imports
import pandas as pd
import numpy as np
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing

#Base
adult = pd.read_csv("../input/data-ep/train_data.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
adult.head()
import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
plt.subplot(121)
adult["sex"].value_counts().plot(kind="bar")
t = plt.title("Histograma sexos")
plt.subplot(122)
plt.hist(adult["age"])
t = plt.title("Histograma idades")
plt.show()
print("Horas de trabalho")
plt.figure(figsize=(10,5))
plt.subplot(121)
plt.hist(adult.loc[adult["income"]==">50K"]["hours.per.week"])
plt.title("Histograma de horas de trabalho\nindividuos com renda >50K")
plt.subplot(122)
plt.hist(adult.loc[adult["income"]=="<=50K"]["hours.per.week"])
t=plt.title("Histograma de horas de trabalho\nindividuos com renda <=50K")
plt.show()
print("Anos de estudo")
plt.figure(figsize=(10,5))
plt.subplot(121)
plt.hist(adult.loc[adult["income"]==">50K"]["education.num"])
plt.xticks(np.arange(0,18,2))
plt.title("Histograma de anos de estudo\nindividuos com renda >50K")
plt.subplot(122)
plt.hist(adult.loc[adult["income"]=="<=50K"]["education.num"],np.arange(0,18,2))
plt.xticks(np.arange(0,18,2))
t=plt.title("Histograma de anos de estudo\nindividuos com renda <=50K")
plt.show()
print("Nacionalidades")
print("Nativos dos EUA: ", len(adult.loc[adult["native.country"]=="United-States"]))
print("Nativos do Mexico: ", len(adult.loc[adult["native.country"]=="Mexico"]))
print("Nativos de outros países: ", len(adult.loc[(adult["native.country"] !="Mexico") & (adult["native.country"]!="United-States")] ))


nadult = adult.dropna()
#Selecionando algumas colunas numéricas da base para um primeiro teste

# Colunas da Adult:
#Id,age,workclass,fnlwgt,education,education.num,marital.status,occupation,
#relationship,race,sex,capital.gain,capital.loss,hours.per.week,native.country,income

Xadult = nadult[["age","education.num","capital.gain", "capital.loss", "hours.per.week"]]
Yadult = np.ravel(nadult[["income"]])

#Instanciando o classificador, com K = 3
knn = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(knn, Xadult, Yadult, cv=5)
print("O resultado da validacao cruzada foi (media): ", scores.mean())

numAdult = nadult.apply(preprocessing.LabelEncoder().fit_transform)
Xadult = numAdult[["age","education.num","capital.gain", "capital.loss", "hours.per.week", "sex"]]
Yadult = np.ravel(nadult[["income"]])
scores = cross_val_score(knn, Xadult, Yadult, cv=5)
print("O resultado da validacao cruzada foi (media): ", scores.mean())

numAdult = nadult.apply(preprocessing.LabelEncoder().fit_transform)
Xadult = numAdult[["age","education.num","capital.gain", "capital.loss", "hours.per.week", "sex", "marital.status"]]
Yadult = np.ravel(nadult[["income"]])
scores = cross_val_score(knn, Xadult, Yadult, cv=5)
print("O resultado da validacao cruzada foi (media): ", scores.mean())

numAdult = nadult.apply(preprocessing.LabelEncoder().fit_transform)
Xadult = numAdult[["age","education.num","capital.gain", "capital.loss", "hours.per.week", "sex", "marital.status", "relationship"]]
Yadult = np.ravel(nadult[["income"]])
scores = cross_val_score(knn, Xadult, Yadult, cv=5)
print("O resultado da validacao cruzada foi (media): ", scores.mean())
score_list = []
numAdult = nadult.apply(preprocessing.LabelEncoder().fit_transform)
Xadult = numAdult[["age","education.num","capital.gain", "capital.loss", "hours.per.week", "sex", "marital.status", "relationship"]]
Yadult = np.ravel(nadult[["income"]])

for k in range(1,50):
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, Xadult, Yadult, cv=5)
    score_list.append(scores.mean())
    
import matplotlib.pyplot as plt
plt.plot(score_list)
plt.title("Gráfico Score X num_neighbors")
plt.xlabel("Num Neighbors")
plt.ylabel("Score")
numAdult = nadult.apply(preprocessing.LabelEncoder().fit_transform)
Xadult = numAdult[["age","education.num","capital.gain", "capital.loss", "hours.per.week", "sex", "marital.status", "relationship"]]
Yadult = np.ravel(nadult[["income"]])
knn = KNeighborsClassifier(n_neighbors=25)
scores = cross_val_score(knn, Xadult, Yadult, cv=5)
print("O resultado da validacao cruzada foi (media): ", scores.mean())
test_adult = pd.read_csv("../input/data-ep/test_data.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
ntest_adult = test_adult.dropna()
num_test_adult = ntest_adult.apply(preprocessing.LabelEncoder().fit_transform)
Xtest_adult = num_test_adult[["age","education.num","capital.gain", "capital.loss", "hours.per.week", "sex", "marital.status", "relationship"]]
knn = KNeighborsClassifier(n_neighbors=25)
knn.fit(Xadult,Yadult)
Y_test_pred = knn.predict(Xtest_adult)

result = np.vstack((ntest_adult["Id"],Y_test_pred)).T

result
