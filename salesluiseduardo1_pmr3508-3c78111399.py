import pandas as pd
import sklearn
import os
adult = pd.read_csv("../input/adult-base/adult.data.txt",
        names=[
        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country", "Target"],
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
adult.shape #número de colunas e linhas
adult.head() 
adult["Country"].value_counts() # mostrar o número de pessoas por cada nacionalidade
import matplotlib.pyplot as plt
adult["Age"].value_counts().plot(kind="bar") #gráfico com as idades e número de pessoas
plt.rcParams['figure.figsize'] = (12,12)
adult["Sex"].value_counts().plot(kind="bar")
adult["Education"].value_counts().plot(kind="bar")
nadult = adult.dropna()# retirando linhas com dados faltantes
nadult
testAdult = pd.read_csv("../input/test-datacsv/test_data.csv",
        names=[
        "Id","Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country"],
        sep=r'\s*,\s*',
        skiprows=1,
        engine='python',
        na_values="?")
#criando objeto com a base de teste
testAdult
testAdult.shape
nTestAdult = testAdult.dropna()
nTestAdult.shape
nTestAdult
from sklearn import preprocessing
numAdult = nadult.apply(preprocessing.LabelEncoder().fit_transform) # transformando os dados para valores numéricos
numTestAdult = nTestAdult.apply(preprocessing.LabelEncoder().fit_transform)
Xadult = numAdult[["Age", "Workclass", "Education-Num", 
        "Occupation", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week"]]
Xadult
Yadult = nadult.Target
Yadult
XtestAdult = numTestAdult[["Age", "Workclass", "Education-Num", 
        "Occupation", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week"]]
XtestAdult # from sklearn import preprocessingdados para predizer o knn
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=40)

from sklearn.model_selection import cross_val_score

scores = cross_val_score(knn, Xadult, Yadult, cv=10)

scores

knn.fit(Xadult,Yadult)

YtestPred = knn.predict(XtestAdult)

YtestPred