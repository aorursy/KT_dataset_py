import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
adult = pd.read_csv("../input/data-adult/train_data.csv", sep=r'\s*,\s*', engine='python', na_values="?")
adult
adult.columns
adult.shape
adult = adult.rename(index = str, columns ={
            'age':"Age",
            'workclass':"Workclass",
            'education':"Education",
            'education.num':"Education-Num",
            'marital.status':"Martial Status",
            'occupation':"Occupation",
            'relationship':"Relationship",
            'race':"Race",
            'sex':"Sex",
            'capital.gain': "Capital Gain",
            'capital.loss':"Capital Loss",
            'hours.per.week':"Hours per week",
            'native.country': "Country",
            'income':"Target"})


#         Renomeei as features para os nomes passados pelo professor
#         "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
#         "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
#         "Hours per week", "Country", "Target"],
adult["Country"].value_counts()
#ver os países que mais apareceram na base
dfadult = adult.dropna()
dfadult.shape
#vou trabalhar agora apenas com o dataframe dfadult sem os dados missing
dfadult.dtypes
dfadult.groupby("Age").describe()
dfadult["Age"].value_counts().plot(kind="bar")
dfadult["Sex"].value_counts().plot(kind="bar")
dfadult["Education"].value_counts().plot(kind="bar")
adult["Occupation"].value_counts().plot(kind="bar")
dfadult.Sex.value_counts()
dfadult.groupby(['Sex', 'Target'])['Id'].count()
dfadult.columns
numAdult = dfadult.apply(preprocessing.LabelEncoder().fit_transform)
dfadultsex = numAdult.Sex*10
Xadult = dfadult[["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]]
Xadult = Xadult.assign(Sex = dfadultsex)
Xadult
Yadult = dfadult.Target
knn = KNeighborsClassifier(n_neighbors=20)
# gera o classificador
scores = cross_val_score(knn, Xadult, Yadult, cv=10)
# faz a validação cruzada do modelo afim de descobrir a acuracia do modelo antes de testa-lo na base de teste
scores.mean()
# media dos valores de cada validação cruzada
knn.fit(Xadult,Yadult)
testadult = pd.read_csv("../input/data-adult/test_data.csv", sep=r'\s*,\s*', engine='python', na_values="?")
testadult = testadult.rename(index = str, columns ={
            'age':"Age",
            'workclass':"Workclass",
            'education':"Education",
            'education.num':"Education-Num",
            'marital.status':"Martial Status",
            'occupation':"Occupation",
            'relationship':"Relationship",
            'race':"Race",
            'sex':"Sex",
            'capital.gain': "Capital Gain",
            'capital.loss':"Capital Loss",
            'hours.per.week':"Hours per week",
            'native.country': "Country"})
numadult = testadult[['Sex','Id']]
testadultsex = numadult.apply(preprocessing.LabelEncoder().fit_transform)
testadultsex = testadultsex.Sex*10
testadultsex
testadult.head()
Xtadult = testadult[["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]]
Xtadult = Xtadult.assign(Sex = testadultsex)
YtestPred = knn.predict(Xtadult)
FinalTestAdult = Xtadult.assign(income = YtestPred)
FinalTestAdult
FinalTestAdult = FinalTestAdult.join(testadult, lsuffix='_caller', rsuffix='_other')
FinalTestAdult = FinalTestAdult[["Id", "income"]]
FinalTestAdult.to_csv("submission.csv", index = False, header = True)