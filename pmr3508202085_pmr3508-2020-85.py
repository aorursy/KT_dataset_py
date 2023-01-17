import pandas as pd

import sklearn as sk

import numpy as np

data = pd.read_csv("../input/adult-pmr3508/train_data.csv",sep=r'\s*,\s*', engine='python', na_values="?")
test = datatest = pd.read_csv("../input/adult-pmr3508/test_data.csv", sep=r'\s*,\s*', engine='python', na_values="?")
data.head()
d_faltante = data.isna().sum()

print(d_faltante)
print("workclass:")

print(data["workclass"].describe())

print("\n")



print("occupation:")

print(data["occupation"].describe())

print("\n")



print("native.country:")

print(data["native.country"].describe())

print("\n")
data["workclass"] = data["workclass"].fillna("Private")



data["occupation"] = data["occupation"].fillna("Prof-specialty")



data["native.country"] = data["native.country"].fillna("United-States")

d_faltante = data.isnull().sum()

print(d_faltante)
t_faltante = test.isnull().sum()

print(t_faltante)
print("workclass:")

print(test["workclass"].describe())

print("\n")



print("occupation:")

print(test["occupation"].describe())

print("\n")



print("native.country:")

print(test["native.country"].describe())

print("\n")
test["workclass"] = test["workclass"].fillna("Private")



test["occupation"] = test["occupation"].fillna("Prof-specialty")



test["native.country"] = test["native.country"].fillna("United-States")
t_faltante = test.isnull().sum()

print(t_faltante)
import matplotlib.pyplot as plt

maior = data[data['income'] == ">50K"]

menor = data[data['income'] == "<=50K"]
grafico = pd.concat([maior["age"].value_counts(), menor["age"].value_counts()], keys=[">50K", "<=50K"], axis=1)

grafico.plot(kind='bar', figsize=(20, 10))
grafico = pd.concat([maior["workclass"].value_counts(), menor["workclass"].value_counts()], keys=[">50K", "<=50K"], axis=1)

grafico.plot(kind='bar')
grafico = pd.concat([maior["marital.status"].value_counts(), menor["marital.status"].value_counts()], keys=[">50K", "<=50K"], axis=1)

grafico.plot(kind='bar')
grafico = pd.concat([maior["occupation"].value_counts(), menor["occupation"].value_counts()], keys=[">50K", "<=50K"], axis=1)

grafico.plot(kind='bar')
grafico = pd.concat([maior["relationship"].value_counts(), menor["relationship"].value_counts()], keys=[">50K", "<=50K"], axis=1)

grafico.plot(kind='bar')
grafico = pd.concat([maior["race"].value_counts(), menor["race"].value_counts()], keys=[">50K", "<=50K"], axis=1)

grafico.plot(kind='bar')
grafico = pd.concat([maior["sex"].value_counts(), menor["sex"].value_counts()], keys=[">50K", "<=50K"], axis=1)

grafico.plot(kind='bar')
grafico = pd.concat([maior["capital.gain"].value_counts(), menor["capital.gain"].value_counts()], keys=[">50K", "<=50K"], axis=1)

grafico.plot(kind='bar',figsize=(20, 10))
grafico = pd.concat([maior["capital.loss"].value_counts(), menor["capital.loss"].value_counts()], keys=[">50K", "<=50K"], axis=1)

grafico.plot(kind='bar',figsize=(20, 10))
grafico = pd.concat([maior["hours.per.week"].value_counts(), menor["hours.per.week"].value_counts()], keys=[">50K", "<=50K"], axis=1)

grafico.plot(kind='bar',figsize=(20, 10))
grafico = pd.concat([maior["native.country"].value_counts(), menor["native.country"].value_counts()], keys=[">50K", "<=50K"], axis=1)

grafico.plot(kind='bar',figsize=(20, 10))
from sklearn import preprocessing
numData = data.apply(preprocessing.LabelEncoder().fit_transform)

numTest = test.apply(preprocessing.LabelEncoder().fit_transform)
for col in numTest:

    normData = np.linalg.norm(numData[col])

    numData[col] = numData[col]/normData

    normTest = np.linalg.norm(numTest[col])

    numTest[col] = numTest[col]/normData
numData.head()
XData = numData[["age", "workclass", "education.num", "marital.status",

        "occupation", "relationship", "race", "sex", "capital.gain", "capital.loss",

        "hours.per.week", "native.country"]]

YData = numData.income

XTest = numTest[["age", "workclass", "education.num", "marital.status",

        "occupation", "relationship", "race", "sex", "capital.gain", "capital.loss",

        "hours.per.week", "native.country"]]
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score
BestResult = 0

k = 0

knn = None

for n in range(20,40):

    aux = KNeighborsClassifier(n, algorithm = "brute")

    score = cross_val_score(aux, XData, YData, cv=10)

    if score.mean() > BestResult:

        BestResult = score.mean()

        k = n

        knn = aux

print(k)

print(BestResult)
knn.fit(XData, YData)
YTest = knn.predict(XTest)
Resultado = ["<=50K" if x== 0 else ">50K" for x in YTest]
s = "Id,income\n"

for k,income in enumerate(Resultado):

    s += f"{k},{income}\n"

with open(f"PMR3508-2020-85.csv", "w+") as f:

    f.write(s)