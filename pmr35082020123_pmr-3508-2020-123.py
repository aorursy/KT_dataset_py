import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

import sklearn

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

from sklearn import preprocessing

from sklearn.metrics import accuracy_score
data = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv", index_col=['Id'], na_values="?")

test_data = pd.read_csv("/kaggle/input/adult-pmr3508/test_data.csv", index_col=['Id'], na_values="?")
print("formato da tabela de dados")

data.shape
data.head()
data["occupation"].value_counts().plot(kind="bar")
data["sex"].value_counts().plot(kind="pie")
data["education"].value_counts().plot(kind="bar")
data["race"].value_counts().plot(kind="pie")
data["age"].value_counts().plot(kind="bar")
max_miss = 0

for column in data.columns:

    if data[column].isnull().sum() > max_miss:

    	max_miss = data[column].isnull().sum()

max_miss = 100*max_miss/data.shape[0]	

print(str(max_miss) + "%")
cleandata = data.dropna()
xdata = cleandata[["age","education.num","capital.gain", "capital.loss", "hours.per.week"]]

ydata = cleandata["income"]
kteste = [1, 5, 10, 15, 20, 30, 35]

resultados = []

for k in kteste:

	knn = KNeighborsClassifier(n_neighbors=k)

	score_cross = cross_val_score(knn, xdata, ydata, cv = 10)

	resultados.append(score_cross.mean())
print("k            resultado[%]")

for i in range(len(kteste)):

    if kteste[i] > 9:

        print(str(kteste[i]) + "          " + str(resultados[i]*100))

    else:

        print(str(kteste[i]) + "           " + str(resultados[i]*100))
classifier = KNeighborsClassifier(n_neighbors=30)

classifier.fit(xdata, ydata)
xtest = test_data[["age","education.num","capital.gain", "capital.loss", "hours.per.week"]]

pred = classifier.predict(xtest)

pred
final_submission = pd.Series(pred)

final_submission
final_submission.to_csv("final_submission.csv", header=["income"], index_label="Id")