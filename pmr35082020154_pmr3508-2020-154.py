import pandas as pd

import sklearn

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
trainAdult = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv",

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")
trainAdult.shape
trainAdult.head()
testAdult = pd.read_csv("/kaggle/input/adult-pmr3508/test_data.csv",

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")
testAdult.shape
testAdult.head()
ntrainAdult = trainAdult.dropna()
ntrainAdult
dadosanalise = ntrainAdult.copy()
from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()



dadosanalise["income"] = le.fit_transform(dadosanalise["income"])

dadosanalise["Id"] = "-"
mask = np.triu(np.ones_like(dadosanalise.corr(), dtype=np.bool))



plt.figure(figsize=(10,10))



sns.heatmap(dadosanalise.corr(), mask=mask, square = True, annot=True, vmin=-1, vmax=1, cmap='autumn')

plt.show()
XtrainAdult = ntrainAdult[["age", "education.num", "capital.gain", "capital.loss", "hours.per.week"]]

YtrainAdult = ntrainAdult.income
XtestAdult = testAdult[["age", "education.num", "capital.gain", "capital.loss", "hours.per.week"]]
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score
best_n = 0

best_acc = 0

for n in range (20,30):

    knn = KNeighborsClassifier(n_neighbors = n)

    scores = cross_val_score(knn, XtrainAdult, YtrainAdult, cv = 10)

    score = np.mean(scores)

    if (best_acc<score):

        best_acc = score

        best_n = n



print("Melhor hiperparâmetro: ", best_n)

print("Melhor acurácia: ", best_acc)
knn = KNeighborsClassifier(n_neighbors=24)
scores = cross_val_score(knn, XtrainAdult, YtrainAdult, cv=10)
scores
knn.fit(XtrainAdult,YtrainAdult)
YtestPred = knn.predict(XtestAdult)
accuracy = np.mean(scores)

accuracy
id_index = pd.DataFrame({'Id' : list(range(len(YtestPred)))})

income = pd.DataFrame({'income' : YtestPred})

result = income
result.to_csv("submission.csv", index = True, index_label = 'Id')