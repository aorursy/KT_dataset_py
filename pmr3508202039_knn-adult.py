import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import preprocessing 

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_validate

from sklearn.inspection import permutation_importance
data = pd.read_csv("../input/adult-pmr3508/train_data.csv",

        index_col=['Id'],

        na_values="?")
data.shape
data.info()
data.head()
data.isnull().sum()
data = data.dropna()

data.shape
encodedColumns = ["workclass", "education", "marital.status", "relationship", "occupation", "race", "sex", "native.country", "income"]

encoders = {column: pd.Categorical(data[column]).categories for column in encodedColumns}



processedData = data.copy()

for column in encodedColumns:

    if (column in data.columns):

        processedData[column] = pd.Categorical(data[column], encoders[column]).codes
processedData.head()
corr = processedData.corr()

plt.figure(figsize=(15,15))

sns.heatmap(corr, cmap='autumn', annot=True, square=True)

plt.show()
usedColumns = ["age", "education.num", "marital.status", "occupation", "relationship", "race", "sex", "capital.gain", "capital.loss"]



X = processedData[usedColumns]

Y = processedData["income"]
max_n = -1

max_score = 0

for n in range(1, 31):

    print(f"{n}: ", end="")

    knn = KNeighborsClassifier(n_neighbors = n)

    cv_results = cross_validate(knn, X, Y, cv = 20, n_jobs = -1)

    score = np.average(cv_results['test_score'])



    print(f"{score:0<7.5}")

    

    if score > max_score:

        max_n = n

        max_score = score

    

print()

print(f"Melhor resultado: n = {max_n}, score = {max_score}")
knn = KNeighborsClassifier(n_neighbors = 16)

knn.fit(X, Y)
permutation_scores = permutation_importance(knn, X, Y)

permutation_scores.importances_mean
X.columns
usedColumns = ["age", "education.num", "marital.status", "occupation", "relationship", "capital.gain", "capital.loss"]

X = processedData[usedColumns]



cv_results = cross_validate(knn, X, Y, cv = 20, n_jobs = -1)

score = np.average(cv_results['test_score'])

score
X = processedData[usedColumns]



knn = KNeighborsClassifier(n_neighbors = 16)

knn.fit(X, Y)
data_test = pd.read_csv("../input/adult-pmr3508/test_data.csv",

        na_values="?")



for column in encodedColumns[: -1]:

    if (column in data.columns):

        data_test[column] = pd.Categorical(data_test[column], encoders[column]).codes



X_test = data_test[usedColumns]
Y_test = knn.predict(X_test)
Y_test = pd.DataFrame(Y_test, columns=['income'])

Y_test.income = encoders['income'][Y_test.income]
Y_test.to_csv("answers.csv", index_label="Id")