import pandas as pd

import sklearn

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

from sklearn import preprocessing

from sklearn.preprocessing import OneHotEncoder



import matplotlib.pyplot as plt

import numpy as np



import seaborn as sns
adultTrain = pd.read_csv('../input/adult-pmr3508/train_data.csv',

                      index_col=['Id'], na_values="?")

adultTest = pd.read_csv('../input/adult-pmr3508/test_data.csv',

                        index_col=['Id'], na_values="?")
adultTrainFit = adultTrain.fillna(0)
adultTrain.head()
adultTest.head()
adultTrainNum = adultTrain[['age','fnlwgt','education.num','capital.gain','capital.loss','hours.per.week','income']]
%%capture

encoder = preprocessing.LabelEncoder()

graph_data  = adultTrainNum.copy()

graph_data['income'] = encoder.fit_transform(graph_data['income'])
plt.figure(figsize=(10, 8))

sns.heatmap(graph_data.corr(), annot=True, vmin=-1, vmax=1, cmap = 'rocket')
## Comparação entre sexos 

plt.figure(figsize=(12,12))

plt.suptitle("Comparação entre sexos")

plt.subplot(221)

adultTrain[adultTrain['income'] == '<=50K'].sex.value_counts().plot(kind = "pie")

plt.title("income <=50K")

plt.subplot(222)

adultTrain[adultTrain['income'] == '>50K'].sex.value_counts().plot(kind = "pie")

plt.title("income >50K")

plt.show()
## Comparação entre classes de trabalho 

plt.figure(figsize=(12,12))

plt.suptitle("Comparação entre ocupações de trabalho")

plt.subplot(221)

adultTrain[adultTrain['income'] == '<=50K'].workclass.value_counts().plot(kind = "pie")

plt.title("income <=50K")

plt.subplot(222)

adultTrain[adultTrain['income'] == '>50K'].workclass.value_counts().plot(kind = "pie")

plt.title("income >50K")

plt.show()
plt.figure(figsize=(16,4))

sns.countplot(x="income", hue='marital.status', data=adultTrain, palette='rocket')
plt.figure(figsize=(25,4))

sns.countplot(x="income", hue='occupation', data=adultTrain, palette='rocket')
plt.figure(figsize=(16,4))

sns.countplot(x="income", hue='relationship', data=adultTrain, palette='rocket')
plt.figure(figsize=(16,4))

sns.countplot(x="income", hue='race', data=adultTrain, palette='rocket')
plt.figure(figsize=(16,4))

sns.countplot(x="income", hue='sex', data=adultTrain, palette='rocket')
knn = KNeighborsClassifier(n_neighbors=5)
X = adultTrainFit[['age','education.num','sex','capital.gain','capital.loss','hours.per.week']]

Y = adultTrainFit['income']
X
X["sex-int"] = np.where(X.sex=='Male', 1, 0)

X = X.drop(['sex'], axis = 1)
X
accuracy = cross_val_score(knn, X, Y, cv = 5, scoring="accuracy")

print("Acurácia com cross validation:", accuracy.mean())
k = 1

best_k = k

best_accuracy = 0

k_array = []

accuracy_array = []

while k <=30:

    knn = KNeighborsClassifier(n_neighbors=k)

    accuracy = cross_val_score(knn, X, Y, cv = 5)

    

    k_array.append(k)

    accuracy_array.append(accuracy.mean())

    if accuracy.mean() >= best_accuracy:

        best_accuracy = accuracy.mean()

        best_k = k

        print('K = ', k)

        print('Accuracy: ', accuracy.mean())

    k = k + 1

    

print("O melhor k encontrado foi {0} que levou a uma acurácia de {1}".format(best_k,best_accuracy))
k = best_k



KNN = KNeighborsClassifier(n_neighbors=k)

KNN.fit(X, Y)
adultTestFit = adultTest.fillna(0)
Xt = adultTestFit[['age','education.num','sex','capital.gain','capital.loss','hours.per.week']]

Xt["sex-int"] = np.where(Xt.sex=='Male', 1, 0)

Xt = Xt.drop(['sex'], axis = 1)

predicoes = KNN.predict(Xt)
submission = pd.DataFrame()

submission['income'] = predicoes

submission.to_csv('submission.csv',index_label = 'Id')