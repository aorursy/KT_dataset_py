import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score
import os

os.listdir('../input')
data = pd.read_csv("../input/adult-pmr3508/train_data.csv")

data = data.dropna()

test = pd.read_csv("../input/adulttest/adult.test.txt",

        names=[

        "age", "workclass", "fnlwgt", "education", "education.num", "marital.status",

        "occupation", "relationship", "race", "sex", "capital.gain", "capital.loss",

        "hours.per.week", "native.country", "income"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")

test = test.dropna()
data.head()
data.dtypes
def categorical_summarized(dataframe, x=None, y=None, hue=None, palette='Set1', verbose=True):

    if x == None:

        column_interested = y

    else:

        column_interested = x

    series = dataframe[column_interested]

    print(series.describe())

    print('mode: ', series.mode())

    if verbose:

        print('='*80)

        print(series.value_counts())



    sns.countplot(x=x, y=y, hue=hue, data=dataframe, palette=palette)

    plt.show()
categorical_summarized(data, x = "sex", hue = "income", verbose = True)
categorical_summarized(data, x = "education.num", hue = "income", verbose = True)
le = preprocessing.LabelEncoder()

data["workclass"] = le.fit_transform(data["workclass"])

data["education"] = le.fit_transform(data["education"])

data["marital.status"] = le.fit_transform(data["marital.status"])

data["occupation"] = le.fit_transform(data["occupation"])

data["relationship"] = le.fit_transform(data["relationship"])

data["race"] = le.fit_transform(data["race"])

data["sex"] = le.fit_transform(data["sex"])

data["native.country"] = le.fit_transform(data["native.country"])

data["income"] = le.fit_transform(data["income"])
train_target = data["income"]

cols = [col for col in data.columns if col not in ["Id", "fnlwgt", "income"]]

train_data = data[cols]
test["workclass"] = le.fit_transform(test["workclass"])

test["education"] = le.fit_transform(test["education"])

test["marital.status"] = le.fit_transform(test["marital.status"])

test["occupation"] = le.fit_transform(test["occupation"])

test["relationship"] = le.fit_transform(test["relationship"])

test["race"] = le.fit_transform(test["race"])

test["sex"] = le.fit_transform(test["sex"])

test["native.country"] = le.fit_transform(test["native.country"])

test["income"] = le.fit_transform(test["income"])
test_target = test["income"]

cols2 = [col for col in test.columns if col not in ["fnlwgt", "income"]]

test_data = test[cols2]
score_max, k_max = 0, 0

print("Otimizando o valor de k para pesos uniformes...")

for i in range(1,21):

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(train_data, train_target)

    pred = knn.predict(test_data)

    score = accuracy_score(test_target, pred)

    if score > score_max:

        score_max = score

        k_max = i

        print(k_max, " é o melhor valor até o momento...")

print("Valor de k ∈ [1,20] ótimo: ", k_max)

print("Acurácia obtida: ", score_max)

print("Acurácias de validação cruzada: ", cross_val_score(knn, train_data, train_target, cv=10))
train_target = data["income"]

cols = [col for col in data.columns if col not in ["Id", "fnlwgt", "sex", "income"]]

train_data = data[cols]
test_target = test["income"]

cols2 = [col for col in test.columns if col not in ["fnlwgt", "sex", "income"]]

test_data = test[cols2]
knn = KNeighborsClassifier(n_neighbors=20)

knn.fit(train_data, train_target)

pred = knn.predict(test_data)

score = accuracy_score(test_target, pred)



print("Acurácia obtida: ", score)

print("Acurácias de validação cruzada: ", cross_val_score(knn, train_data, train_target, cv=10))
train_target = data["income"]

cols = [col for col in data.columns if col not in ["Id", "fnlwgt", "education", "income"]]

train_data = data[cols]
test_target = test["income"]

cols2 = [col for col in test.columns if col not in ["fnlwgt", "education", "income"]]

test_data = test[cols2]
knn = KNeighborsClassifier(n_neighbors=10)

knn.fit(train_data, train_target)

pred = knn.predict(test_data)

score = accuracy_score(test_target, pred)



print("Acurácia obtida: ", score)

print("Acurácias de validação cruzada: ", cross_val_score(knn, train_data, train_target, cv=10))
id_index = pd.DataFrame({'Id' : list(range(len(pred)))})

result = pd.DataFrame({'income' : pred})