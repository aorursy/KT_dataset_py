# Скачайте train и test выборку датасета Titanic (также доступно в материалах семинара)

# и прочитайте с помощью pandas

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



import os

# List data files that are connected to the kernel

os.listdir('../input/')



sns.set_style("whitegrid")

%matplotlib inline



train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



print(type(train), "Dimensions of 'train' dataset:", np.ndim(train))

print("Shape of 'train' dataset:", np.shape(train), "\n")



print(type(test), "Dimensions of 'test' dataset:", np.ndim(test))

print("Shape of 'test' dataset:", np.shape(test), "\n")



train.head()



# Imbalanced Classification Problem: Accuracy Paradox so using Cohen’s Kappa coefficient for metric

# ct = pd.crosstab(train.Survived, train.Sex, margins=True, normalize = 'columns')

print("Binary Class 'Survived' is Imbalanced !!!!!", "\n", train.Survived.value_counts() / np.shape(train)[0])



sns.catplot(x="Sex", y="Survived", kind="violin", data=train, jitter=True)
# Почистите / предобработайте данные в формате для sklearn (отдельно - X, отдельно - Y)

test_passengerids = test["PassengerId"]



def preprocess_data(data):

    columns_to_drop = ["Ticket", "PassengerId", "Name", "Cabin"]

    data.drop(columns_to_drop, axis=1, inplace=True)



    data["Sex"] = (data["Sex"] == "female").astype(int)

    data["Embarked"] = data["Embarked"].map({"S":0, "C":1, "Q":2})



    data.fillna(-1, inplace=True)

    

preprocess_data(train)

preprocess_data(test)



labels = train["Survived"]

train.drop("Survived", axis=1, inplace=True)
# Обучите классификатор kNN на train и оцените качество с помощью cross_val_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score





knn = KNeighborsClassifier(metric = "manhattan")

knn.fit(train, labels)

cross_val_scores = cross_val_score(knn, train, labels, scoring="accuracy", cv=5)

print("Accuracy on Validation Sets =", cross_val_scores)

print("Mean and SD of Accuracy on Validation Sets =", cross_val_scores.mean(), cross_val_scores.std())
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.metrics import cohen_kappa_score



np.random.seed(2019)

Xtrain, Xtest, ytrain, ytest = train_test_split(train, labels, train_size=0.7)

knn.fit(Xtrain, ytrain)

cross_val_scores = cross_val_score(knn, train, labels, scoring="accuracy", cv=5)

print("Accuracy on Validation Sets =", cross_val_scores)

print("Mean and SD of Accuracy on Validation Sets =", cross_val_scores.mean(), cross_val_scores.std(), "\n")

predictions = knn.predict(Xtest)

# print(type(predictions)) # <class 'numpy.ndarray'>

# print(predictions)

# print(type(ytest)) # <class 'pandas.core.series.Series'>

# print(ytest)



# Accuracy

print("Accuracy on Test Set =", accuracy_score(y_true=ytest, y_pred=predictions), "\n")



# Cohen’s Kappa coefficient

print("Cohen’s Kappa on Test Set =", cohen_kappa_score(y1=ytest, y2=predictions))
# Добейтесь качества 0.7 по метрике accuracy локально, за счет добавления новых факторов

print(id(train))

train_new_features = train.copy()

print(id(train_new_features))



train_new_features ["Sex_Age"] = train_new_features.Sex*train_new_features.Age

train_new_features ["Age_Pclass"] = train_new_features.Age*train_new_features.Pclass

# train_new_features ["Embarked_Sex"] = train_new_features.Embarked*train_new_features.Sex

train_new_features ["SibSp_Parch"] = train_new_features.SibSp*train_new_features.Parch



np.random.seed(2019)

Xtrain, Xtest, ytrain, ytest = train_test_split(train_new_features, labels, train_size=0.7)

knn.fit(Xtrain, ytrain)

cross_val_scores = cross_val_score(knn, train, labels, scoring="accuracy", cv=5)

print("Accuracy on Validation Sets =", cross_val_scores)

print("Mean amd SD of Accuracy on Validation Sets =", cross_val_scores.mean(), cross_val_scores.std(), "\n")



predictions = knn.predict(Xtest)



# Accuracy

print("Accuracy on Test Set =", accuracy_score(y_true=ytest, y_pred=predictions), '\n')



# Cohen’s Kappa coefficient

print("Cohen’s Kappa on Test Set =", cohen_kappa_score(y1=ytest, y2=predictions))

# Берем проверочный набор данных и обогащаем его производными предикторами

test0 = pd.read_csv("../input/test.csv")



test0.head()



test_new_features = test0.copy()

preprocess_data(test_new_features)

test_new_features["Sex_Age"] = test_new_features.Sex*test_new_features.Age

test_new_features["Age_Pclass"] = test_new_features.Age*test_new_features.Pclass

# test_new_features["Embarked_Sex"] = test_new_features.Embarked*test_new_features.Sex

test_new_features["SibSp_Parch"] = test_new_features.SibSp*test_new_features.Parch



predictions0 = knn.predict(test_new_features)



df = pd.DataFrame({'Survived':predictions0.astype(int)})



result = pd.concat([test0.take([0], axis=1), df], axis=1) # , test.loc[:, test.columns != 'PassengerId']], axis=1)

# result.to_csv('c:/Temp/predictions.csv', index=False)