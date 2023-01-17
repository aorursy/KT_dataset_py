#Import libraries

import pandas as pd

import sklearn as skl

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import StandardScaler

from sklearn import preprocessing

from sklearn.model_selection import GridSearchCV



#Import dataset

train_adult = pd.read_csv("../input/adult-pmr3508/train_data.csv", na_values="?")

test_adult = pd.read_csv("../input/adult-pmr3508/test_data.csv", na_values="?")
print(train_adult.shape,

test_adult.shape)
train_adult.head()
train_adult = train_adult.dropna()

#Quantitativos

test_adult["age"] = test_adult["age"].fillna(test_adult["age"].mean())

test_adult["fnlwgt"] = test_adult["fnlwgt"].fillna(test_adult["fnlwgt"].mean())

test_adult["education.num"] = test_adult["education.num"].fillna(test_adult["education.num"].mean())

test_adult["capital.gain"] = test_adult["capital.gain"].fillna(test_adult["capital.gain"].mean())

test_adult["capital.loss"] = test_adult["capital.loss"].fillna(test_adult["capital.loss"].mean())

test_adult["hours.per.week"] = test_adult["hours.per.week"].fillna(test_adult["hours.per.week"].mean())



#Qualitativos

test_adult["workclass"] = test_adult["workclass"].fillna(test_adult["workclass"].mode()[0])

test_adult["education"] = test_adult["education"].fillna(test_adult["education"].mode()[0])

test_adult["marital.status"] = test_adult["marital.status"].fillna(test_adult["marital.status"].mode()[0])

test_adult["occupation"] = test_adult["occupation"].fillna(test_adult["occupation"].mode()[0])

test_adult["relationship"] = test_adult["relationship"].fillna(test_adult["relationship"].mode()[0])

test_adult["race"] = test_adult["race"].fillna(test_adult["race"].mode()[0])

test_adult["sex"] = test_adult["sex"].fillna(test_adult["sex"].mode()[0])

test_adult["native.country"] = test_adult["native.country"].fillna(test_adult["native.country"].mode()[0])







test_adult.shape

quantitative = train_adult.describe(include=[np.number])

qualitative = train_adult.describe(exclude=[np.number])
quantitative.columns

quantitative
qualitative.columns

qualitative
correlation_sample = train_adult.copy()

Y_train = train_adult.pop('income')

X_train = train_adult

X_test = test_adult

print(X_test.shape, X_train.shape)

X_train.head()
X_test = X_test.apply(preprocessing.LabelEncoder().fit_transform)

X_train = X_train.apply(preprocessing.LabelEncoder().fit_transform)

correlation_sample = correlation_sample.apply(preprocessing.LabelEncoder().fit_transform)



X_train.head()
plt.figure(figsize=(20, 20))

sns.heatmap(correlation_sample.corr(), annot=True, cmap="RdYlBu", vmin=-1)
X_train = X_train.drop(['workclass','fnlwgt', 'native.country', 'education','Id'], axis=1)

X_test = X_test.drop(['workclass','fnlwgt', 'native.country', 'education','Id'], axis=1)



X_train.head()
scaler = preprocessing.MinMaxScaler()

X_train = pd.DataFrame(data=scaler.fit_transform(X_train))

X_test = pd.DataFrame(data=scaler.fit_transform(X_test))



X_train.head()
knn = KNeighborsClassifier()



param_grid = {'n_neighbors': np.arange(25, 35)}



knn_gscv = GridSearchCV(knn, param_grid, cv=10)



knn_gscv.fit(X_train, Y_train)



print(knn_gscv.best_score_)



predicoes = knn_gscv.predict(X_test)



predicoes

submissao = pd.DataFrame()

submissao[0] = X_test.index

submissao[1] = predicoes

submissao.columns = ['Id', 'income']



submissao.to_csv('submission.csv', index = False)