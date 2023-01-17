import pandas as pd

import numpy as np

import matplotlib.pyplot as plt 

import sklearn
train_data = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv",

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")



train_data.shape
train_data.head()
# remove dados faltantes

train_data = train_data.dropna()

train_data.shape
test_data = pd.read_csv("/kaggle/input/adult-pmr3508/test_data.csv",

                       sep=r'\s*,\s*',

                       engine='python',

                       na_values="?")

#test_data = test_data.dropna()

test_data.shape
test_data.head()
train_data.describe()
train_data["native.country"].value_counts()
train_data['sex'].value_counts().plot(kind = 'pie')
train_data["education"].value_counts().plot(kind="bar")
train_data["income"].value_counts().plot(kind="pie")

train_data["income"].value_counts()
train_data["occupation"].value_counts().plot(kind="bar")
aux_data = train_data[train_data['income'] == '>50K']

aux_data["native.country"].value_counts()
aux_data = train_data[train_data['income'] == '>50K']

aux_data["age"].value_counts()
aux_data = train_data[train_data['sex'] == 'Male']

aux_data["income"].value_counts().plot(kind="bar")

aux_data["income"].value_counts()
aux_data = train_data[train_data['sex'] == 'Female']

aux_data["income"].value_counts().plot(kind="bar")

aux_data["income"].value_counts()
aux_data = train_data[train_data['education'] == 'HS-grad']

aux_data["income"].value_counts().plot(kind="bar")

aux_data["income"].value_counts()
aux_data = train_data[train_data['education'] == 'Doctorate']

aux_data["income"].value_counts().plot(kind="bar")

aux_data["income"].value_counts()
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score



# entradas do classificador (apenas numericas)

x_train = train_data[['age','education.num','capital.gain','capital.loss','hours.per.week']]



# target

y_train = train_data['income']



n_folds = 12

k_max = 50

k_best_num = 1

best_score_num = 0



for k in range(1,k_max+1):

    knn = KNeighborsClassifier(n_neighbors=k)

    scores = cross_val_score(knn, x_train, y_train, cv = n_folds)

    

    if scores.mean() > best_score_num:

        best_score_num = scores.mean()

        k_best_num = k



print("Best value of k only numeric = ", k_best_num)

print("Best score only numeric = ", best_score_num)  
from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler



train_data_num = train_data.apply(preprocessing.LabelEncoder().fit_transform)

x_train = train_data_num

scaler = StandardScaler()



x_train_scaled = scaler.fit_transform(x_train)

y_train = train_data_num['income']



n_folds = 12

k_max = 50

k_best_all = 1

best_score_all = 0



for k in range(1,k_max+1):

    knn = KNeighborsClassifier(n_neighbors=k)

    scores = cross_val_score(knn, x_train, y_train, cv = n_folds)

    

    if scores.mean() > best_score_all:

        best_score_all = scores.mean()

        k_best_all = k

        

print("Best value of k = ", k_best_all)

print("Best score = ", best_score_all)        
# escolhe o melhor entre os k's e os melhores atributos



k_best = k_best_num

x_train = train_data[['age','education.num','capital.gain','capital.loss','hours.per.week']]

y_train = train_data['income']    

knn = KNeighborsClassifier(n_neighbors = k_best)

knn.fit(x_train, y_train)
# dados de teste

x_test = test_data[['age','education.num','capital.gain','capital.loss','hours.per.week']] 

y_pred = knn.predict(x_test)



predicted_data = pd.DataFrame(y_pred, columns=['Income'])

predicted_data
predicted_data.to_csv("subimission.csv", index_label = 'Id')