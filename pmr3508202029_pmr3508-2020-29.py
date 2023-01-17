# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn

from sklearn import preprocessing

from sklearn.model_selection import cross_val_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn import preprocessing as prep

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

#Dados de treino

train_data = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv", sep=r'\s*,\s*', engine='python', na_values="?")

train_data.describe() # checando se está tudo dentro do esperado

#dados de treino

test_data = pd.read_csv("/kaggle/input/adult-pmr3508/test_data.csv",

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")

train_data.describe() # checando se está tudo dentro do esperado
# Verifica o tamanho da base

train_data.shape
# head() retorna as primeiras n linhas do objeto com base na posição.

# Útil para testar rapidamente se o seu objeto contém o tipo certo de dados.

train_data.head()
#O método isnull() retorna True caso encontre valores do tipo NaN ou Nulo

#Fazendo isso para os dados de treino

train_data.isnull()

#Fazendo isso para os dados de teste:

test_data.isnull()
#soma os registros faltantes por atributos.

#Fazendo isso para os dados de treino

train_data.isnull().sum()
#soma os registros faltantes por atributos.

#Fazendo isso para os dados de teste

test_data.isnull().sum()
#verifica a incidência dos valores dos atributos especificados

print(' occupation:\n')

top_occupation = train_data['occupation'].describe().top #armazena o valor do atributo de maior incidencia na variável

print(train_data['occupation'].describe())



print('\n workclass:\n')

top_workclass = train_data['workclass'].describe().top #armazena o valor do atributo de maior incidencia na variável

print(train_data['workclass'].describe())



print('\n native.country:\n')

top_native_country = train_data['native.country'].describe().top #armazena o valor do atributo de maior incidencia na variável

print(train_data['native.country'].describe())
#preenche os dados faltantes com os valores  das modas dos atributos especificados

#Fazendo isso para a base de treino

train_data['native.country'].fillna(top_native_country,inplace=True)

train_data['workclass'].fillna(top_workclass,inplace=True)

train_data['occupation'].fillna(top_occupation,inplace=True)



#soma os registros faltantes por atributos.

train_data.isnull().sum()
#preenche os dados faltantes com os valores  das modas dos atributos especificados

#Fazendo isso para a base de teste

test_data['native.country'].fillna(top_native_country,inplace=True)

test_data['workclass'].fillna(top_workclass,inplace=True)

test_data['occupation'].fillna(top_occupation,inplace=True)



#soma os registros faltantes por atributos.

test_data.isnull().sum()
train_data["age"].value_counts().plot(kind="bar",figsize=(15,5))
train_data["workclass"].value_counts().plot(kind="bar")
train_data["education"].value_counts().plot(kind="bar")
train_data["marital.status"].value_counts().plot(kind="bar")
train_data["occupation"].value_counts().plot(kind="bar")
train_data["relationship"].value_counts().plot(kind="bar")
train_data["race"].value_counts().plot(kind="pie",figsize=(10,10))
train_data["sex"].value_counts().plot(kind="pie")
train_data["hours.per.week"].value_counts().plot(kind="bar",figsize=(15,5))
train_data["native.country"].value_counts().plot(kind="bar",figsize=(10,5))
train_data["income"].value_counts().plot(kind="pie")
train_data.skew()
#definição X de treino

Xtrain = train_data.drop(['Id', 'income','education'], axis=1).copy() #remove dados desnecessários

Xtrain = Xtrain.apply(preprocessing.LabelEncoder().fit_transform) #transforma os dados que não são numéricos em numéricos

Xtrain.head()
#definição X de teste

test_data = test_data.apply(preprocessing.LabelEncoder().fit_transform).copy() #transforma os dados que não são numéricos em numéricos

Xtest = test_data.drop(['Id', 'education'], axis=1).copy() #remove dados desnecessários

Xtest.head()
#definição Y de treino

Ytrain = train_data['income']

Ytrain.head()

mat_scores = []

num_knn = []

for i in range(30):

    if i>0:

        soma = 0 

        knn = KNeighborsClassifier(n_neighbors=i) #definição do número de vizinho que devem ser comparados para a classificação

        scores = cross_val_score(knn, Xtrain, Ytrain, cv=10) #score da validação cruzada

        for j in scores:

            soma+=j    

        mat_scores.append(soma/len(scores))

        num_knn.append(i)
plt.plot(num_knn,mat_scores)

plt.xlabel("knn")

plt.ylabel("Média scores")
knn = KNeighborsClassifier(n_neighbors=15)

knn.fit(Xtrain, Ytrain)

Ypredicted = knn.predict(Xtest)

arquivo_csv = pd.Series(Ypredicted)

arquivo_csv.to_csv("submission.csv",header=["income"],index_label="Id")

arquivo_csv