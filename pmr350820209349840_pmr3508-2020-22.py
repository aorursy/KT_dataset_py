#Importação das bibliotecas a serem utilizados no modelo



import pandas as pd

import sklearn

import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

from sklearn import preprocessing

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score
#Carregamento da base de Treino



train = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv",

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")
#Verificação do tamanho da base de Treino



train.shape
#Verificação da distribuição da base nas linha e colunas



train.head()
#Preparação dos Dados de Treino



#Criação de uma nova base, preenchendo as linhas com falta de informações

train.fillna(method ='ffill', inplace = True)

new_train=train



new_train.head()
#Análise do atributo "age"



train["age"].value_counts().plot(kind="bar")
#Análise do atributo "workclass"



train["workclass"].value_counts().plot(kind="bar")
#Análise do atributo "fnlwgt"



train["fnlwgt"].value_counts()
#Análise do atributo "education"



train["education"].value_counts().plot(kind="bar")
#Análise do atributo "education.num"



train["education.num"].value_counts().plot(kind="bar")
#Análise do atributo "marital.status"



train["marital.status"].value_counts().plot(kind="bar")
#Análise do atributo "occupation"



train["occupation"].value_counts().plot(kind="bar")
#Análise do atributo "relationship"



train["relationship"].value_counts().plot(kind="bar")
#Análise do atributo "race"



train["race"].value_counts().plot(kind="bar")
#Análise do atributo "sex"



train["sex"].value_counts().plot(kind="bar")
#Análise do atributo "capital.gain"



train["capital.gain"].value_counts()
#Análise do atributo "capital.loss"



train["capital.loss"].value_counts()
#Análise do atributo "hours.per.week"



train["hours.per.week"].value_counts()
#Análise do atributo "native.country"



train["native.country"].value_counts()
#Carregamento da base de Teste



test = pd.read_csv("/kaggle/input/adult-pmr3508/test_data.csv",

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")
#Verificação do tamanho da base de Teste



test.shape
#Verificação da distribuição da base nas linha e colunas



test.head()
#Preparação dos Dados de Treino



#Criação de uma nova base, preenchendo as linhas com falta de informações

test.fillna(method ='ffill', inplace = True)

new_test=test



new_test.head()
new_test.shape
#Coloca os rótulos numéricos nos dados não-numéricos na base de treino



num_train = new_train.apply(preprocessing.LabelEncoder().fit_transform)
#Coloca os rótulos numéricos nos dados não-numéricos na base de teste



num_test = new_test.apply(preprocessing.LabelEncoder().fit_transform)
#Escolha do conjunto de atributos para a base de treino



X_train = num_train[["age", "education","education.num", "marital.status", "occupation", "relationship", "sex", "capital.gain", "capital.loss", "hours.per.week"]]
#"Income" é escolhido como sendo o atributo a ser testado



Y_train = num_train.income
#Escolha de k=3



knn = KNeighborsClassifier(n_neighbors=3)
#Validação cruzado com 10 folds, e cálculo da acurácia



scores = cross_val_score(knn, X_train, Y_train, cv=10)



np.mean(scores)
#Escolha de k=10



knn = KNeighborsClassifier(n_neighbors=10)
#Validação cruzado com 10 folds, e cálculo da acurácia



scores = cross_val_score(knn, X_train, Y_train, cv=10)



np.mean(scores)
#Escolha de k=30



knn = KNeighborsClassifier(n_neighbors=30)
#Validação cruzado com 10 folds, e cálculo da acurácia



scores = cross_val_score(knn, X_train, Y_train, cv=10)



np.mean(scores)
#Escolha do conjunto de atributos para a base de treino



X1_train = num_train[["age", "education","education.num", "marital.status", "occupation", "relationship", "sex", "capital.gain", "capital.loss", "hours.per.week"]]
#"Income" é escolhido como sendo o atributo a ser testado



Y1_train = num_train.income
#Escolha do conjunto de atributos para a base de teste



X1_test = num_test[["age", "education","education.num", "marital.status", "occupation", "relationship", "sex", "capital.gain", "capital.loss", "hours.per.week"]]
#Escolha de k=10



knn = KNeighborsClassifier(n_neighbors=10)
#Validação cruzado com 10 folds, e cálculo da acurácia



scores = cross_val_score(knn, X1_train, Y1_train, cv=10)



np.mean(scores)
#Escolha do conjunto de atributos para a base de treino



X2_train = num_train[["age", "education","education.num", "marital.status", "occupation", "relationship", "sex","hours.per.week"]]
#"Income" é escolhido como sendo o atributo a ser testado



Y2_train = num_train.income
#Escolha do conjunto de atributos para a base de teste



X2_test = num_test[["age", "education","education.num", "marital.status", "occupation", "relationship", "sex","hours.per.week"]]
#Escolha de k=10



knn = KNeighborsClassifier(n_neighbors=10)
#Validação cruzado com 10 folds, e cálculo da acurácia



scores = cross_val_score(knn, X2_train, Y2_train, cv=10)



np.mean(scores)
#Escolha do conjunto de atributos para a base de treino



X3_train = num_train[["age", "marital.status", "occupation", "relationship", "sex", "capital.gain", "capital.loss"]]
#"Income" é escolhido como sendo o atributo a ser testado



Y3_train = num_train.income
#Escolha do conjunto de atributos para a base de teste



X3_test = num_test[["age", "marital.status", "occupation", "relationship", "sex", "capital.gain", "capital.loss"]]
#Escolha de k=10



knn = KNeighborsClassifier(n_neighbors=10)
#Validação cruzado com 10 folds, e cálculo da acurácia



scores = cross_val_score(knn, X3_train, Y3_train, cv=10)



np.mean(scores)
#Base de Treino é treinada



knn.fit(X3_train,Y3_train)
#Predição da base de Teste



Y_test_pred = knn.predict(X3_test)
# Preparando arquivo para submissao

savepath = "predictions.csv"

submition = pd.DataFrame(Y_test_pred, columns = ["income"])

submition = submition.replace([0,1],['<=50K', '>50K'])

submition.to_csv(savepath, index_label="Id")

submition