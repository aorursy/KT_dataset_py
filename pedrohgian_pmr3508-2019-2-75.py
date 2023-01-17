import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np



%matplotlib inline

import os
adult = pd.read_csv('../input/adult-data-and-test/adult.data',

        names=[

        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country", "Target"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")

adult.head()
adult.shape #Há 32561 atributos, e 15 parâmetros.
list(adult)
adult["Age"].value_counts().plot(kind="bar")
adult["Country"].value_counts().plot(kind="bar")
adult["Sex"].value_counts().plot(kind="pie")
adult["Martial Status"].value_counts().plot(kind="bar")
adult["Race"].value_counts().plot(kind="pie")
adult["Occupation"].value_counts().plot(kind="bar")
##adult["fnlwgt"].value_counts().plot(kind="bar")
adult.isnull().sum().plot(kind="bar")
nAdult = adult.dropna() 

nAdult
nAdult.shape #Há menos linhas de atributos!
testAdult = pd.read_csv('../input/adult-data-and-test/adult.test',

        names=[

        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country", "Target"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")

testAdult.shape
nTestAdult = testAdult.dropna()

nTestAdult
nTestAdult.shape
#Consideraremos inicialmente todos os valores numéricos

Xadult = nAdult[["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]]

Xadult.head()
Yadult = nAdult.Target

Yadult.head


#Consideraremos inicialmente todos os valores numéricos

XtestAdult = nTestAdult[["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]]
YtestAdult = nTestAdult.Target
#Passando dados não numéricos para numéricos

from sklearn import preprocessing

numAdult = nAdult.apply(preprocessing.LabelEncoder().fit_transform)

numTestAdult = nTestAdult.apply(preprocessing.LabelEncoder().fit_transform)
#Aplicação da Validação Cruzada e Treinamento do modelo:

import sklearn

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score
knn = KNeighborsClassifier(n_neighbors=30)

scores = cross_val_score(knn, Xadult, Yadult, cv=10)

scores.mean()


Xadult = numAdult.iloc[:,0:14]
Yadult = numAdult.Target
XtestAdult = numTestAdult.iloc[:,0:14] 
YtestAdult = numTestAdult.Target
knn = KNeighborsClassifier(n_neighbors=30)

scores = cross_val_score(knn, Xadult, Yadult, cv=10)

scores.mean()
knn.fit(Xadult,Yadult)
YtestPred = knn.predict(XtestAdult)
from sklearn.metrics import accuracy_score 

accuracy_score(YtestAdult,YtestPred)
Xadult = numAdult[["Age", "Workclass", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week"]]
XtestAdult = numTestAdult[["Age", "Workclass", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week"]] 
knn = KNeighborsClassifier(n_neighbors=30)

scores = cross_val_score(knn, Xadult, Yadult, cv=10)

scores.mean()
knn.fit(Xadult,Yadult)
YtestPred = knn.predict(XtestAdult)
accuracy_score(YtestAdult,YtestPred)
## Usando todos os parâmetros

Xadult = numAdult.iloc[:,0:14]

Yadult = numAdult.Target

XtestAdult = numTestAdult.iloc[:,0:14] 

YtestAdult = numTestAdult.Target

#Vamos criar o nosso modelo de Regressão Logística

from sklearn.linear_model import LogisticRegression

classificador = LogisticRegression(max_iter = 10000) #max_iter = quantas vezes vamos realizar o treinamento e ajustar o argumento

#Treinamento do algoritmo classificador

classificador.fit(Xadult, Yadult)
#Fazer as previsões

previsoes = classificador.predict(XtestAdult)

previsoes



#Comparar as previsões da classificação com as classificações reais (YtestAdult)

from sklearn.metrics import accuracy_score

taxa_acerto = accuracy_score(YtestAdult, previsoes)

#quantas previsões acertaram? 

taxa_acerto
# Binary Classification with the Keras Deep Learning Library

from pandas import read_csv

from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline



X = numAdult.iloc[:,0:14] #Atributos previsores

Y = numAdult.Target #Valores reais da classe

# create model

model = Sequential()

model.add(Dense(14, input_dim=14, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

# Compile model

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#Fitting the data to the training dataset

model.fit(X,Y, batch_size=10, epochs=100)
#Evaluation

eval_model=model.evaluate(X,Y)

eval_model #Loss and accuracy of the mode;

#predict the output for test dataset.

Y_pred=model.predict(XtestAdult)

Y_pred =(Y_pred>0.5)
#check the accuracy on the test dataset

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(YtestAdult, Y_pred)

print(cm)
accuracy = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]) #accuracy = (TP + TN)/(TP + FP + FN +TN)

accuracy
classifier = Sequential()

#First Hidden Layer

classifier.add(Dense(7, activation='relu', kernel_initializer='random_normal', input_dim=14))

#Second  Hidden Layer

classifier.add(Dense(7, activation='relu', kernel_initializer='random_normal'))

#Output Layer

classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))

#Compiling the neural network

classifier.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])
#Fitting the data to the training dataset

classifier.fit(X,Y, batch_size=10, epochs=100)
#Evaluation

eval_model=classifier.evaluate(X,Y)

eval_model #Loss and accuracy of the mode;
#predict the output for test dataset.

Y_pred=classifier.predict(XtestAdult)

Y_pred =(Y_pred>0.5)



#check the accuracy on the test dataset

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(YtestAdult, Y_pred)

print(cm)
accuracy = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]) #accuracy = (TP + TN)/(TP + FP + FN +TN)

accuracy
from sklearn.tree import DecisionTreeClassifier

## Usando todos os parâmetros

Xadult = numAdult.iloc[:,0:14]

Yadult = numAdult.Target

XtestAdult = numTestAdult.iloc[:,0:14] 

YtestAdult = numTestAdult.Target



# Criando e treinando o algoritmo da árvore aleatória

arvore = DecisionTreeClassifier()

arvore.fit(Xadult,Yadult)

# avaliação por validacao cruzada

scores = cross_val_score(arvore, Xadult, Yadult, cv=10)

scores.mean()

# avaliação da acurácia com a base de dado de testes

YtestPred = arvore.predict(XtestAdult)

accuracy =accuracy_score(YtestAdult,YtestPred,normalize=True,sample_weight=None)

accuracy
XtestAdult = numTestAdult[["Age", "Workclass", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week"]] 

Xadult = numAdult[["Age", "Workclass", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week"]]
# Criando e treinando o algoritmo da árvore aleatória

arvore = DecisionTreeClassifier()

arvore.fit(Xadult,Yadult)

# avaliação por validacao cruzada

scores = cross_val_score(arvore, Xadult, Yadult, cv=10)

scores.mean()

# avaliação da acurácia com a base de dado de testes

YtestPred = arvore.predict(XtestAdult)

accuracy =accuracy_score(YtestAdult,YtestPred,normalize=True,sample_weight=None)

accuracy
import pandas as pd

test_data = pd.read_csv("../input/california-housing-data/test.csv",

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")

train_data = pd.read_csv("../input/california-housing-data/train.csv",

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")

train_data.head()
train_data.shape

train_data, test_data = train_data.dropna(), test_data.dropna()
#Passando dados não numéricos para numéricos

from sklearn import preprocessing

numTrainData = train_data.apply(preprocessing.LabelEncoder().fit_transform)

numTestData = test_data.apply(preprocessing.LabelEncoder().fit_transform)

numTrainData.head()
numTestData.head()
## Usando todos os parâmetros

X = numTrainData.iloc[:,0:9]

Y = numTrainData.iloc[:,9]

Xtest = numTestData.iloc[:,0:8] 

Ytest = numTestData.iloc[:,8]
#Para criar um modelo de regressão linear

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X, Y) #Treinamento do regressor

regressor.predict(X)
#Comparar as previsões da regressão com a regressão real

from sklearn.model_selection import cross_val_score

accuracy = cross_val_score(regressor, X, Y, cv=10)

accuracy.mean()