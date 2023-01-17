'''

Importando as bbiotecas necessárias

'''

import sklearn

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as graphic

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score as acs

from sklearn import preprocessing

from time import time

'''

Guardando as bases de teste e treino

'''



adultTrain = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv",

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")



adultTest = pd.read_csv("/kaggle/input/adult-pmr3508/test_data.csv")

adultTest.shape
adultTrain.shape
adultTrain.head()
'''

Devemos eliminar os dados faltantes das bases de teste e de treinamento

'''

nTrain = adultTrain.dropna()

adultTest.set_index('Id',inplace=True)

nTest = adultTest.dropna()

nTest.shape
nTrain.shape

nTrain.head()
nTest.shape

nTest.head()
'''

Transformando os valores em númericos

'''

numTrain = nTrain.apply(preprocessing.LabelEncoder().fit_transform)

numTest = adultTest.apply(preprocessing.LabelEncoder().fit_transform)

numTrain.head()
adultTrain["native.country"].value_counts()
adultTrain["age"].value_counts().plot(kind="bar")
adultTrain["sex"].value_counts().plot(kind="bar")
adultTrain["education"].value_counts().plot(kind="bar")
adultTrain["occupation"].value_counts().plot(kind="bar")
adultTrain["income"].value_counts().plot(kind="bar")
adultTrain["income"].value_counts().plot(kind="bar")
adultTrain["race"].value_counts().plot(kind="pie")
numTrain2 = nTrain.apply(preprocessing.LabelEncoder().fit_transform)

numTrain2.corr()
'''

Pegando os dados com os maiores módulos entre as correlações (maior que 0.09)

("age","education.num","marital.status","relationship","sex","capital.gain","capital.loss","hours.per.week")

'''

xTrain = numTrain[["age","education.num","marital.status","relationship","sex","capital.gain","capital.loss","hours.per.week"]]

yTrain = nTrain.income



xTest = numTest[["age","education.num","marital.status","relationship","sex","capital.gain","capital.loss","hours.per.week"]]

def mediaClass (Class,CV,Xadult, Yadult):

    '''

    Essa função calcula a média dos scores para CV foldes eilizando k neighbors 

    '''

    scores = cross_val_score(Class, Xadult, Yadult, cv=CV)

    average = 0

    for i in scores:

        average += i

    average = average/len(scores)

    return average , scores 
from sklearn.ensemble import RandomForestClassifier



# fazendo uma floresta aleatória

#Testando alguns valores para achar melhor profundidade e número de estimadores

'''

mediaMax=0

scoreMax=0

tempMax=0



for i in range (50,101,10):

    for j in range (5,10):

        randomForst=RandomForestClassifier(n_estimators=i,max_depth = j) 



        # Testando por validação cruzada e o tempo de processamento:

        

        tempo0 = time()

        media,scores = mediaClass (randomForst,10,xTrain, yTrain)

        tempo1 = time()

        tempoRndFrst = tempo1 - tempo0

        if media>mediaMax:

            mediaMax = media

            scoreMax=scores

            tempMax=tempoRndFrst

            a=i

            b=j

'''
from sklearn.tree import DecisionTreeClassifier 



#Fazendo uma árvore de decisão

#Testando alguns valores para achar melhor profundidade

'''

mediaMax=0

scoreMax=0

tempMax=0



for i in range (1,21):

    Tree = DecisionTreeClassifier(max_depth=i)

    

    tempo0 = time()

    media,scores = mediaClass (Tree,10,xTrain, yTrain)

    tempo1 = time()

    tempoTree = tempo1 - tempo0

    if media>mediaMax:

        mediaMax = media

        scoreMax=scores

        tempMax=tempoTree

        a=i

'''
from sklearn import svm

'''

svm = svm.SVC(gamma='auto')

tempo0 = time()

media,scores = mediaClass (svm,10,xTrain, yTrain)

tempo1 = time()

temposvm = tempo1 - tempo0

'''
Tree = DecisionTreeClassifier(max_depth=9)

Tree.fit(xTrain,yTrain)

yTest=Tree.predict(xTest)
savepath = "predictions82.csv"

prev = pd.DataFrame(yTest, columns = ["income"])

prev.to_csv(savepath, index_label="Id")

prev