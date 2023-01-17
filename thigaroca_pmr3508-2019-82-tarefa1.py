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

from sklearn.metrics import accuracy_score

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
adultTrain["native.country"].value_counts()
adultTrain["age"].value_counts().plot(kind="bar")
adultTrain["sex"].value_counts().plot(kind="bar")
adultTrain["education"].value_counts().plot(kind="bar")
adultTrain["occupation"].value_counts().plot(kind="bar")
adultTrain["income"].value_counts().plot(kind="bar")
adultTrain["income"].value_counts().plot(kind="bar")
adultTrain["race"].value_counts().plot(kind="pie")
'''

Primeiramente devemos eliminar os dados faltantes das bases de teste e de treinamento

'''

nTrain = adultTrain.dropna()

adultTest.set_index('Id',inplace=True)

nTest = adultTest.dropna()

nTest.shape
nTrain.shape

nTrain
nTest.shape

nTest
'''

Transformando os valores em númericos

'''

numTrain = nTrain.apply(preprocessing.LabelEncoder().fit_transform)

numTest = adultTest.apply(preprocessing.LabelEncoder().fit_transform)
numTrain
numTrain2 = nTrain.apply(preprocessing.LabelEncoder().fit_transform)

numTrain2.corr()
'''

Pegando os valores que tem correlação positiva com o income temos:



Pegando sómente as colunas númericas:

'''

xTrain = numTrain[["age", "workclass", "education","education.num",

        "occupation", "race", "sex", "capital.gain", "capital.loss",

        "hours.per.week", "native.country"]]

YTrain = nTrain.income



'''

Pegando os Targets do Test data:

'''

xTest = numTest[["age", "workclass", "education","education.num",

        "occupation", "race", "sex", "capital.gain", "capital.loss",

        "hours.per.week", "native.country"]]

#YTest = numTest.income

def mediaKNN (K,CV,Xadult, Yadult):

    '''

    Essa função calcula a média dos scores para CV foldes eilizando k neighbors 

    '''

    knn = KNeighborsClassifier(n_neighbors= K )

    scores = cross_val_score(knn, Xadult, Yadult, cv=CV)

    average = 0

    for i in scores:

        average += i

    average = average/len(scores)

    return average , scores , K 

    
'''

mediaMax = 0 

K = 0

Scores = [0]

for i in range (1,31):

    average,Pscores,PK =  mediaKNN (i,10,xTrain, YTrain)

    if average>mediaMax:

        mediaMax = average

        K = PK

        Scores = Pscores

print (mediaMax)

print (K)

print (Scores)

'''
knn = KNeighborsClassifier(n_neighbors = 30 )# o Trinta foi definido pelo código acima

knn.fit(xTrain,YTrain)
YtestPred = knn.predict(xTest)
#accuracy_score(YTest,YtestPred)
savepath = "predictions1.csv"

prev = pd.DataFrame(YtestPred, columns = ["income"])

prev.to_csv(savepath, index_label="Id")

prev