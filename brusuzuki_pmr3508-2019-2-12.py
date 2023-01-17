# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import sklearn

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score

from sklearn import preprocessing
trainAdult = pd.read_csv("../input/adult-pmr3508/train_data.csv",

        names=[

        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Marital Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country", "Income"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")
trainAdult.shape
trainAdult.head()
trainAdult.drop(trainAdult.index[0],inplace=True) #tirando primeira linha

#ntrainAdult = trainAdult.dropna() #retirando linhas com dados faltantes.



ntrainAdult = trainAdult

ntrainAdult.fillna(method ='ffill', inplace = True) #preenchendo linhas

ntrainAdult
ntrainAdult.shape
testAdult = pd.read_csv("../input/adult-pmr3508/test_data.csv",

        names=[

        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Marital Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country", "Income"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")
testAdult.head()
testAdult.shape
testAdult.drop(testAdult.index[0],inplace=True) #tirando primeira linha

ntestAdult = trainAdult.dropna() #retirando linhas com dados faltantes.



#ntestAdult = testAdult

#ntestAdult.fillna(method ='ffill', inplace = True) #preenchendo linhas
ntestAdult
ntestAdult.shape
testAdult.shape
XtrainAdult = ntrainAdult[["Age", "Workclass", "Education-Num", "Marital Status", "Relationship", "Capital Gain", "Capital Loss", "Hours per week", "Country"]].apply(preprocessing.LabelEncoder().fit_transform)

YtrainAdult = ntrainAdult.Income 



YtrainAdult.shape
YtrainAdult.head()
XtestAdult = ntestAdult[["Age", "Workclass", "Education-Num", "Marital Status", "Relationship", "Capital Gain", "Capital Loss", "Hours per week", "Country"]].apply(preprocessing.LabelEncoder().fit_transform) 

YtestAdult = ntestAdult.Income 
##procurando o melhor valor para n_estimators (The number of trees in the forest)



#n_n = 95

#cval = 10

#acuracia_esperada = 0

#acuracia_best = 0

#n_n_best = 0;

#array_acuracia = []

#while n_n <=100:



    ###########################

    ##para RandomForest:

    #rf = RandomForestClassifier(n_estimators = n_n)

    #rf.fit(XtrainAdult,YtrainAdult)

    #scores = cross_val_score(rf, XtrainAdult, YtrainAdult, cv=cval)

    #YtestPred = rf.predict(XtestAdult)

    ############################

    

    

#    total = 0

#    for i in scores:

#        total += i

#    acuracia_esperada = total/cval

#    array_acuracia.append(acuracia_esperada)

#    if acuracia_esperada > acuracia_best:

#        acuracia_best = acuracia_esperada

#        n_n_best = n_n

#    n_n = n_n + 1   

#acuracia_best
#n_n_best

#array_acuracia
#para RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 62)

rf.fit(XtrainAdult,YtrainAdult)

cval = 10

scores = cross_val_score(rf, XtrainAdult, YtrainAdult, cv=cval)

scores

#para RandomForestClassifier

YtestPred_rf = rf.predict(XtestAdult)

YtestPred_rf
total = 0

for i in scores:

    total += i

acuracia_esperada = total/cval

acuracia_esperada
#para MLPClassifier  

clf = MLPClassifier(hidden_layer_sizes=(100,50,50), max_iter=100,activation = 'relu',solver='adam',random_state=1) 

clf.fit(XtrainAdult,YtrainAdult)

cval = 10

scores = cross_val_score(clf, XtrainAdult, YtrainAdult, cv=cval)

scores
#para MLPClassifier  

YtestPred_clf = clf.predict(XtestAdult)

YtestPred_clf
total = 0

for i in scores:

    total += i

acuracia_esperada = total/cval

acuracia_esperada
#para AdaBoostClassifier

abc = AdaBoostClassifier(n_estimators=150)

abc.fit(XtrainAdult,YtrainAdult) 

cval = 10

scores = cross_val_score(abc, XtrainAdult, YtrainAdult, cv=cval)

scores
#para AdaBoostClassifier

YtestPred_abc = abc.predict(XtestAdult)

YtestPred_abc
total = 0

for i in scores:

    total += i

acuracia_esperada = total/cval

acuracia_esperada
#preparando arquivos para submissão 

df_pred_rf = pd.DataFrame({'Income':YtestPred_rf})

df_pred_clf = pd.DataFrame({'Income':YtestPred_clf})

df_pred_abc = pd.DataFrame({'Income':YtestPred_abc})





df_pred_rf.to_csv("rf_prediction.csv", index = True, index_label = 'Id')

df_pred_clf.to_csv("clf_prediction.csv", index = True, index_label = 'Id')

df_pred_abc.to_csv("abc_prediction.csv", index = True, index_label = 'Id')