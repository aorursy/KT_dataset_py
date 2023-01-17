import pandas as pd

import sklearn

import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
bd = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv",  #base de treino

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")
bd.head()
bd = bd.dropna() #removendo os valores faltantes

bd.head()

bd.shape
from sklearn import preprocessing

numerical = ["age","education.num","capital.gain", "capital.loss", "sex", "marital.status", "hours.per.week"]

bd[numerical] = bd[numerical].apply(preprocessing.LabelEncoder().fit_transform)
Xtrain = bd[["age","education.num","capital.gain", "capital.loss", "sex", "marital.status", "hours.per.week"]]

Ytrain = bd.income
testbd = pd.read_csv("/kaggle/input/adult-pmr3508/test_data.csv",  #base de teste

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")
Xtest = testbd[["age","education.num","capital.gain", "capital.loss", "sex", "marital.status", "hours.per.week"]]
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(Xtrain,Ytrain,test_size = 0.50)
LReg = LogisticRegression()

LReg.fit(Xtrain, Ytrain)

LPred = LReg.predict(Xtest)
print("Metrics: Accuracy, Precision, Recall, F1-score")

print(confusion_matrix(Ytest, LPred))

print(classification_report(Ytest, LPred))

print('Accuracy: ',accuracy_score(Ytest, LPred))
from sklearn.naive_bayes import GaussianNB
NB = GaussianNB()

NB.fit(Xtrain,Ytrain)

NB_Pred = NB.predict(Xtest)
print("Métricas: Accuracy, Precision, Recall, F1-score")

print(confusion_matrix(Ytest, NB_Pred))

print(classification_report(Ytest, NB_Pred))

print('Accuracy: ',accuracy_score(Ytest, NB_Pred))
from sklearn.ensemble import RandomForestClassifier #importando o classificador Random Forest
RF = RandomForestClassifier(n_estimators = 100) 

RF.fit(Xtrain, Ytrain)

RF_Pred = RF.predict(Xtest)
print("Métricas: Accuracy, Precision, Recall, F1-score")

print(confusion_matrix(Ytest, RF_Pred))

print(classification_report(Ytest, RF_Pred))

print('Accuracy: ',accuracy_score(Ytest, RF_Pred))