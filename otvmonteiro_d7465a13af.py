#Importando as bibliotecas

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.naive_bayes import GaussianNB 
from sklearn.naive_bayes import BernoulliNB 
from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import fbeta_score
#Importando os arquivos

train = pd.read_csv("../input/spamnaivebayes/train_data.csv",sep=r'\s*,\s*',engine='python')
test = pd.read_csv("../input/spamnaivebayes/test_features.csv",sep=r'\s*,\s*',engine='python')

features = train.drop(columns=['ham','Id'])
label = train['ham']

features.head(8)
gnb = GaussianNB()
gnb.fit(features, label)

bnb = BernoulliNB()
bnb.fit(features, label)

mnb = MultinomialNB()
mnb.fit(features, label)
gScores = cross_val_score(gnb, features, label, cv=10, scoring='f1')
bScores = cross_val_score(bnb, features, label, cv=10, scoring='f1')
mScores = cross_val_score(mnb, features, label, cv=10, scoring='f1')


print("Gaussian Mean Score:   ", gScores.mean())
print("Bernoulli Mean Score:  ", bScores.mean())
print("Multinomial Mean Score:", mScores.mean())
xTrain, xTest, yTrain, yTest = train_test_split(features, label, test_size=0.2, random_state=101)

matriz_bnb = BernoulliNB()
matriz_bnb.fit(xTrain,yTrain)

yPred = matriz_bnb.predict(xTest)

print(confusion_matrix(yTest, yPred))
X = test.drop(columns=['Id'])
predicao = pd.DataFrame(columns=["Id","ham"])
predicao["Id"]=test.Id
predicao["ham"] = bnb.predict(X)
predicao.head(10)
predicao.to_csv("Predicao.csv", index=False)