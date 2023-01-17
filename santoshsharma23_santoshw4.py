import pandas as pd

Task5 = pd.read_csv("../input/play-golf-or-not/Task5.csv")
import numpy as np

import matplotlib.pyplot as plt
# Importing the dataset

dataset = Task5

X=dataset[['Home/Away','In/Out','Media']]

Y=dataset[['Play']]
dataset
XOHE = pd.get_dummies(X) #Alternative way to do one hot encoding (OHE)
XOHE
#Xtrain = Xtrain.astype('category')

Xtrain=XOHE.head(24)

Xtest=XOHE.tail(12)

Xtrain
Ytrain=Y.head(24)

Ytrue=Y.tail(12)

Ytrain
from sklearn.naive_bayes import GaussianNB

clf=GaussianNB()

clf.fit(Xtrain, Ytrain)
Ypred = clf.predict(Xtest)
Ypred=pd.DataFrame(Ypred)
Ypred
Ytrue
# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Ytrue, Ypred)
cm
from sklearn.metrics import accuracy_score

accuracy_score(Ytrue, Ypred)
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()

Ytrue = lb_make.fit_transform(Ytrue)

Ypred = lb_make.fit_transform(Ypred)
from sklearn.metrics import precision_score

precision_score(Ytrue, Ypred)
from sklearn.metrics import recall_score

recall_score(Ytrue, Ypred)
from sklearn.metrics import f1_score

f1_score(Ytrue, Ypred)