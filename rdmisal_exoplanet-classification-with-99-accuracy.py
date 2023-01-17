import numpy as np 

import pandas as pd 

import os

print(os.listdir("../input"))

train=pd.read_csv('../input/exoTrain.csv')

test=pd.read_csv('../input/exoTest.csv')
train.head()
test.head()
x_train=train.drop('LABEL',axis=1)

y_train=train[['LABEL']]

x_test=test.drop('LABEL',axis=1)

y_test=test[['LABEL']]
from sklearn.naive_bayes import BernoulliNB

from sklearn.pipeline import Pipeline

model=BernoulliNB(alpha=0.1)

model.fit(x_train,y_train)
prediction=model.predict(x_test)

from sklearn.metrics import accuracy_score, confusion_matrix

accuracy_score(y_test,prediction)
confusion_matrix(y_test,prediction)
from sklearn.svm import SVC

model=SVC()

model.fit(x_train,y_train)
prediction=model.predict(x_test)

from sklearn.metrics import accuracy_score, confusion_matrix

accuracy_score(y_test,prediction)
confusion_matrix(y_test,prediction)