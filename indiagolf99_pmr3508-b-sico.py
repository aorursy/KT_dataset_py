import pandas as pd
import numpy as np
import sklearn
import os 
os.listdir('../input/adultb')
adult=pd.read_csv('../input/adultb/train_data.csv',
                  sep=',', engine='python',
                  na_values="?")
adult.shape
adult.info()
nadult = adult.copy()
nadult
naadult = adult.isnull()
adult.isnull().sum()
Xadult = adult[['age','education.num',
                 'capital.gain','capital.loss',
                 'hours.per.week']]
testAdult = pd.read_csv('../input/adultb/test_data.csv',
                        sep=',',engine='python',
                        na_values='?')
testAdult.isnull().sum()
XtestAdult = testAdult[['age','education.num',
                 'capital.gain','capital.loss',
                 'hours.per.week']]
Yadult = adult.income
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=30)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(knn, Xadult, Yadult, cv=10)
scores
scores.mean()
knn.fit(Xadult,Yadult)
YtestPred = knn.predict(XtestAdult)
result = np.vstack((testAdult["Id"], YtestPred)).T
x = ["id","income"]
submit = pd.DataFrame(columns = x, data = result)
submit.to_csv("Resultados.csv", index = False)
