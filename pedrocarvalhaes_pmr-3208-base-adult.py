import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import os
print(os.listdir('../input'))
adult = pd.read_csv('../input/adult-datacsv/train_data.csv',
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
adult.shape
adult
nadult = adult.dropna()
nadult.shape
testAdult = pd.read_csv('../input/adult-datacsv/test_data.csv',
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
testAdult
testAdult.shape
Xadult = nadult[["age","education.num","capital.gain", "capital.loss", "hours.per.week"]]
Yadult = nadult.income
XtestAdult = testAdult[["age","education.num","capital.gain", "capital.loss", "hours.per.week"]]
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import numpy as np
k=1
v=[]
K=[]
while k<=50:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(Xadult, Yadult)
    scores = cross_val_score(knn, Xadult, Yadult, cv=10)
    x=np.mean(scores)
    print(x)
    v.append(x)
    K.append(k)
    k+=1
print(np.amax(v),np.argmax(v))
vetor = pd.DataFrame(data = v)
plt.scatter(K, vetor)
plt.plot(K, vetor)
knn = KNeighborsClassifier(n_neighbors=25)
knn.fit(Xadult, Yadult)
scores = cross_val_score(knn, Xadult, Yadult, cv=10)
scores
np.mean(scores)
YtestPred = knn.predict(XtestAdult)
YtestPred
predicted = pd.DataFrame(data = YtestPred)
predicted[0].value_counts()
predicted[0].value_counts().plot(kind = 'bar')
result = np.vstack((testAdult["Id"],YtestPred)).T
result
