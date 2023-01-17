import pandas as pd
import sklearn
import matplotlib.pyplot as plt
adult = pd.read_csv("../input/pmradult/train_data.csv",na_values='?')
adult.head()
adult.shape
adult["income"].value_counts().plot(kind="bar")
adult.head()
adult = adult.dropna()
adult.shape
testadult=  pd.read_csv("../input/pmradult/test_data.csv",na_values='?')
testadult.shape
testadult = testadult.dropna()
testadult.shape
xadult= adult[['age','education.num','capital.gain','capital.loss','hours.per.week']]
yadult=adult.income
xtestadult=testadult[['age','education.num','capital.gain','capital.loss','hours.per.week']]

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=35)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(knn, xadult, yadult,cv=10)
scores
scores.mean()
knn.fit(xadult,yadult)
from sklearn.metrics import accuracy_score
pred=knn.predict(xtestadult)
pred.shape
pred
